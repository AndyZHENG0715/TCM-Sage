"""Standard-library, offline regression tests for release_limits; no app imports."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from release_limits import GateError, ReleaseGate

SECRET = "test-access-code-only"
HASH = hashlib.sha256(SECRET.encode()).hexdigest()
AUTH = "Bearer " + SECRET
IP = "198.51.100.23"


class GateTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / "vectorstore/release-limit-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.path = Path(self.temp.name) / "quota.sqlite"
        self.now = datetime(2026, 10, 24, tzinfo=timezone.utc).timestamp()

    def tearDown(self):
        resolved = Path(self.temp.name).resolve()
        expected_root = Path(__file__).resolve().parents[1] / "vectorstore/release-limit-tests"
        self.assertEqual(resolved.parent, expected_root.resolve())
        self.temp.cleanup()

    def gate(self, **limits):
        return ReleaseGate(self.path, HASH, enabled=True, limits=limits, clock=lambda: self.now)

    def denied(self, gate, status=429, auth=AUTH, ip=IP):
        with self.assertRaises(GateError) as error:
            gate.acquire(auth, ip)
        self.assertEqual(error.exception.status_code, status)
        self.assertNotIn(SECRET, str(error.exception))
        self.assertNotIn(IP, str(error.exception))

    def count(self):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT COUNT(*) FROM admissions").fetchone()[0]

    def test_no_code_and_wrong_code_charge_zero(self):
        gate = self.gate()
        self.denied(gate, 401, None)
        self.denied(gate, 401, "Bearer incorrect")
        self.assertEqual(self.count(), 0)

    def test_provider_failure_keeps_charge_and_releases_concurrency(self):
        gate = self.gate()
        with self.assertRaises(RuntimeError):
            with gate.acquire(AUTH, IP):
                raise RuntimeError("mock provider failure")
        self.assertEqual(self.count(), 1)
        with gate.acquire(AUTH, IP):
            pass
        self.assertEqual(self.count(), 2)

    def test_minute_rate_survives_new_instance(self):
        gate = self.gate()
        gate.acquire(AUTH, IP).release()
        gate.acquire(AUTH, IP).release()
        self.denied(self.gate())
        self.now += 61
        self.gate().acquire(AUTH, IP).release()

    def test_single_code_concurrency_and_idempotent_release(self):
        gate = self.gate()
        lease = gate.acquire(AUTH, IP)
        self.denied(self.gate(), ip="198.51.100.24")
        lease.release()
        lease.release()
        self.gate().acquire(AUTH, IP).release()

    def test_atomic_competing_admissions(self):
        def reserve(_):
            try:
                return self.gate().acquire(AUTH, IP)
            except GateError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(reserve, range(2)))
        leases = [result for result in results if result is not None]
        self.assertEqual(len(leases), 1)
        self.assertEqual(self.count(), 1)
        leases[0].release()

    def test_global_concurrency_across_distinct_codes(self):
        first = self.gate().acquire(AUTH, IP)
        second_secret = "other-test-code"
        third_secret = "third-test-code"
        second = ReleaseGate(self.path, hashlib.sha256(second_secret.encode()).hexdigest(),
                             enabled=True, clock=lambda: self.now)
        third = ReleaseGate(self.path, hashlib.sha256(third_secret.encode()).hexdigest(),
                            enabled=True, clock=lambda: self.now)
        lease = second.acquire("Bearer " + second_secret, "198.51.100.24")
        with self.assertRaises(GateError) as error:
            third.acquire("Bearer " + third_secret, "198.51.100.25")
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(self.count(), 2)
        first.release()
        lease.release()

    def test_interrupted_active_lease_is_not_automatically_forgiven(self):
        lease = self.gate().acquire(AUTH, IP)
        self.now += 86400
        self.denied(self.gate())
        lease.release()
        self.gate().acquire(AUTH, IP).release()

    def test_daily_global_limit_and_utc_reset(self):
        gate = self.gate(global_day=1)
        gate.acquire(AUTH, IP).release()
        self.now += 61
        self.denied(self.gate(global_day=1), ip="198.51.100.24")
        self.now += 86400
        self.gate(global_day=1).acquire(AUTH, IP).release()

    def test_monthly_global_limit_survives_day_and_new_instance(self):
        self.gate(global_month=1).acquire(AUTH, IP).release()
        self.now += 86400
        self.denied(self.gate(global_month=1))
        self.now = datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
        self.gate(global_month=1).acquire(AUTH, IP).release()

    def test_per_ip_daily_and_code_daily(self):
        self.gate(ip_day=1, code_day=2).acquire(AUTH, IP).release()
        self.now += 61
        self.denied(self.gate(ip_day=1, code_day=2))
        self.gate(ip_day=1, code_day=2).acquire(AUTH, "198.51.100.24").release()
        self.now += 61
        self.denied(self.gate(ip_day=1, code_day=2), ip="198.51.100.25")

    def test_public_read_limit(self):
        gate = self.gate(read_minute=1)
        gate.check_read(IP)
        with self.assertRaises(GateError) as error:
            gate.check_read(IP)
        self.assertEqual(error.exception.status_code, 429)
        self.now += 61
        gate.check_read(IP)

    def test_sensitive_values_not_in_db(self):
        self.gate().acquire(AUTH, IP).release()
        data = self.path.read_bytes()
        for sensitive in (SECRET, HASH, IP, AUTH):
            self.assertNotIn(sensitive.encode(), data)

    def test_db_unavailable_fails_closed(self):
        gate = ReleaseGate(Path(self.temp.name), HASH, enabled=True)
        with self.assertRaises(GateError) as error:
            gate.acquire(AUTH, IP)
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(str(error.exception), "Live quota ledger unavailable")

    def test_disabled_switch_is_fresh_and_compatibility_secret(self):
        with patch.dict(os.environ, {"LIVE_ENABLED": "1", "LIVE_ACCESS_CODE": SECRET,
                                   "RELEASE_LEDGER_PATH": str(self.path)}, clear=True):
            gate = ReleaseGate.from_env()
            gate.acquire(AUTH, IP).release()
            os.environ["LIVE_ENABLED"] = "0"
            self.denied(gate, 503)
            self.assertEqual(self.count(), 1)

    def test_default_disabled_no_ledger(self):
        with patch.dict(os.environ, {"RELEASE_LEDGER_PATH": str(self.path)}, clear=True):
            self.denied(ReleaseGate.from_env(), 503)
        self.assertFalse(self.path.exists())

    def test_limits_cannot_be_increased(self):
        gate = self.gate(ip_minute=999, global_concurrent=99)
        self.assertEqual(gate.limits["ip_minute"], 2)
        self.assertEqual(gate.limits["global_concurrent"], 2)
        with self.assertRaises(GateError):
            self.gate(global_month=-1)


if __name__ == "__main__":
    unittest.main()
