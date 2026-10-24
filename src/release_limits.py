"""Atomic count quotas for a fixed-free-provider release; no monetary accounting.

Only successful admission writes a charged reservation. Release clears concurrency,
never refunds counts. Persistence requires the SAME persistent SQLite file; an
ephemeral host filesystem cannot preserve quotas across replacement instances.
Interrupted leases remain active and deny access until an operator reconciles them.
Never use this ledger to claim a dollar ceiling for arbitrary or paid providers.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
import time


class GateError(RuntimeError):
    def __init__(self, status_code: int, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


class Lease:
    """Use in a finally block or with statement, including provider failures."""

    def __init__(self, gate: "ReleaseGate", lease_id: str):
        self._gate = gate
        self._id = lease_id
        self._released = False

    def release(self):
        if not self._released:
            self._gate._release(self._id)
            self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class ReleaseGate:
    DEFAULTS = {"ip_minute": 2, "ip_day": 10, "code_minute": 2,
                "code_day": 20, "global_day": 50, "global_month": 500,
                "global_concurrent": 2, "code_concurrent": 1, "read_minute": 60}

    def __init__(self, ledger_path, code_hash: str = "", *, enabled=False,
                 limits=None, clock=None, refresh_enabled=False):
        self.ledger_path = Path(ledger_path)
        self.code_hash = code_hash.lower()
        self.enabled = enabled
        self.refresh_enabled = refresh_enabled
        self.clock = clock or time.time
        self.limits = dict(self.DEFAULTS)
        for key, value in (limits or {}).items():
            if key not in self.DEFAULTS or type(value) is not int or value < 0:
                raise GateError(503, "Live quota configuration unavailable")
            self.limits[key] = min(value, self.DEFAULTS[key])
        if self.code_hash and (len(self.code_hash) != 64 or
                               any(c not in "0123456789abcdef" for c in self.code_hash)):
            raise GateError(503, "Live access configuration unavailable")

    @classmethod
    def from_env(cls):
        expected = os.environ.get("LIVE_ACCESS_CODE_HASH", "").strip()
        if not expected and os.environ.get("LIVE_ACCESS_CODE"):
            expected = hashlib.sha256(os.environ["LIVE_ACCESS_CODE"].encode()).hexdigest()
        limits = {}
        try:
            for name in cls.DEFAULTS:
                raw = os.environ.get("RELEASE_" + name.upper())
                if raw is not None:
                    limits[name] = int(raw)
        except ValueError:
            raise GateError(503, "Live quota configuration unavailable") from None
        default_path = Path(__file__).resolve().parents[1] / "vectorstore/release-quota.sqlite"
        return cls(os.environ.get("RELEASE_LEDGER_PATH", str(default_path)), expected,
                   enabled=False, refresh_enabled=True, limits=limits)

    def _private_id(self, kind: str, value: str):
        # The configured hash remains server-side; raw IPs/codes never enter SQLite.
        return hmac.new((self.code_hash or "disabled").encode(),
                        (kind + ":" + value).encode(), hashlib.sha256).hexdigest()

    @contextmanager
    def _transaction(self):
        connection = None
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.ledger_path), timeout=3, isolation_level=None)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE IF NOT EXISTS admissions (lease TEXT PRIMARY KEY, ts REAL NOT NULL, ip TEXT NOT NULL, code TEXT NOT NULL, active INTEGER NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS read_hits (ts REAL NOT NULL, ip TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS admissions_ts ON admissions(ts)")
            connection.execute("CREATE INDEX IF NOT EXISTS read_hits_ts ON read_hits(ts)")
            yield connection
            connection.commit()
        except GateError:
            if connection is not None:
                connection.rollback()
            raise
        except (sqlite3.Error, OSError, ValueError):
            if connection is not None:
                connection.rollback()
            raise GateError(503, "Live quota ledger unavailable") from None
        finally:
            if connection is not None:
                connection.close()

    def check_read(self, client_ip: str):
        """Call on public read routes and denied attempts as well as query routes."""
        now = float(self.clock())
        identity = self._private_id("ip", str(client_ip))
        with self._transaction() as db:
            db.execute("DELETE FROM read_hits WHERE ts <= ?", (now - 60,))
            count = db.execute("SELECT COUNT(*) FROM read_hits WHERE ip=? AND ts>?",
                               (identity, now - 60)).fetchone()[0]
            if count >= self.limits["read_minute"]:
                raise GateError(429, "Request rate limit reached", 60)
            db.execute("INSERT INTO read_hits VALUES (?,?)", (now, identity))

    def acquire(self, authorization: str | None, client_ip: str):
        live = (os.environ.get("LIVE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
                if self.refresh_enabled else self.enabled)
        if not live:
            raise GateError(503, "Live access is disabled")
        if not self.code_hash:
            raise GateError(503, "Live access configuration unavailable")
        self.check_read(client_ip)
        prefix, _, token = (authorization or "").partition(" ")
        supplied = hashlib.sha256(token.encode()).hexdigest()
        if prefix.lower() != "bearer" or not token or not hmac.compare_digest(supplied, self.code_hash):
            raise GateError(401, "Valid live access code required")
        now = float(self.clock())
        moment = datetime.fromtimestamp(now, timezone.utc)
        day_start = moment.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        month_start = moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        ip = self._private_id("ip", str(client_ip))
        code = self._private_id("code", "configured-code")
        with self._transaction() as db:
            checks = [
                ("ip_minute", "ip=? AND ts>?", (ip, now - 60), 60),
                ("code_minute", "code=? AND ts>?", (code, now - 60), 60),
                ("ip_day", "ip=? AND ts>=?", (ip, day_start), 86400),
                ("code_day", "code=? AND ts>=?", (code, day_start), 86400),
                ("global_day", "ts>=?", (day_start,), 86400),
                ("global_month", "ts>=?", (month_start,), 2678400),
                ("global_concurrent", "active=1", (), 1),
                ("code_concurrent", "code=? AND active=1", (code,), 1),
            ]
            for limit, where, values, retry in checks:
                count = db.execute("SELECT COUNT(*) FROM admissions WHERE " + where, values).fetchone()[0]
                if count >= self.limits[limit]:
                    raise GateError(429, "Live usage limit reached", retry)
            lease_id = secrets.token_hex(24)
            db.execute("INSERT INTO admissions VALUES (?,?,?,?,1)", (lease_id, now, ip, code))
        return Lease(self, lease_id)

    def _release(self, lease_id):
        with self._transaction() as db:
            db.execute("UPDATE admissions SET active=0 WHERE lease=?", (lease_id,))
