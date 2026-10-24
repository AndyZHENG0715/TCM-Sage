import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";

function loadProxy(fetchImpl, env = { BACKEND_URL: "https://private-backend.example" }) {
  const runtimeModule = { exports: {} };
  let timer;
  const source = ts.transpileModule(readFileSync(new URL("../app/api/backend/[...path]/route.ts", import.meta.url), "utf8"), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  vm.runInNewContext(source, { module: runtimeModule, exports: runtimeModule.exports, require: () => ({}), process: { env }, Headers, Response, ReadableStream, TextEncoder, AbortController, encodeURIComponent, fetch: fetchImpl, setTimeout: fn => { timer = fn; return 1; }, clearTimeout: () => {} });
  return { ...runtimeModule.exports, expire: () => timer() };
}
function request() {
  return { method: "POST", headers: new Headers({ authorization: "Bearer test-access", host: "frontend.example" }), signal: new AbortController().signal, body: null, nextUrl: { search: "" } };
}
const params = { params: Promise.resolve({ path: ["query", "stream"] }) };
test("proxy preserves incremental SSE, authorization and bounded configuration", async () => {
  let stream;
  const app = loadProxy(async (url, init) => {
    assert.equal(url, "https://private-backend.example/query/stream");
    assert.equal(init.headers.get("authorization"), "Bearer test-access");
    assert.equal(init.headers.has("host"), false);
    assert.equal(init.redirect, "error");
    return new Response(new ReadableStream({ start(c) { stream = c; c.enqueue(new TextEncoder().encode("data: first\n\n")); } }), { headers: { "content-type": "text/event-stream" } });
  });
  assert.equal(app.maxDuration, 300);
  const response = await app.POST(request(), params);
  const reader = response.body.getReader();
  assert.equal(new TextDecoder().decode((await reader.read()).value), "data: first\n\n");
  stream.enqueue(new TextEncoder().encode("data: second\n\n"));
  stream.close();
  assert.equal(new TextDecoder().decode((await reader.read()).value), "data: second\n\n");
  assert.equal((await reader.read()).done, true);
  assert.equal(response.headers.get("cache-control"), "no-store");
});
test("proxy sanitizes fetch errors and reports timeout without leaking upstream details", async () => {
  const app = loadProxy(async () => { throw new Error("https://private-backend.example secret"); });
  const response = await app.POST(request(), params);
  assert.equal(response.status, 502);
  assert.doesNotMatch(await response.text(), /private-backend|secret/);
  const timeout = loadProxy((_url, init) => new Promise((_resolve, reject) => init.signal.addEventListener("abort", () => reject(new Error("secret")))));
  const pending = timeout.POST(request(), params);
  await new Promise(resolve => setImmediate(resolve));
  timeout.expire();
  assert.equal((await pending).status, 504);
});
test("stream failure emits safe SSE error and cancelling aborts upstream", async () => {
  let upstreamSignal;
  const app = loadProxy(async (_url, init) => { upstreamSignal = init.signal; return new Response(new ReadableStream({ pull() { throw new Error("secret"); } }), { headers: { "content-type": "text/event-stream" } }); });
  const response = await app.POST(request(), params);
  assert.match(await response.text(), /"type":"error"/);
  const cancel = loadProxy(async (_url, init) => { upstreamSignal = init.signal; return new Response(new ReadableStream({})); });
  const pending = await cancel.POST(request(), params);
  await pending.body.cancel();
  assert.equal(upstreamSignal.aborted, true);
});


test("production without a backend fails closed before any network request", async () => {
  const app = loadProxy(() => { throw new Error("must not fetch localhost"); }, { NODE_ENV: "production" });
  const response = await app.POST(request(), params);
  assert.equal(response.status, 503);
  assert.doesNotMatch(await response.text(), /localhost|127\.0\.0\.1/);
});
