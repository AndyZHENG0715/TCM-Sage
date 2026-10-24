import { NextRequest } from "next/server";

// Vercel Hobby with Fluid compute supports up to 300 seconds.
export const runtime = "nodejs";
export const maxDuration = 300;
const UPSTREAM_TIMEOUT_MS = 240_000;
const INTERNAL_BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";

async function proxy(req: NextRequest, pathParts: string[]) {
  if (process.env.NODE_ENV === "production" && !process.env.BACKEND_URL && !process.env.NEXT_PUBLIC_BACKEND_URL) {
    return Response.json({ detail: "Live service is not configured. Recorded examples remain available." }, { status: 503, headers: { "Cache-Control": "no-store" } });
  }
  const targetUrl = `${INTERNAL_BACKEND_URL.replace(/\/+$/, "")}/${pathParts.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  const headers = new Headers(req.headers);
  for (const name of ["host", "connection", "content-length", "accept-encoding"]) headers.delete(name);
  const abort = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; abort.abort(); }, UPSTREAM_TIMEOUT_MS);
  const disconnect = () => abort.abort();
  req.signal.addEventListener("abort", disconnect, { once: true });
  if (req.signal.aborted) abort.abort();
  const cleanup = () => {
    clearTimeout(timer);
    req.signal.removeEventListener("abort", disconnect);
  };
  try {
    const requestInit: RequestInit & { duplex?: "half" } = {
      method: req.method, headers, signal: abort.signal, cache: "no-store", redirect: "error",
    };
    if (req.method !== "GET" && req.method !== "HEAD") {
      requestInit.body = req.body;
      requestInit.duplex = "half";
    }
    const response = await fetch(targetUrl, requestInit);
    const responseHeaders = new Headers(response.headers);
    for (const name of ["content-encoding", "content-length", "connection", "set-cookie"]) responseHeaders.delete(name);
    responseHeaders.set("Cache-Control", "no-store");
    const isSse = responseHeaders.get("content-type")?.includes("text/event-stream");
    if (isSse) responseHeaders.set("X-Accel-Buffering", "no");
    if (!response.body) {
      cleanup();
      return new Response(null, { status: response.status, headers: responseHeaders });
    }
    const reader = response.body.getReader();
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const { done, value } = await reader.read();
          if (done) { cleanup(); controller.close(); }
          else controller.enqueue(value);
        } catch {
          cleanup();
          // HTTP headers may already be sent. End SSE with a safe, parseable error.
          if (isSse) {
            controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ type: "error", message: "Live service unavailable. Please try again later." })}\n\n`));
            controller.close();
          } else controller.error(new Error("Backend response interrupted"));
        }
      },
      async cancel() {
        abort.abort();
        cleanup();
        try { await reader.cancel(); } catch { /* Disconnected upstream. */ }
      },
    });
    return new Response(body, { status: response.status, headers: responseHeaders });
  } catch {
    cleanup();
    return Response.json({ detail: timedOut ? "Live service timed out. Please try again later." : "Live service unavailable. Please try again later." }, { status: timedOut ? 504 : 502, headers: { "Cache-Control": "no-store" } });
  }
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  return proxy(req, (await params).path);
}
export async function POST(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  return proxy(req, (await params).path);
}
