import { NextRequest } from "next/server";

const INTERNAL_BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "http://127.0.0.1:8000";

function buildBackendUrl(pathParts: string[], search: string): string {
  const safePath = pathParts.join("/");
  const base = INTERNAL_BACKEND_URL.replace(/\/+$/, "");
  return `${base}/${safePath}${search}`;
}

async function proxy(req: NextRequest, pathParts: string[]) {
  const targetUrl = buildBackendUrl(pathParts, req.nextUrl.search);
  const headers = new Headers(req.headers);
  headers.delete("host");
  headers.delete("connection");
  headers.delete("content-length");
  headers.delete("accept-encoding");

  const requestInit: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers,
  };

  if (req.method !== "GET" && req.method !== "HEAD") {
    requestInit.body = req.body;
    // Required by Node fetch when streaming a request body.
    requestInit.duplex = "half";
  }

  const response = await fetch(targetUrl, requestInit);

  const responseHeaders = new Headers(response.headers);
  responseHeaders.delete("content-encoding");
  responseHeaders.delete("content-length");

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxy(req, path);
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxy(req, path);
}
