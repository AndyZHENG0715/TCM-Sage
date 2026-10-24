"""Lean public FastAPI entrypoint; historical presentation remains in api.py."""

from functools import lru_cache
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
load_dotenv(ROOT / '.env')

from release_limits import GateError, ReleaseGate
import release_runtime as runtime

app = FastAPI(title='TCM-Sage research prototype', version='1.0.0', docs_url=None, redoc_url=None)
origins = [v.strip() for v in os.getenv('ALLOWED_ORIGINS', '').split(',') if v.strip() and v.strip() != '*']
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                   allow_methods=['GET', 'POST'], allow_headers=['Authorization', 'Content-Type'])


class BodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message['type'] != 'http.request':
                return
            size += len(message.get('body', b''))
            if size > 16384:
                return await JSONResponse({'detail': 'Request body exceeds public limit'}, status_code=413)(scope, receive, send)
            messages.append(message)
            if not message.get('more_body', False):
                break

        async def bounded_receive():
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, bounded_receive, send)


app.add_middleware(BodyLimit)


@lru_cache(maxsize=1)
def gate():
    return ReleaseGate.from_env()


def client_ip(request):
    # Do not trust arbitrary client-supplied forwarding headers. Hosts may share a quota.
    return request.client.host if request.client else 'unknown'


def public_read(request):
    gate().check_read(client_ip(request))


@app.exception_handler(GateError)
async def quota_error(request, error):
    return JSONResponse({'detail': error.message}, status_code=error.status_code,
                        headers={'Retry-After': str(error.retry_after)} if error.retry_after else {})


@app.exception_handler(Exception)
async def unexpected_error(request, error):
    return JSONResponse({'detail': 'Research service unavailable; try a recorded example.'}, status_code=503)


class Query(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=runtime.MAX_QUESTION_CHARS)
    chat_history: list[dict] = Field(default_factory=list, max_length=6)
    settings: dict | None = None


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'research-prototype',
            'live_enabled': os.getenv('LIVE_ENABLED', '0') == '1'}


@app.get('/config')
def config(request: Request):
    public_read(request)
    return {'provider': 'openrouter', 'model': runtime.CHAT_MODEL,
            'informational_temperature': 0, 'prescriptive_temperature': 0,
            'classifier_follow_main': True, 'classifier_provider': 'openrouter', 'classifier_model': runtime.CHAT_MODEL,
            'verifier_follow_main': True, 'verifier_provider': 'openrouter', 'verifier_model': runtime.CHAT_MODEL,
            'retrieval_k': 6, 'hybrid_enabled': False, 'hybrid_available': True, 'graph_depth': 1,
            'graph_max_results': 30, 'research_only': True}


@app.post('/query')
def query(body: Query, request: Request):
    if not body.question.strip():
        raise HTTPException(400, 'Question cannot be empty')
    if body.settings:
        raise HTTPException(400, 'Public provider and retrieval settings are fixed')
    lease = gate().acquire(request.headers.get('authorization'), client_ip(request))

    def stream():
        try:
            for item in runtime.run_query(body.question):
                if isinstance(item, dict):
                    yield 'event: metadata\ndata: ' + json.dumps(item, ensure_ascii=False) + '\n\n'
                else:
                    yield 'data: ' + str(item).replace('\n', '\\n') + '\n\n'
        except Exception:
            yield 'event: error\ndata: ' + json.dumps({'type': 'error', 'message': 'Live research is temporarily unavailable. Try a recorded example.'}) + '\n\n'
        finally:
            lease.release()

    return StreamingResponse(stream(), media_type='text/event-stream',
                             background=BackgroundTask(lease.release),
                             headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


@app.get('/source/{chunk_id}/context')
def source(chunk_id: str, request: Request):
    public_read(request)
    if len(chunk_id) > 200:
        raise HTTPException(400, 'Invalid source ID')
    result = runtime.source_context(chunk_id)
    if result is None:
        raise HTTPException(404, 'Source not found')
    return result


@app.get('/books/{book_name}')
def book(book_name: str, request: Request):
    public_read(request)
    available = {p.stem: p for p in (ROOT / 'data/source').glob('*.txt')}
    if book_name not in available:
        raise HTTPException(404, 'Book not found')
    path = available[book_name]
    if path.stat().st_size > 2_000_000:
        raise HTTPException(413, 'Use the cited excerpt; this book exceeds the public response limit')
    return {'content': path.read_text(encoding='utf-8')}


@app.get('/graph/subgraph')
def graph_subgraph(request: Request, entity: str, hops: int = 1):
    public_read(request)
    if not 1 <= len(entity) <= 100 or not 1 <= hops <= 2:
        raise HTTPException(400, 'Graph request exceeds public bounds')
    return runtime.subgraph(entity, hops)


@app.get('/graph/search')
def graph_search(request: Request, q: str, limit: int = 20):
    public_read(request)
    if not 1 <= len(q) <= 100 or not 1 <= limit <= 20:
        raise HTTPException(400, 'Graph request exceeds public bounds')
    kg = runtime.graph()
    ids = kg.search_by_name(q)[:limit]
    return {'results': [{'id': cid, 'label': kg.graph.nodes[cid].get('name', cid),
                         'type': kg.graph.nodes[cid].get('type', 'Unknown')} for cid in ids]}


@app.api_route('/arena/{path:path}', methods=['GET', 'POST'])
def disabled_arena(path: str):
    raise HTTPException(403, 'Public Arena queries, submissions and raw statistics are disabled')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
