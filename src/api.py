"""
FastAPI backend for TCM-Sage.

This module exposes the RAG pipeline as a production-ready REST API
with Server-Sent Events (SSE) streaming support.

Deployment:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Environment Variables:
    See .env.example for required configuration.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from functools import lru_cache

# Ensure we can import from the existing CLI module
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from ui_backend import get_runtime_config, run_query_stream, PipelineConfig, get_shared_vectorstore

load_dotenv()

# =============================================================================
# FUTURE: Authentication Middleware
# =============================================================================
# When implementing authentication, uncomment and configure:
#
# from fastapi import Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
#
# security = HTTPBearer(auto_error=False)
#
# async def get_current_user(
#     credentials: HTTPAuthorizationCredentials = Depends(security)
# ) -> dict:
#     """
#     Validate JWT token and return user info.
#     
#     Returns:
#         dict with keys: 'id', 'email', 'tier' ('anonymous' | 'free' | 'premium')
#     """
#     if credentials is None:
#         return {"id": None, "email": None, "tier": "anonymous"}
#     
#     # TODO: Validate JWT token here
#     # token = credentials.credentials
#     # payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#     # return {"id": payload["sub"], "email": payload["email"], "tier": payload["tier"]}
#     
#     return {"id": None, "email": None, "tier": "anonymous"}
# =============================================================================

# =============================================================================
# FUTURE: Rate Limiting
# =============================================================================
# When implementing rate limiting, consider using slowapi or a Redis-based solution:
#
# from slowapi import Limiter
# from slowapi.util import get_remote_address
#
# RATE_LIMITS = {
#     "anonymous": "5/hour",
#     "free": "20/hour",
#     "premium": "1000/hour",  # Effectively unlimited
# }
#
# def get_rate_limit_key(request: Request) -> str:
#     """Generate rate limit key based on user tier."""
#     user = request.state.user  # Set by auth middleware
#     if user["tier"] == "premium":
#         return f"premium:{user['id']}"
#     elif user["tier"] == "free":
#         return f"free:{user['id']}"
#     else:
#         return f"anonymous:{get_remote_address(request)}"
#
# limiter = Limiter(key_func=get_rate_limit_key)
# =============================================================================

# =============================================================================
# FUTURE: Model Routing by Tier
# =============================================================================
# When implementing tiered model access:
#
# MODEL_TIERS = {
#     "anonymous": {"provider": "alibaba", "model": "qwen-turbo"},  # Cheaper
#     "free": {"provider": "alibaba", "model": "qwen-plus"},
#     "premium": {"provider": "openai", "model": "gpt-4o"},  # Best quality
# }
#
# def get_model_config(tier: str) -> dict:
#     return MODEL_TIERS.get(tier, MODEL_TIERS["anonymous"])
# =============================================================================


app = FastAPI(
    title="TCM-Sage API",
    description="Evidence-synthesis API for Traditional Chinese Medicine",
    version="1.0.0",
)

# CORS configuration for frontend access
# FUTURE: Restrict origins in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    question: str


class ConfigResponse(BaseModel):
    """Response body for the /config endpoint."""
    provider: str
    model: str | None
    informational_temperature: float
    prescriptive_temperature: float
    classifier_provider: str
    classifier_model: str | None
    verifier_provider: str
    verifier_model: str | None
    retrieval_k: int


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Get the current pipeline configuration.
    
    Useful for debugging and displaying settings in the UI.
    """
    try:
        config: PipelineConfig = get_runtime_config()
        return ConfigResponse(
            provider=config.provider,
            model=config.model,
            informational_temperature=config.informational_temperature,
            prescriptive_temperature=config.prescriptive_temperature,
            classifier_provider=config.classifier_provider,
            classifier_model=config.classifier_model,
            verifier_provider=config.verifier_provider,
            verifier_model=config.verifier_model,
            retrieval_k=config.retrieval_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")


def generate_sse_stream(question: str) -> Generator[str, None, None]:
    """
    Generate Server-Sent Events from the RAG pipeline stream.
    
    SSE Format:
        data: <chunk>\n\n           # For text chunks
        event: metadata\n
        data: <json>\n\n            # For final metadata
    """
    try:
        for item in run_query_stream(question):
            if isinstance(item, dict) and item.get("type") == "metadata":
                # Final metadata event
                import json
                yield f"event: metadata\ndata: {json.dumps(item)}\n\n"
            else:
                # Text chunk - escape newlines for SSE
                chunk = str(item).replace("\n", "\\n")
                yield f"data: {chunk}\n\n"
    except Exception as e:
        import json
        error_data = {"type": "error", "message": str(e)}
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    """
    Execute a query against the RAG pipeline with streaming response.
    
    Returns Server-Sent Events (SSE) stream:
    - `data:` events contain text chunks as they are generated
    - `event: metadata` contains the final response metadata including
      verification results, severity classification, and citations
    
    Example usage with fetch:
    ```javascript
    const response = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: '陰陽是什麼？' })
    });
    
    const reader = response.body.getReader();
    // Process SSE stream...
    ```
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # FUTURE: Add rate limiting check here
    # user = request.state.user
    # check_rate_limit(user)
    
    # FUTURE: Select model based on user tier
    # model_config = get_model_config(user["tier"])
    
    return StreamingResponse(
        generate_sse_stream(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/source/{chunk_id}/context")
async def get_chunk_context(chunk_id: str) -> Dict[str, Any]:
    """
    Get the full chapter context for a specific chunk.
    
    Args:
        chunk_id: The ID of the chunk to retrieve context for.
        
    Returns:
        JSON object with book, chapter, full text, and highlight offsets.
    """
    try:
        # 1. Initialize VectorStore (reuse shared instance)
        vectorstore = get_shared_vectorstore()
        
        # 2. Get chunk metadata from VectorStore
        # include=["metadatas"] is enough, we don't need embeddings
        result = vectorstore._collection.get(ids=[chunk_id], include=["metadatas"])
        
        if not result or not result["ids"]:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found in VectorStore")
            
        metadata = result["metadatas"][0]
        
        # Ingest.py stores: metadata={"book": book_name, "source": chapter_title, ...}
        book = metadata.get("book")
        chapter = metadata.get("source") # 'source' is the chapter title
        
        if not book or not chapter:
             # Fallback or strict error? 
             # If ingest changed, we might need to handle it.
             pass

        # 3. Load all chunks to reconstruct the chapter
        chunks_data = load_chunks_data()
        
        # 4. Filter and sort chunks for this chapter (check nested metadata)
        chapter_chunks = [
            c for c in chunks_data 
            if c.get("metadata", {}).get("book") == book and c.get("metadata", {}).get("source") == chapter
        ]
        
        if not chapter_chunks:
            # Should not happen if vectorstore has it, but maybe chunks.json is out of sync
            raise HTTPException(status_code=404, detail=f"Chapter chunks not found in data store for {book} - {chapter}")
            
        # Sort by chunk_index
        chapter_chunks.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        
        # 5. Construct full text and find offsets
        full_text = ""
        highlight_start = 0
        highlight_end = 0
        found_chunk = False
        
        for chunk in chapter_chunks:
            chunk_content = chunk.get("content", "")
            current_start = len(full_text)
            
            # Append content (assuming ingest preserves exact content, usually strict concatenation involves checking overlap
            # but for this simple RAG, we likely just concatenated or split. 
            # If chunks.json has 'content', we join them.
            # We might need a separator if they were split without overlap? 
            # Looking at chunks.json sample: "content": "<ç¯‡å>è¥å«ç”Ÿä¼šç¯‡ç¬¬åå…«\n\nå..."
            # It seems to be raw text.
            full_text += chunk_content
            
            if chunk.get("id") == chunk_id:
                highlight_start = current_start
                highlight_end = len(full_text)
                found_chunk = True
                
        if not found_chunk:
             # Fallback if ID matching failed (e.g. slight mismatch in ID format?)
             # But exact ID match is expected.
             pass

        return {
            "chunk_id": chunk_id,
            "book": book,
            "chapter": chapter,
            "chunk_index": metadata.get("chunk_index"),
            "full_chapter_text": full_text,
            "highlight_start": highlight_start,
            "highlight_end": highlight_end,
            "total_chunks_in_chapter": len(chapter_chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for data loading

@lru_cache(maxsize=1)
def load_chunks_data() -> list[dict]:
    """Load and cache the chunks.json file."""
    chunks_path = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
    if not chunks_path.exists():
        raise RuntimeError(f"Chunks data not found at {chunks_path}")
        
    import json
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
