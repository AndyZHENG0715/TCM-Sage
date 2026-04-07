"""
TCM-Sage Centralized Configuration

This module consolidates configuration constants that were previously
duplicated across multiple files. Import from here instead of hardcoding.
"""
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "chroma"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.json"
GRAPH_DIR = DATA_DIR / "graph"
# SymMap-shaped KG (see scripts/import_symmap_kg.py); override with GRAPH_DATA_PATH in .env
GRAPH_DATA_PATH = GRAPH_DIR / "symmap" / "symmap_entities.json"
GRAPH_DATA_DEFAULT_RELATIVE = "data/graph/symmap/symmap_entities.json"

# ============================================================================
# Embedding Configuration
# ============================================================================
# Embeddings: DashScope text-embedding-v4 (configured in src/embeddings.py)

# ============================================================================
# LLM Configuration (defaults, can be overridden by .env)
# ============================================================================
DEFAULT_LLM_PROVIDER = "alibaba"
DEFAULT_LLM_MODEL = "qwen3:8b"  # For local Ollama
DEFAULT_LLM_TEMPERATURE = 0.1

# ============================================================================
# Retrieval Configuration
# ============================================================================
DEFAULT_RETRIEVAL_K = 5
DEFAULT_GRAPH_DEPTH = 2

# ============================================================================
# KG Extraction Configuration
# ============================================================================
KG_EXTRACTOR_MODEL = "qwen3:8b"
KG_EXTRACTOR_NUM_CTX = 4096
KG_SAVE_INTERVAL = 5  # Save every N chunks in durable extraction
