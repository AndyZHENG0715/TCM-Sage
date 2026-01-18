"""
Subset KG extraction with tracking for incremental processing.
Tracks which chunk IDs have been processed so they can be excluded in later runs.
"""
import json
import os
import sys
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kg_extractor import extract_kg_batch, save_kg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CHUNKS_PATH = os.path.join(DATA_DIR, 'processed', 'chunks.json')
OUTPUT_DIR = os.path.join(DATA_DIR, 'graph')
PROCESSED_IDS_PATH = os.path.join(OUTPUT_DIR, 'processed_chunk_ids.json')
PARTIAL_ENTITIES_PATH = os.path.join(OUTPUT_DIR, 'entities_partial.json')

SUBSET_SIZE = 100  # Number of chunks to process in this run


def load_processed_ids():
    """Load set of already-processed chunk IDs."""
    if os.path.exists(PROCESSED_IDS_PATH):
        with open(PROCESSED_IDS_PATH, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()


def save_processed_ids(ids: set):
    """Save processed chunk IDs for future exclusion."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROCESSED_IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(ids), f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(ids)} processed chunk IDs to {PROCESSED_IDS_PATH}")


def load_partial_results():
    """Load existing partial extraction results."""
    if os.path.exists(PARTIAL_ENTITIES_PATH):
        with open(PARTIAL_ENTITIES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"entities": [], "relationships": [], "extraction_stats": {}}


def save_partial_results(kg_data: dict):
    """Save partial extraction results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PARTIAL_ENTITIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved partial results to {PARTIAL_ENTITIES_PATH}")


def main():
    print("=" * 60)
    print("🧠 TCM-Sage Subset KG Extraction (with Tracking)")
    print("=" * 60)
    
    # Load chunks
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)
    print(f"📖 Loaded {len(all_chunks)} total chunks")
    
    # Load already-processed IDs
    processed_ids = load_processed_ids()
    print(f"📋 Already processed: {len(processed_ids)} chunks")
    
    # Filter out already-processed chunks
    remaining_chunks = [c for c in all_chunks if c.get('id') not in processed_ids]
    print(f"📋 Remaining to process: {len(remaining_chunks)} chunks")
    
    if not remaining_chunks:
        print("✅ All chunks have been processed!")
        return
    
    # Take subset
    subset = remaining_chunks[:SUBSET_SIZE]
    print(f"\n🔍 Processing subset of {len(subset)} chunks...\n")
    
    # Extract KG
    result = extract_kg_batch(subset, model='qwen3:8b', num_ctx=4096)
    
    # Track processed IDs
    new_processed_ids = {c.get('id') for c in subset}
    all_processed_ids = processed_ids | new_processed_ids
    save_processed_ids(all_processed_ids)
    
    # Merge with existing partial results
    existing = load_partial_results()
    merged = {
        "entities": existing.get("entities", []) + result.get("entities", []),
        "relationships": existing.get("relationships", []) + result.get("relationships", []),
        "extraction_stats": {
            "total_chunks_processed": len(all_processed_ids),
            "last_run": datetime.now().isoformat(),
            "last_run_chunks": len(subset),
            "last_run_entities": len(result.get("entities", [])),
            "last_run_relationships": len(result.get("relationships", []))
        }
    }
    save_partial_results(merged)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Subset Extraction Complete")
    print("=" * 60)
    print(f"   Chunks processed this run: {len(subset)}")
    print(f"   Total chunks processed: {len(all_processed_ids)}/{len(all_chunks)}")
    print(f"   Entities extracted this run: {len(result.get('entities', []))}")
    print(f"   Relationships extracted this run: {len(result.get('relationships', []))}")
    print(f"\n   Partial results saved to: {PARTIAL_ENTITIES_PATH}")
    print(f"   Processed IDs saved to: {PROCESSED_IDS_PATH}")


if __name__ == "__main__":
    main()
