"""
Durable KG extraction with incremental saving after each chunk.
Prevents data loss on cancellation by saving progress after every chunk.
"""
import json
import os
import sys
import signal
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kg_extractor import extract_kg_from_chunk, deduplicate_entities, deduplicate_relationships
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
CHUNKS_PATH = os.path.join(DATA_DIR, 'processed', 'chunks.json')
OUTPUT_DIR = os.path.join(DATA_DIR, 'graph')
PROCESSED_IDS_PATH = os.path.join(OUTPUT_DIR, 'processed_chunk_ids.json')
PARTIAL_ENTITIES_PATH = os.path.join(OUTPUT_DIR, 'entities_partial.json')

SUBSET_SIZE = 10000  # Process all remaining chunks
SAVE_EVERY = 5     # Save progress every N chunks


class GracefulExit:
    """Handle graceful exit on Ctrl+C."""
    should_stop = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, signum, frame):
        print("\n⚠️ Received interrupt signal. Saving progress and exiting...")
        self.should_stop = True


def load_processed_ids():
    if os.path.exists(PROCESSED_IDS_PATH):
        with open(PROCESSED_IDS_PATH, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()


def save_processed_ids(ids: set):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROCESSED_IDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(list(ids), f, ensure_ascii=False, indent=2)


def load_partial_results():
    if os.path.exists(PARTIAL_ENTITIES_PATH):
        with open(PARTIAL_ENTITIES_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"entities": [], "relationships": [], "extraction_stats": {}}


def save_partial_results(kg_data: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PARTIAL_ENTITIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("🧠 TCM-Sage Durable KG Extraction (with Incremental Saving)")
    print("=" * 60)
    
    graceful_exit = GracefulExit()
    
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
    print(f"\n🔍 Processing subset of {len(subset)} chunks...")
    print(f"   Progress saved every {SAVE_EVERY} chunks (Ctrl+C safe)\n")
    
    # Load existing results
    existing = load_partial_results()
    all_entities = existing.get("entities", [])
    all_relationships = existing.get("relationships", [])
    
    # Process chunk by chunk
    chunks_processed = 0
    entities_this_run = 0
    relationships_this_run = 0
    
    for chunk in tqdm(subset, desc="Extracting KG"):
        if graceful_exit.should_stop:
            break
        
        # Extract from single chunk
        result = extract_kg_from_chunk(
            chunk.get('content', ''),
            chunk.get('metadata', {}),
            model='qwen3:8b',
            num_ctx=4096
        )
        
        # Collect results
        new_entities = result.get('entities', [])
        new_relations = result.get('relationships', [])
        
        all_entities.extend(new_entities)
        all_relationships.extend(new_relations)
        
        entities_this_run += len(new_entities)
        relationships_this_run += len(new_relations)
        
        # Only mark as processed if extraction produced entities
        # This prevents false "100% complete" when Ollama fails
        if new_entities or new_relations:
            processed_ids.add(chunk.get('id'))
        chunks_processed += 1
        
        # Periodic save
        if chunks_processed % SAVE_EVERY == 0:
            save_processed_ids(processed_ids)
            save_partial_results({
                "entities": all_entities,
                "relationships": all_relationships,
                "extraction_stats": {
                    "total_chunks_processed": len(processed_ids),
                    "last_save": datetime.now().isoformat()
                }
            })
    
    # Deduplicate
    print("\n🔄 Deduplicating entities and relationships...")
    unique_entities = deduplicate_entities(all_entities)
    unique_relationships = deduplicate_relationships(all_relationships)
    
    # Final save
    final_data = {
        "entities": unique_entities,
        "relationships": unique_relationships,
        "extraction_stats": {
            "total_chunks_processed": len(processed_ids),
            "last_run": datetime.now().isoformat(),
            "last_run_chunks": chunks_processed,
            "last_run_entities": entities_this_run,
            "last_run_relationships": relationships_this_run,
            "unique_entities": len(unique_entities),
            "unique_relationships": len(unique_relationships)
        }
    }
    save_partial_results(final_data)
    save_processed_ids(processed_ids)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Extraction Complete")
    print("=" * 60)
    print(f"   Chunks processed this run: {chunks_processed}")
    print(f"   Total chunks processed: {len(processed_ids)}/{len(all_chunks)}")
    print(f"   Entities: {len(unique_entities)} (extracted: {entities_this_run})")
    print(f"   Relationships: {len(unique_relationships)} (extracted: {relationships_this_run})")
    
    if graceful_exit.should_stop:
        print("\n⚠️ Run was interrupted. Progress has been saved.")
        print("   Re-run the script to continue from where you left off.")


if __name__ == "__main__":
    main()
