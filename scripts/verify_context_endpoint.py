import json
import random
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

def test_endpoint():
    # 1. Get a valid chunk ID from chunks.json
    chunks_path = Path("data/processed/chunks.json")
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("Error: chunks.json is empty")
        return

    # Pick a random chunk
    chunk = random.choice(chunks)
    chunk_id = chunk.get("id")
    chunk_meta = chunk.get("metadata", {})
    expected_book = chunk_meta.get("book")
    expected_chapter = chunk_meta.get("source") # Ingest uses 'source' for chapter

    print(f"Testing with Chunk ID: {chunk_id}")
    try:
        print(f"Book: {expected_book}")
        print(f"Chapter: {expected_chapter}")
    except:
        print(f"Book: {repr(expected_book)}")
        print(f"Chapter: {repr(expected_chapter)}")
    
    # 2. Call the API
    encoded_chunk_id = urllib.parse.quote(chunk_id)
    url = f"http://localhost:8000/source/{encoded_chunk_id}/context"
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                print("\n✅ Success!")
                print(json.dumps(data, indent=2, ensure_ascii=True))
                
                # Basic validation
                assert data["chunk_id"] == chunk_id
                # API returns 'book' correctly now
                assert data["book"] == expected_book
                assert data["chapter"] == expected_chapter
                assert "full_chapter_text" in data
                assert data["highlight_start"] >= 0
                assert data["highlight_end"] > data["highlight_start"]
                
                print("\nValidation passed.")
            else:
                print(f"\n❌ Failed with status {response.status}")
                
    except urllib.error.HTTPError as e:
        print(f"\n❌ HTTP Error: {e.code} {e.reason}")
        print(e.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"\n❌ URL Error: {e.reason}. Is the server running?")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_endpoint()
