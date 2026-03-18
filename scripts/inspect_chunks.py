import json
from pathlib import Path

p = Path("data/processed/chunks.json")
if p.exists():
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
        if data and isinstance(data, list):
            print(json.dumps(data[0], indent=2, ensure_ascii=False))
        else:
            print("Data is not a list or empty")
else:
    print("File not found")
