import urllib.request
import json
import sys

try:
    with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as response:
        if response.status == 200:
            print("OK")
            sys.exit(0)
        else:
            print(f"Status: {response.status}")
            sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
