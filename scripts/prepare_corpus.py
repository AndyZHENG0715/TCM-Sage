"""
Prepare TCM corpus: convert 16 texts from temp_tcm_books (GB18030→UTF-8),
clean lfglib 灵枢, remove old files, place all 17 in data/source/.

Usage:
    venv\Scripts\python.exe scripts\prepare_corpus.py
"""

import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "source"
TEMP_DIR = ROOT / "data" / "source" / "temp_tcm_books"

# ── 17 texts to ingest ──────────────────────────────────────────────────────

TEXTS_FROM_TEMP = [
    # (source filename in temp_tcm_books, target filename in data/source/)
    ("437-黄帝内经素问.txt", "黄帝内经素问.txt"),
    ("457-伤寒论.txt", "伤寒论.txt"),
    ("499-金匮要略方论.txt", "金匮要略方论.txt"),
    ("000-神农本草经.txt", "神农本草经.txt"),
    ("421-八十一难经.txt", "八十一难经.txt"),
    ("301-针灸甲乙经.txt", "针灸甲乙经.txt"),
    ("013-本草纲目.txt", "本草纲目.txt"),
    ("532-备急千金要方.txt", "备急千金要方.txt"),
    ("614-脾胃论.txt", "脾胃论.txt"),
    ("232-内外伤辨.txt", "内外伤辨惑论.txt"),
    ("581-兰室秘藏.txt", "兰室秘藏.txt"),
    ("570-丹溪心法.txt", "丹溪心法.txt"),
    ("453-黄帝素问宣明论方.txt", "宣明论方.txt"),
    ("572-儒门事亲.txt", "儒门事亲.txt"),
    ("526-温病条辨.txt", "温病条辨.txt"),
    ("544-温热论.txt", "温热论.txt"),
]

# lfglib 灵枢 (already UTF-8, needs header/footer cleanup)
LINGSHU_SOURCE = ROOT / "黄帝内经灵枢 [流芳阁 lfglib.cn]-5d3a.txt"
LINGSHU_TARGET = "黄帝内经灵枢.txt"

# Old files to remove from data/source/
OLD_FILES = [
    "431-黄帝内经灵枢集注.txt",
    "437-黄帝内经素问.txt",
    "439-黄帝内经太素.txt",
]


def convert_gb18030_to_utf8(src: Path, dst: Path) -> int:
    """Convert a GB18030-encoded file to UTF-8. Returns char count."""
    raw = src.read_bytes()
    # Try GB18030 first (superset of GBK)
    text = raw.decode("gb18030", errors="replace")
    # Normalize line endings to \n
    text = text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    # Remove excessive blank lines (max 2 consecutive)
    cleaned = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    text = "\n".join(cleaned).strip() + "\n"
    dst.write_text(text, encoding="utf-8")
    return len(text)


def clean_lingshu(src: Path, dst: Path) -> int:
    """Clean lfglib 灵枢 file: strip header/footer ads."""
    text = src.read_text(encoding="utf-8")
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")

    # Strip lfglib header (first line is their branding)
    start = 0
    for i, line in enumerate(lines):
        if "流芳阁" in line or line.strip() == "":
            start = i + 1
        elif "书籍名称" in line:
            start = i + 1
        else:
            break

    # Strip lfglib footer (last lines have their branding)
    end = len(lines)
    for i in range(len(lines) - 1, max(0, len(lines) - 20), -1):
        if "流芳阁" in lines[i] or "---" in lines[i]:
            end = i
        elif lines[i].strip() == "":
            continue
        else:
            break

    cleaned_lines = lines[start:end]
    # Remove excessive blank lines
    result = []
    blank_count = 0
    for line in cleaned_lines:
        stripped = line.rstrip()
        if not stripped:
            blank_count += 1
            if blank_count <= 2:
                result.append("")
        else:
            blank_count = 0
            result.append(stripped)

    # Add metadata header
    header = "书名：黄帝内经灵枢\n来源：流芳阁 (lfglib.cn)\n\n"
    final = header + "\n".join(result).strip() + "\n"
    dst.write_text(final, encoding="utf-8")
    return len(final)


def main() -> None:
    print("=" * 60)
    print("TCM-Sage Corpus Preparation")
    print("=" * 60)

    # Step 1: Remove old files
    print("\n── Removing old files ──")
    for old in OLD_FILES:
        path = SOURCE_DIR / old
        if path.exists():
            path.unlink()
            print(f"  Removed: {old}")
        else:
            print(f"  Already gone: {old}")

    # Step 2: Convert 16 texts from temp_tcm_books
    print("\n── Converting GB18030 → UTF-8 ──")
    total_chars = 0
    for src_name, dst_name in TEXTS_FROM_TEMP:
        src = TEMP_DIR / src_name
        dst = SOURCE_DIR / dst_name
        if not src.exists():
            print(f"  ✗ NOT FOUND: {src_name}")
            continue
        chars = convert_gb18030_to_utf8(src, dst)
        total_chars += chars
        print(f"  ✓ {dst_name:24s} ({chars:>8,} chars)")

    # Step 3: Clean and copy 灵枢
    print("\n── Cleaning 灵枢 ──")
    if LINGSHU_SOURCE.exists():
        dst = SOURCE_DIR / LINGSHU_TARGET
        chars = clean_lingshu(LINGSHU_SOURCE, dst)
        total_chars += chars
        print(f"  ✓ {LINGSHU_TARGET:24s} ({chars:>8,} chars)")
    else:
        print(f"  ✗ NOT FOUND: {LINGSHU_SOURCE}")

    # Step 4: Summary
    print("\n── Final data/source/ contents ──")
    txt_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith(".txt") and not f.startswith(".")])
    for f in txt_files:
        size = os.path.getsize(SOURCE_DIR / f)
        print(f"  {f:36s} {size//1024:>6} KB")

    print(f"\n  Total texts: {len(txt_files)}")
    print(f"  Total chars: {total_chars:,}")
    print("\nDone! Ready for ingestion.")


if __name__ == "__main__":
    main()
