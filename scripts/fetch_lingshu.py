"""
Fetch 灵枢经 (81 chapters) from ctext.org and save as clean UTF-8 text.

Source: 《四部丛刊初编》本《黄帝素问灵枢经》
This is the most authoritative base text for 灵枢.

Usage:
    venv\Scripts\python.exe scripts/fetch_lingshu.py
"""

import html
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# All 81 chapters of 灵枢经 with Chinese titles
CHAPTERS = [
    ("jiu-zhen-shi-er-yuan", "九针十二原"),
    ("ben-shu", "本输"),
    ("xiao-zhen-jie", "小针解"),
    ("xie-qi-cang-fu-bing-xing", "邪气藏府病形"),
    ("gen-jie", "根结"),
    ("shou-yao-gang-rou", "寿夭刚柔"),
    ("guan-zhen", "官针"),
    ("ben-shen", "本神"),
    ("zhong-shi", "终始"),
    ("jing-mai", "经脉"),
    ("jing-bie", "经别"),
    ("jing-shui", "经水"),
    ("jing-jin", "经筋"),
    ("gu-du", "骨度"),
    ("wu-shi-ying", "五十营"),
    ("ying-qi", "营气"),
    ("mo-du", "脉度"),
    ("ying-wei-sheng-hui", "营卫生会"),
    ("si-shi-qi", "四时气"),
    ("wu-xie", "五邪"),
    ("han-re-bing", "寒热病"),
    ("lai-kuang-bing", "癞狂病"),
    ("re-bing", "热病"),
    ("jue-bing", "厥病"),
    ("bing-ben", "病本"),
    ("za-bing", "杂病"),
    ("zhou-bi", "周痹"),
    ("kou-wen", "口问"),
    ("shi-chuan", "师传"),
    ("jue-qi", "决气"),
    ("chang-wei", "肠胃"),
    ("ping-ren-jue-gu", "平人绝谷"),
    ("hai-lun", "海论"),
    ("wu-luan", "五乱"),
    ("zhang-lun", "胀论"),
    ("wu-long-jin-ye-bie", "五癃津液别"),
    ("wu-yue-wu-shi", "五阅五使"),
    ("ni-shun-fei-shou", "逆顺肥瘦"),
    ("xue-luo-lun", "血络论"),
    ("yin-yang-qing-zhuo", "阴阳清浊"),
    ("yin-yang-xi-ri-yue", "阴阳系日月"),
    ("bing-chuan", "病传"),
    ("yin-xie-fa-meng", "淫邪发梦"),
    ("shun-qi-yi-ri-fen-wei-si-shi", "顺气一日分为四时"),
    ("wai-chuai", "外揣"),
    ("wu-bian", "五变"),
    ("ben-cang", "本藏"),
    ("jin-fu", "禁服"),
    ("wu-se", "五色"),
    ("lun-yong", "论勇"),
    ("bei-shu", "背腧"),
    ("wei-qi", "卫气"),
    ("lun-tong", "论痛"),
    ("tian-nian", "天年"),
    ("ni-shun2", "逆顺"),
    ("wu-wei", "五味"),
    ("shui-zhang", "水胀"),
    ("zei-feng", "贼风"),
    ("wei-qi-shi-chang", "卫气失常"),
    ("yu-ban", "玉版"),
    ("wu-jin", "五禁"),
    ("dong-shu", "动输"),
    ("wu-wei-lun", "五味论"),
    ("yin-yang-er-shi-wu-ren", "阴阳二十五人"),
    ("wu-yin-wu-wei", "五音五味"),
    ("bai-bing-shi-sheng", "百病始生"),
    ("xing-zhen", "行针"),
    ("shang-ge", "上膈"),
    ("you-hui-wu-yan", "忧恚无言"),
    ("han-re2", "寒热"),
    ("xie-ke", "邪客"),
    ("tong-tian", "通天"),
    ("guan-neng", "官能"),
    ("lun-ji-zhen-chi", "论疾诊尺"),
    ("ci-jie-zhen-xie", "刺节真邪"),
    ("wei-qi-xing", "卫气行"),
    ("jiu-gong-ba-feng", "九宫八风"),
    ("jiu-zhen-lun", "九针论"),
    ("sui-lu-lun", "岁露论"),
    ("da-huo-lun", "大惑论"),
    ("yong-ju", "痈疽"),
]

BASE_URL = "https://ctext.org/huangdi-neijing"
HEADERS = {"User-Agent": "Mozilla/5.0 (TCM-Sage Academic Project)"}
OUTPUT_DIR = Path("data/source")


def fetch_chapter(slug: str) -> str:
    """Fetch a single chapter and extract classical text paragraphs."""
    url = f"{BASE_URL}/{slug}/zhs"
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=30)
    raw = resp.read().decode("utf-8", errors="replace")

    # Decode HTML entities
    text = html.unescape(raw)

    # Extract Chinese text paragraphs — ctext.org uses numbered paragraphs
    # Pattern: lines starting with a number followed by Chinese text
    # The actual content appears after the chapter title section
    paragraphs = []

    # Find content between the chapter markers
    # ctext.org format: numbered paragraphs like "1  ..." with Chinese text
    # The text appears in patterns like: \n数字\s+打开字典...\n中文内容\n
    lines = text.split("\n")
    capture = False
    current_para = []

    for line in lines:
        stripped = line.strip()

        # Skip navigation, English translations, metadata
        if not stripped:
            continue
        if "打开字典" in stripped or "显示影印本" in stripped or "显示相似段落" in stripped:
            # This is a paragraph header line — save previous and start new
            if current_para:
                paragraphs.append("".join(current_para))
                current_para = []
            capture = True
            continue

        if capture:
            # Skip English translations and metadata lines
            if re.match(r"^[A-Z]", stripped):  # English translation line
                continue
            if stripped.startswith("URN:"):
                capture = False
                continue
            if "电子底本" in stripped or "喜欢我们的网站" in stripped:
                capture = False
                continue
            if "电子图书馆" in stripped and "底本" in stripped:
                continue
            if stripped.startswith("1.") and "旧脱" in stripped:
                # Textual note, skip
                continue

            # Check if line is mostly Chinese characters
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
            if chinese_chars > len(stripped) * 0.3:
                current_para.append(stripped)

    if current_para:
        paragraphs.append("".join(current_para))

    return "\n".join(paragraphs)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "灵枢经.txt"

    all_text = []
    all_text.append("书名：灵枢经")
    all_text.append("底本：《四部丛刊初编》本《黄帝素问灵枢经》")
    all_text.append("来源：中国哲学书电子化计划 (ctext.org)")
    all_text.append("")

    total = len(CHAPTERS)
    failed = []

    for i, (slug, title) in enumerate(CHAPTERS, 1):
        print(f"[{i:2d}/{total}] Fetching {title} ({slug})...", end=" ", flush=True)
        try:
            content = fetch_chapter(slug)
            if content.strip():
                all_text.append(f"\n{title}\n")
                all_text.append(content)
                print(f"OK ({len(content)} chars)")
            else:
                print("EMPTY — no content extracted")
                failed.append((slug, title, "empty"))
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((slug, title, str(e)))

        # Be polite — 1 second between requests
        if i < total:
            time.sleep(1)

    # Write output
    final_text = "\n".join(all_text)
    output_file.write_text(final_text, encoding="utf-8")
    print(f"\nDone! Written to {output_file}")
    print(f"Total size: {len(final_text):,} characters")
    print(f"Chapters: {total - len(failed)}/{total} succeeded")

    if failed:
        print(f"\nFailed chapters ({len(failed)}):")
        for slug, title, err in failed:
            print(f"  {title} ({slug}): {err}")


if __name__ == "__main__":
    main()
