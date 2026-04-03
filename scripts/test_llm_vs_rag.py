import importlib
import re
import sys
from typing import Any, TypedDict, cast

_ = sys.path.insert(0, "src")
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    _ = stdout_reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_chroma import Chroma


class Rule(TypedDict):
    groups: list[list[str]]
    expected_books: list[str]


class RetrievalRow(TypedDict):
    score: float
    book: str
    chapter: str
    content: str


class SummaryRow(TypedDict):
    q: str
    plain: str
    rag: str
    assessment: str
    recommendation: str


_ = load_dotenv()

embeddings_module = importlib.import_module("embeddings")
main_module = importlib.import_module("main")
get_embedding_model = cast(Any, getattr(embeddings_module, "get_embedding_model"))
create_llm = cast(Any, getattr(main_module, "create_llm"))

QUESTIONS: list[str] = [
    "水蛭的性味是什么？出自哪本经典？",
    "蛇床子有哪些配伍禁忌？恶什么药？",
    "独活寄生汤中独活的用量是多少两？",
    "四逆散的组成是什么？",
    "马兜铃有什么毒性？",
    "丹溪心法中湿痰用什么药？热痰用什么药？",
    "当归四逆汤由哪些药物组成？",
    "附子在神农本草经中的性味记载是什么？",
    "温病条辨中银翘散的组成有哪些药物？",
    "脾胃论的作者是谁？成书于什么年代？",
]


RULES: list[Rule] = [
    {
        "groups": [["咸"], ["平"], ["神农本草经", "本草经"]],
        "expected_books": ["神农本草经"],
    },
    {
        "groups": [["蛇床子"], ["恶"], ["牡丹", "巴豆", "贝母"]],
        "expected_books": ["神农本草经", "本草"],
    },
    {
        "groups": [["独活"], ["三两", "3两"]],
        "expected_books": ["备急千金要方", "千金要方"],
    },
    {
        "groups": [["柴胡"], ["枳实"], ["芍药", "白芍"], ["甘草"]],
        "expected_books": ["伤寒论"],
    },
    {
        "groups": [["马兜铃酸"], ["肾", "肾毒", "肾损伤"], ["致癌", "癌"]],
        "expected_books": ["本草", "药典"],
    },
    {
        "groups": [
            ["湿痰"],
            ["热痰"],
            ["半夏", "二陈", "苍术", "白术", "茯苓"],
            ["黄芩", "黄连", "瓜蒌", "青黛"],
        ],
        "expected_books": ["丹溪心法"],
    },
    {
        "groups": [["当归"], ["桂枝"], ["芍药", "白芍"], ["细辛"], ["通草"], ["甘草"], ["大枣"]],
        "expected_books": ["伤寒论", "金匮要略"],
    },
    {
        "groups": [["附子"], ["辛"], ["温", "热"], ["毒", "大毒"], ["神农本草经", "本草经"]],
        "expected_books": ["神农本草经"],
    },
    {
        "groups": [["银花", "金银花"], ["连翘"], ["桔梗"], ["薄荷"], ["竹叶"], ["甘草"], ["荆芥", "荆芥穗"], ["淡豆豉"], ["牛蒡子"], ["芦根"]],
        "expected_books": ["温病条辨"],
    },
    {
        "groups": [["李杲", "李东垣"], ["金", "金代", "元初"]],
        "expected_books": ["脾胃论"],
    },
]


def get_text(resp: Any) -> str:
    return resp.content if hasattr(resp, "content") else str(resp)


def clip(text: str, n: int = 200) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:n]


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def score_answer(answer: str, groups: list[list[str]]) -> float:
    if not answer:
        return 0.0
    hit = 0
    for group in groups:
        if any(k in answer for k in group):
            hit += 1
    return hit / len(groups) if groups else 0.0


def retrieval_hit(retrieved_books: list[str], expected_books: list[str]) -> bool:
    if not expected_books:
        return True
    if not retrieved_books:
        return False
    for b in retrieved_books:
        for expected in expected_books:
            if expected in b or b in expected:
                return True
    return False


def build_rag_prompt(question: str, retrieval_rows: list[RetrievalRow]) -> str:
    context_blocks: list[str] = []
    for idx, row in enumerate(retrieval_rows, start=1):
        context_blocks.append(
            f"[{idx}] 书名: {row['book']} | 章节: {row['chapter']}\n{row['content']}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "你是中医经典文献问答助手。\n"
        "请严格依据给定资料回答；若资料不足，请明确说“资料不足，无法确定”。\n"
        "不要编造来源、剂量或配伍。\n\n"
        f"参考资料:\n{context}\n\n"
        f"问题: {question}\n"
        "请给出简洁结论，并说明对应依据。"
    )


def assess(plain_score: float, rag_score: float, rag_retrieval_ok: bool) -> tuple[str, str]:
    if rag_score >= 0.5 and rag_score - plain_score >= 0.2 and rag_retrieval_ok:
        return f"RAG更准确（plain={plain_score:.2f}, rag={rag_score:.2f}, 检索命中）", "KEEP"
    if rag_score < 0.5 and plain_score < 0.5:
        return f"两者都不稳定（plain={plain_score:.2f}, rag={rag_score:.2f}）", "DROP"
    if rag_score <= plain_score:
        return f"RAG无明显优势（plain={plain_score:.2f}, rag={rag_score:.2f}）", "DROP"
    return f"RAG略优（plain={plain_score:.2f}, rag={rag_score:.2f}）", "KEEP"


def main() -> None:
    print("=== Init ===")
    llm: Any = create_llm("alibaba", "qwen-turbo", 0.3)
    vs = Chroma(
        persist_directory="vectorstore/chroma",
        embedding_function=get_embedding_model(),
    )

    rows: list[SummaryRow] = []

    for i, q in enumerate(QUESTIONS):
        print(f"\n{'=' * 80}")
        print(f"Q{i + 1}: {q}")

        print(f"=== PLAIN LLM Q{i + 1} ===")
        plain_resp: Any = llm.invoke(q)
        plain_text = get_text(plain_resp).strip()
        print(f"Q: {q}")
        print(f"A: {plain_text[:300]}")

        print(f"\n=== RAG RETRIEVAL Q{i + 1} ===")
        raw_results: list[Any] = vs.similarity_search_with_score(q, k=2)
        results: list[tuple[Any, float]] = [
            (item[0], float(item[1])) for item in raw_results if isinstance(item, tuple) and len(item) == 2
        ]
        retrieval_rows: list[RetrievalRow] = []
        retrieved_books: list[str] = []
        for r, score in results:
            metadata_obj = getattr(r, "metadata", None)
            metadata: dict[str, Any] = metadata_obj if isinstance(metadata_obj, dict) else {}
            book = str(metadata.get("book", "?"))
            chapter = str(metadata.get("source", "?"))
            content = str(getattr(r, "page_content", ""))
            retrieved_books.append(book)
            retrieval_row: RetrievalRow = {
                "score": score,
                "book": book,
                "chapter": chapter[:25],
                "content": content,
            }
            retrieval_rows.append(retrieval_row)
            print(f"Q: {q}")
            print(f"  Score: {score:.3f} | {book} | {chapter[:25]}")
            print(f"  Content: {content[:150]}")

        rag_prompt = build_rag_prompt(q, retrieval_rows)
        rag_resp: Any = llm.invoke(rag_prompt)
        rag_text = get_text(rag_resp).strip()

        print(f"\n=== RAG ANSWER Q{i + 1} ===")
        print(f"A: {rag_text[:300]}")

        rule = RULES[i]
        plain_score = score_answer(plain_text, rule["groups"])
        rag_score = score_answer(rag_text, rule["groups"])
        rag_retrieval_ok = retrieval_hit(retrieved_books, rule["expected_books"])
        assessment, recommendation = assess(plain_score, rag_score, rag_retrieval_ok)

        summary_row: SummaryRow = {
            "q": q,
            "plain": clip(plain_text, 200),
            "rag": clip(rag_text, 200),
            "assessment": assessment,
            "recommendation": recommendation,
        }
        rows.append(summary_row)

    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY TABLE")
    print("| # | Question | Plain LLM (first 200 chars) | RAG (first 200 chars) | Factual accuracy assessment | Recommendation |")
    print("|---|---|---|---|---|---|")
    for i, row in enumerate(rows, start=1):
        print(f"| {i} | {markdown_escape(row['q'])} | {markdown_escape(row['plain'])} | {markdown_escape(row['rag'])} | {markdown_escape(row['assessment'])} | {row['recommendation']} |")

    keep_count = sum(1 for row in rows if row["recommendation"] == "KEEP")
    print(f"\nRAG wins (KEEP): {keep_count}/{len(rows)}")


if __name__ == "__main__":
    main()
