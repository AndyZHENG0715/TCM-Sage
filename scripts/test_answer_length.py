# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false
import sys, os
sys.path.insert(0, 'src')
stdout_reconfigure = getattr(sys.stdout, 'reconfigure', None)
if stdout_reconfigure:
    stdout_reconfigure(encoding='utf-8')
os.chdir('D:/Dev/TCM-Sage')
from dotenv import load_dotenv
_ = load_dotenv()
from main import create_llm, DEFAULT_SYSTEM_PROMPT, build_prompt_template
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from embeddings import get_embedding_model

llm = create_llm('alibaba', 'qwen-turbo', 0.3)
vs = Chroma(persist_directory='vectorstore/chroma', embedding_function=get_embedding_model())
prompt_template = build_prompt_template(DEFAULT_SYSTEM_PROMPT)

question = "水蛭的性味是什么？出自哪本经典？主治什么？"

results = vs.similarity_search(question, k=5)
from main import format_docs_with_citations
context, citations = format_docs_with_citations(results)

chain = prompt_template | llm | StrOutputParser()
rag_answer = chain.invoke({"context": context, "question": question})

plain_answer_resp = llm.invoke(question)
plain_answer = plain_answer_resp.content if hasattr(plain_answer_resp, 'content') else str(plain_answer_resp)

print("=" * 60)
print("RAG ANSWER")
print("=" * 60)
print(rag_answer)
print()
print(f"Length: {len(rag_answer)} chars")
print(f"Has ##: {'YES' if '##' in rag_answer else 'NO'}")
print(f"Has **: {'YES' if '**' in rag_answer else 'NO'}")
print(f"Has - list: {'YES' if '- ' in rag_answer else 'NO'}")
print(f"Has numbered list: {'YES' if '1.' in rag_answer or '1.' in rag_answer else 'NO'}")
print(f"Has > blockquote: {'YES' if '> ' in rag_answer else 'NO'}")
print(f"Has table: {'YES' if '|' in rag_answer else 'NO'}")

print()
print("=" * 60)
print("PLAIN LLM ANSWER")
print("=" * 60)
print(plain_answer)
print()
print(f"Length: {len(plain_answer)} chars")
print(f"Has ##: {'YES' if '##' in plain_answer else 'NO'}")
print(f"Has **: {'YES' if '**' in plain_answer else 'NO'}")
print(f"Has - list: {'YES' if '- ' in plain_answer else 'NO'}")

print()
print("=" * 60)
print("COMPARISON")
print("=" * 60)
ratio = len(rag_answer) / len(plain_answer) if plain_answer else 0
print(f"RAG length: {len(rag_answer)}")
print(f"Plain length: {len(plain_answer)}")
print(f"Ratio (RAG/Plain): {ratio:.2f}")
print(f"{'RAG is SHORTER' if ratio < 0.8 else 'RAG is COMPARABLE' if ratio < 1.2 else 'RAG is LONGER'}")
