# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false
import sys, os
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
os.chdir('D:/Dev/TCM-Sage')
from dotenv import load_dotenv
_ = load_dotenv()
from main import create_llm, DEFAULT_SYSTEM_PROMPT

llm = create_llm('alibaba', 'qwen-turbo', 0.3)

# Same source content, two formats
plain_context = """[1] 水蛭
内容：味咸，平。主逐恶血、瘀血、月闭，破血瘕积聚，无子，利水道。生池泽。
名医曰：一名蛭，一名至掌。生雷泽。五月、六月采，曝干。"""

formatted_context = """### 来源 [1]：《神农本草经》· 水蛭

> **性味**：味咸，平
> **主治**：逐恶血、瘀血、月闭，破血瘕积聚，无子，利水道
> **产地**：生池泽
> **别名**：蛭、至掌（《名医》）
> **采收**：五月、六月采，曝干"""

question = "水蛭的性味是什么？出自哪本经典？主治什么？"

system = DEFAULT_SYSTEM_PROMPT

# Test A: Plain context
print("=" * 60)
print("TEST A: PLAIN TEXT CONTEXT")
print("=" * 60)
prompt_a = f"{system}\n\n以下为检索到的参考资料，供回答时引用：\n\n{plain_context}\n\n---\n\n用户提问：{question}\n\n请基于以上资料和自身知识，以结构清晰的方式回答。使用行内 [N] 标注引用来源，勿在末尾添加参考文献列表。"
resp_a = llm.invoke(prompt_a)
text_a = resp_a.content if hasattr(resp_a, 'content') else str(resp_a)
print(text_a[:600])
print()

# Test B: Formatted context
print("=" * 60)
print("TEST B: MARKDOWN FORMATTED CONTEXT")
print("=" * 60)
prompt_b = f"{system}\n\n以下为检索到的参考资料，供回答时引用：\n\n{formatted_context}\n\n---\n\n用户提问：{question}\n\n请基于以上资料和自身知识，以结构清晰的方式回答。使用行内 [N] 标注引用来源，勿在末尾添加参考文献列表。"
resp_b = llm.invoke(prompt_b)
text_b = resp_b.content if hasattr(resp_b, 'content') else str(resp_b)
print(text_b[:600])
print()

# Compare
print("=" * 60)
print("COMPARISON")
print("=" * 60)
has_heading_a = '##' in text_a or '**' in text_a
has_heading_b = '##' in text_b or '**' in text_b
has_list_a = '- ' in text_a or '1.' in text_a
has_list_b = '- ' in text_b or '1.' in text_b
print(f"Plain context:     headings={'YES' if has_heading_a else 'NO'}, lists={'YES' if has_list_a else 'NO'}, length={len(text_a)}")
print(f"Formatted context: headings={'YES' if has_heading_b else 'NO'}, lists={'YES' if has_list_b else 'NO'}, length={len(text_b)}")
