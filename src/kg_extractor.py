"""
KG Extractor - 3-Pass LLM-based Knowledge Graph extraction for TCM texts.

Architecture:
    Pass 1: Entity Extraction → entities[] with evidence spans
    Pass 2: Relation Extraction → relations[] using entity IDs only
    Pass 3: Self-Critique → verify & add confidence scores

Uses Ollama with Qwen3 model (8b primary, 4b fallback).
Entity schema aligned with WHO TCM Terminology (2022).
"""

import json
import pathlib
import re
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm

# Debug flag
DEBUG = False

try:
    import ollama
except ImportError:
    ollama = None
    print("⚠️ ollama package not installed. Run: pip install ollama")


# WHO-aligned entity types (Expanded for Neijing physiological content)
ENTITY_TYPES = {
    "Symptom": "症状",
    "Pattern": "证候/证型", 
    "Herb": "中药",
    "Formula": "方剂",
    "TreatmentMethod": "治法",
    "Meridian": "经络",
    "Acupoint": "腧穴",
    "BodyPart": "人体部位/脏腑",
    "Substance": "物质 (精/气/血/津液)"
}

RELATION_TYPES = {
    "TREATS": "治疗",
    "CONTAINS": "含有",
    "INDICATES": "提示",
    "APPLIES_TO": "适用于",
    "LOCATED_ON": "位于",
    "ORIGINATES_FROM": "起始于",
    "FLOWS_THROUGH": "循行于",
    "DERIVED_FROM": "来源于",
    "ENTERS": "入于/传于",
    "GOVERNS": "主持/主管"
}


def normalize_entity_type(raw_type: str) -> str:
    """
    Normalize entity types to handle LLM output corruption.
    Fixes issues like 'Sym,ptom', 'symptom', 'Mer,idian' etc.
    """
    if not raw_type:
        return "Unknown"
    
    # Remove any commas or special chars
    clean = raw_type.replace(',', '').replace('.', '').strip()
    
    # Lowercase for matching
    lower = clean.lower()
    
    # Map to canonical types
    type_map = {
        'symptom': 'Symptom',
        'pattern': 'Pattern',
        'herb': 'Herb',
        'formula': 'Formula',
        'treatmentmethod': 'TreatmentMethod',
        'meridian': 'Meridian',
        'acupoint': 'Acupoint',
        'bodypart': 'BodyPart',
        'substance': 'Substance',
    }
    
    return type_map.get(lower, clean.title())

PASS1_ENTITY_PROMPT = """/nothink
你是一个严格的中医信息抽取引擎。只输出JSON，不要输出任何解释。

任务：从给定《黄帝内经》文本中抽取实体。

⚠️ 严格规则（禁止违反）：
1) 只抽取原文中**字面出现**的实体，禁止推理或补充
2) evidence必须是原文**连续子串**，可直接找到
3) 如果原文说"五脏六腑"，只抽取"五脏六腑"整体，不要拆分成心肝脾肺肾
4) 如果原文只说"太阴"而没有"手太阴"，则保留"太阴"，不要补全为"手太阴肺经"
5) 禁止使用你的中医知识补充原文没有的信息

实体类型（仅限以下9种）：
- Symptom (症状): 疾病症状，包括舌象、脉象
- Pattern (证候): 中医证型
- Herb (中药): 中药材
- Formula (方剂): 方剂复方
- TreatmentMethod (治法): 治疗方法
- Meridian (经络): 经络名称（如原文有完整名称如"手太阴"则用完整名，如原文只有"太阴"则用"太阴"）
- Acupoint (腧穴): 穴位
- BodyPart (人体部位): 脏腑、器官、解剖部位（仅限原文字面出现的）
- Substance (物质): 营气、卫气、宗气、精、血、津液等

输出JSON格式：
{{
  "entities": [
    {{
      "id": "substance_营",
      "type": "Substance",
      "mention": "营",
      "evidence": "其清者为营"
    }}
  ]
}}

文本：
<<<
{chunk}
>>>

来源：{book} - {chapter}

/no_think
"""


PASS2_RELATION_PROMPT = """/nothink
你是一个中医关系抽取引擎。只输出JSON，不要输出任何解释。

任务：基于"实体列表"和"原文"，抽取实体间的关系。

关系类型（共10种）：

【治疗相关】
- TREATS: 药材/方剂/治法 治疗 症状/证候
- CONTAINS: 方剂 含有 药材
- INDICATES: 症状 提示 证候
- APPLIES_TO: 治法 适用于 证候

【经络腧穴相关】
- LOCATED_ON: 穴位 位于 经络
- ORIGINATES_FROM: 气/血 起始于 经络/脏腑 （如：营气起于手太阴肺经）
- FLOWS_THROUGH: 气/血 循行于 经络/脉内/脉外 （如：营在脉中、卫在脉外）

【生理关系】
- DERIVED_FROM: 物质 来源于 物质/脏腑 （如：营气来源于水谷精微）
- ENTERS: 物质 入于/传于 脏腑 （如：谷入于胃、以传于肺）
- GOVERNS: 经络/脏腑 主持/主管 功能/区域 （如：太阴主内、太阳主外）

关系方向提示：
- 动词"入""传""注""归"→ ENTERS
- 动词"起""始""生"→ ORIGINATES_FROM  
- 动词"行""循""在"→ FLOWS_THROUGH
- 动词"主""司"→ GOVERNS

要求：
1) head/tail 必须来自实体列表的id（不能新造）
2) 必须给出evidence（原文片段）
3) 仔细分析原文动词，选择最准确的关系类型

输出JSON格式：
{{
  "relations": [
    {{
      "head": "substance_营气",
      "relation": "ORIGINATES_FROM",
      "tail": "meridian_手太阴肺经",
      "evidence": "营气...起于手太阴"
    }}
  ]
}}

实体列表：
{entities_json}

原文：
<<<
{chunk}
>>>

/no_think
"""


PASS3_CRITIQUE_PROMPT = """/nothink
你是一个知识验证引擎。只输出JSON，不要输出任何解释。

任务：验证已抽取的实体和关系是否被原文明确支持。

要求：
1) 检查每个实体的evidence是否在原文中存在
2) 检查每个关系的evidence是否在原文中存在
3) 给出置信度分数 (0.0-1.0)：
   - 1.0: 原文明确支持
   - 0.7-0.9: 原文暗示但不明确
   - 0.3-0.6: 可能正确但证据不足
   - 0.0-0.2: 原文不支持或幻觉
4) 标记不支持的事实

输出JSON格式：
{{
  "verified_entities": [
    {{
      "id": "...",
      "confidence": 1.0,
      "supported": true
    }}
  ],
  "verified_relations": [
    {{
      "head": "...",
      "tail": "...",
      "confidence": 0.85,
      "supported": true
    }}
  ]
}}

原文：
<<<
{chunk}
>>>

已抽取的实体：
{entities_json}

已抽取的关系：
{relations_json}

/no_think
"""


def extract_entities_pass1(
    chunk: str,
    metadata: Dict,
    model: str = "qwen3:8b",
    num_ctx: int = 4096
) -> List[Dict]:
    """
    Pass 1: Extract entities from text chunk.
    """
    if ollama is None:
        return []
    prompt = PASS1_ENTITY_PROMPT.format(
        chunk=chunk,
        book=metadata.get('book', 'Unknown'),
        chapter=metadata.get('source', 'Unknown')
    )
    
    try:
        if DEBUG:
            print(f"\n[DEBUG] Pass 1 Input: {metadata.get('book')} - {metadata.get('source')}")
            
        response = ollama.generate(
            model=model, 
            prompt=prompt,
            options={"num_ctx": num_ctx, "temperature": 0.0}
        )
        response_text = response.get('response', '').strip()
        
        if DEBUG:
            print(f"[DEBUG] Pass 1 Raw Output: {response_text[:200]}...")
        
        # Extract JSON from response
        # Find the first { and last } to isolate the potential JSON object
        # Robust JSON extraction
        match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
                entities = data if isinstance(data, list) else data.get('entities', [])
                # Normalize entity types
                for e in entities:
                    if 'type' in e:
                        e['type'] = normalize_entity_type(e['type'])
                return entities
            except json.JSONDecodeError:
                # Try simple fix for trailing commas
                try:
                    json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
                    data = json.loads(json_str)
                    entities = data if isinstance(data, list) else data.get('entities', [])
                    for e in entities:
                        if 'type' in e:
                            e['type'] = normalize_entity_type(e['type'])
                    return entities
                except:
                    return []
        return []
        
    except Exception as e:
        print(f"⚠️ Pass 1 error: {e}")
        return []



def extract_relations_pass2(
    chunk: str,
    entities: List[Dict],
    model: str = "qwen3:8b",
    num_ctx: int = 4096
) -> List[Dict]:
    """
    Pass 2: Extract relations using entity IDs only.
    """
    if ollama is None or not entities:
        return []
    
    entities_json = json.dumps(entities, ensure_ascii=False, indent=2)
    
    if DEBUG:
        print(f"\n[DEBUG] Pass 2 Input (Entities): {[e.get('id') for e in entities]}")
        
    prompt = PASS2_RELATION_PROMPT.format(
        chunk=chunk,
        entities_json=entities_json
    )
    
    try:
        response = ollama.generate(
            model=model, 
            prompt=prompt,
            options={"num_ctx": num_ctx, "temperature": 0.0}
        )
        response_text = response.get('response', '').strip()
        
        if DEBUG:
            print(f"[DEBUG] Pass 2 Raw Output: {response_text[:200]}...")
        
        # Robust JSON extraction
        match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
                if isinstance(data, list):
                    return data
                return data.get('relations', [])
            except json.JSONDecodeError:
                try:
                    json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
                    data = json.loads(json_str)
                    if isinstance(data, list):
                        return data
                    return data.get('relations', [])
                except:
                    return []
        return []
        
    except Exception as e:
        print(f"⚠️ Pass 2 error: {e}")
        return []


def critique_kg_pass3(
    chunk: str,
    entities: List[Dict],
    relations: List[Dict],
    model: str = "qwen3:8b",
    num_ctx: int = 4096
) -> Tuple[List[Dict], List[Dict]]:
    """
    Pass 3: Self-critique to verify and add confidence scores.
    """
    if ollama is None:
        return entities, relations
    
    entities_json = json.dumps(entities, ensure_ascii=False, indent=2)
    relations_json = json.dumps(relations, ensure_ascii=False, indent=2)
    
    if DEBUG:
        print(f"\n[DEBUG] Pass 3 Input: {len(entities)} entities, {len(relations)} relations")
        
    prompt = PASS3_CRITIQUE_PROMPT.format(
        chunk=chunk,
        entities_json=entities_json,
        relations_json=relations_json
    )
    
    try:
        response = ollama.generate(
            model=model, 
            prompt=prompt,
            options={"num_ctx": num_ctx, "temperature": 0.0}
        )
        response_text = response.get('response', '').strip()
        
        if DEBUG:
            print(f"[DEBUG] Pass 3 Raw Output: {response_text[:200]}...")
        
        # Robust JSON extraction
        match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
                    data = json.loads(json_str)
                except:
                    return entities, relations

            if isinstance(data, dict):
                verified_entities = data.get('verified_entities', [])
                verified_relations = data.get('verified_relations', [])
                
                # Merge confidence scores back
                entity_map = {e['id']: e for e in entities}
                relation_map = {(r['head'], r['tail']): r for r in relations}
                
                for ve in verified_entities:
                    if ve.get('id') in entity_map:
                        entity_map[ve['id']]['confidence'] = ve.get('confidence', 0.5)
                        entity_map[ve['id']]['supported'] = ve.get('supported', True)
                
                for vr in verified_relations:
                    key = (vr.get('head'), vr.get('tail'))
                    if key in relation_map:
                        relation_map[key]['confidence'] = vr.get('confidence', 0.5)
                        relation_map[key]['supported'] = vr.get('supported', True)
                
                return list(entity_map.values()), list(relation_map.values())
            
    except Exception as e:
        print(f"⚠️ Pass 3 error: {e}")
    
    # Fallback: return original with default confidence
    for e in entities:
        e.setdefault('confidence', 0.5)
        e.setdefault('supported', True)
    for r in relations:
        r.setdefault('confidence', 0.5)
        r.setdefault('supported', True)
    
    return entities, relations


def clean_text_chunk(text: str) -> str:
    """
    Clean text by removing editorial notes/annotations in parentheses.
    This helps improve extraction recall for heavily annotated texts like '太素'.
    """
    if not text:
        return ""
    # Remove full-width parenthetical notes （...）
    text = re.sub(r'（[^）]*）', '', text)
    # Remove half-width parenthetical notes (...)
    text = re.sub(r'\([^)]*\)', '', text)
    return text


def extract_kg_from_chunk(
    chunk: str,
    metadata: Dict,
    model: str = "qwen3:8b",
    num_ctx: int = 4096
) -> Dict:
    """
    3-pass extraction: Entity → Relation → Self-Critique.
    
    NOTE: Cleans the chunk text (removes annotations) before processing to improve result quality.
    """
    # Clean text to remove interruptions
    chunk = clean_text_chunk(chunk)
    
    # Pass 1: Extract entities
    entities = extract_entities_pass1(chunk, metadata, model, num_ctx)
    
    # Pass 2: Extract relations
    relations = extract_relations_pass2(chunk, entities, model, num_ctx)
    
    # Pass 3: Self-critique
    entities, relations = critique_kg_pass3(chunk, entities, relations, model, num_ctx)
    
    # Add provenance
    source_ref = {
        "book": metadata.get('book', 'Unknown'),
        "chapter": metadata.get('source', 'Unknown'),
        "char_start": metadata.get('char_start', 0),
        "char_end": metadata.get('char_end', 0)
    }
    
    for e in entities:
        e['source_ref'] = source_ref
    for r in relations:
        r['source_ref'] = source_ref
    
    return {
        "entities": entities,
        "relationships": relations,
        "source_ref": source_ref
    }


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Deduplicate entities by ID."""
    seen = {}
    for entity in entities:
        entity_id = entity.get('id')
        if not entity_id:
            continue
            
        if entity_id not in seen:
            seen[entity_id] = entity.copy()
        else:
            # Keep higher confidence
            if entity.get('confidence', 0) > seen[entity_id].get('confidence', 0):
                seen[entity_id] = entity.copy()
    
    return list(seen.values())


def deduplicate_relationships(relationships: List[Dict]) -> List[Dict]:
    """Deduplicate relationships by (head, relation, tail) tuple."""
    seen = {}
    for rel in relationships:
        key = (rel.get('head'), rel.get('relation'), rel.get('tail'))
        if key not in seen:
            seen[key] = rel.copy()
        else:
            # Keep higher confidence
            if rel.get('confidence', 0) > seen[key].get('confidence', 0):
                seen[key] = rel.copy()
    
    return list(seen.values())


def extract_kg_batch(
    chunks: List[Dict],
    model: str = "qwen3:8b",
    num_ctx: int = 4096,
    limit: Optional[int] = None
) -> Dict:
    """
    Extract KG from all chunks with 3-pass approach and merge results.
    """
    if limit:
        chunks = chunks[:limit]
    
    all_entities = []
    all_relationships = []
    errors = 0
    
    print(f"🔍 3-Pass KG Extraction from {len(chunks)} chunks using {model}...")
    print(f"   Pass 1: Entity Extraction")
    print(f"   Pass 2: Relation Extraction")
    print(f"   Pass 3: Self-Critique\n")
    
    for chunk in tqdm(chunks, desc="Extracting KG"):
        result = extract_kg_from_chunk(
            chunk.get('content', ''),
            chunk.get('metadata', {}),
            model,
            num_ctx
        )
        
        entities = result.get('entities', [])
        relations = result.get('relationships', [])
        
        # Determine if extraction failed completely
        if not entities and not relations:
            errors += 1
        
        all_entities.extend(entities)
        all_relationships.extend(relations)
    
    # Deduplicate
    print("\n🔄 Deduplicating entities and relationships...")
    unique_entities = deduplicate_entities(all_entities)
    unique_relationships = deduplicate_relationships(all_relationships)
    
    # Filter by confidence and support
    high_conf_entities = [e for e in unique_entities if e.get('confidence', 0) >= 0.7 and e.get('supported', True)]
    high_conf_relations = [r for r in unique_relationships if r.get('confidence', 0) >= 0.7 and r.get('supported', True)]
    
    print(f"✅ Extracted:")
    print(f"   Total entities: {len(unique_entities)} (high conf: {len(high_conf_entities)})")
    print(f"   Total relations: {len(unique_relationships)} (high conf: {len(high_conf_relations)})")
    if errors:
        print(f"⚠️ {errors} chunks had extraction errors")
    
    return {
        "entities": unique_entities,
        "relationships": unique_relationships,
        "extraction_stats": {
            "total_chunks": len(chunks),
            "unique_entities": len(unique_entities),
            "unique_relationships": len(unique_relationships),
            "high_confidence_entities": len(high_conf_entities),
            "high_confidence_relationships": len(high_conf_relations),
            "errors": errors
        }
    }


def save_kg(kg_data: Dict, output_path: pathlib.Path) -> None:
    """Save extracted KG to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved KG to {output_path}")


def main():
    """Main function to run 3-pass KG extraction on processed chunks."""
    script_dir = pathlib.Path(__file__).parent
    chunks_path = script_dir.parent / "data" / "processed" / "chunks.json"
    output_path = script_dir.parent / "data" / "graph" / "entities.json"
    
    print("=" * 60)
    print("🧠 TCM-Sage 3-Pass KG Extraction Pipeline")
    print("=" * 60)
    
    # Check Ollama
    if ollama is None:
        print("❌ ollama package not installed. Run: pip install ollama")
        return
    
    # Load chunks
    if not chunks_path.exists():
        print(f"❌ Chunks file not found: {chunks_path}")
        print("   Run ingest.py first to generate chunks.")
        return
    
    print(f"📖 Loading chunks from {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"   Found {len(chunks)} chunks\n")
    
    # Decide model
    model = "qwen3:8b"
    print(f"🤖 Using model: {model}")
    print("   Optimized settings: num_ctx=4096, temperature=0.0\n")
    
    # Run extraction
    kg_data = extract_kg_batch(chunks, model=model)
    
    # Save results
    save_kg(kg_data, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Extraction Summary")
    print("=" * 60)
    stats = kg_data.get('extraction_stats', {})
    print(f"   Chunks processed: {stats.get('total_chunks', 0)}")
    print(f"   Unique entities: {stats.get('unique_entities', 0)}")
    print(f"   High-conf entities: {stats.get('high_confidence_entities', 0)} (≥0.7)")
    print(f"   Unique relationships: {stats.get('unique_relationships', 0)}")
    print(f"   High-conf relationships: {stats.get('high_confidence_relationships', 0)} (≥0.7)")
    
    # Show entity type distribution
    print("\n📋 Entity distribution by type:")
    entities = kg_data.get('entities', [])
    by_type = {}
    for e in entities:
        t = e.get('type', 'Unknown')
        if t not in by_type:
            by_type[t] = []
        mention = e.get('mention', e.get('name', 'Unknown'))
        by_type[t].append(mention)
    
    for entity_type, mentions in sorted(by_type.items()):
        sample = ', '.join(mentions[:5])
        print(f"   {entity_type}: {sample}{'...' if len(mentions) > 5 else ''} ({len(mentions)} total)")


if __name__ == "__main__":
    main()
