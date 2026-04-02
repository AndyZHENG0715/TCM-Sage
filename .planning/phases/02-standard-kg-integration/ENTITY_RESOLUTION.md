# Entity resolution: RAG terms ↔ SymMap bridge

**Status:** Updated for SymMap-only KG architecture
**Date:** 2026-04-02
**Purpose:** 定義 Bridge 規則，把 RAG 檢索中的古籍術語映射到 SymMap authoritative 節點；不再維護「Neijing 抽取 KG × SymMap 合併圖」路線。

> Architecture decision: KG layer is SymMap-only. `data/graph/entities_partial.json` is legacy and no longer a runtime dependency.

---

## 1. 同類實體（例如都是 Symptom 的「頭痛」）用什麼 key 判定是同一個節點？

採用 **分層判定**，由強到弱；**不**把「裸字串 name」當成全域唯一主鍵。

| 層級 | 條件 | 說明 |
|------|------|------|
| **L0（最高）** | 明確 **SymMap ID** 對齊 | Neijing 節點若帶 `symmap_id` / 或外部 **crosswalk 表**（`neijing_entity_id` → `SMTS…`），則視為同一實體。 |
| **L1** | **正規化後中文名 exact match**，且 **type 相同** | 僅在 **Symptom↔Symptom、Herb↔Herb** 等同類型內比對；name 需同一套正規化（全形/半形、空白、常見異體字規則可配置）。產出節點帶 `merge_confidence: exact_name`。 |
| **L2（預設關閉或僅建議）** | **模糊 / 向量相似** | 僅作 **候選列表** 供人工或離線審核；預設 **不自動合併**，避免 TCM 同名異義、簡繁/別名誤併。 |

**Canonical node id（合併後圖內主鍵）：**

- 若與 SymMap 對齊成功：**以 SymMap 實體 ID 為 canonical**（學術資料集穩定、可審計）。
- 若僅 Neijing 有：**保留 Neijing 抽取圖既有 id**（或專名前綴如 `nj_…`，實作時再定一條規則即可）。

---

## 2. 若判定為同一實體：合併策略

### 2.1 屬性（attributes）

- **原則：合併為聯集，禁止靜默覆蓋。**
- **結構化臨床/現代層**（SymMap：拼音、英文名、現代病證欄位等）優先寫入對應欄位或 `symmap_*` 命名空間。
- **古典出處**（Neijing：書名、篇、章、原文片段、`source_ref`）寫入 `neijing_*` 或 `provenance[]` 列表。
- 建議在節點上保留 **`sources: ["symmap", "neijing"]`**（或等價結構），便於除錯與 UI 標示。

### 2.2 關係（relations）

- **聯集合併**：兩邊的邊都保留。
- **去重**：以 `(source_id, target_id, relationship_type)` 為鍵；若描述不同，可合併為單邊並附 **`source_ref` 列表** 或 `evidence[]`，而非丟棄其中一條。
- **型別衝突**（極少）：若同一對節點出現互斥語義邊，**不要自動解**；記 log / 標 `conflict: true` 供審核。

---

## 3. 若無法對齊（SymMap 有而 Neijing 無，或反之）

| 情況 | 策略 |
|------|------|
| **僅一側存在** | **維持獨立節點**（各自子圖仍可做 traversal）。 |
| **兩側都有節點但無法安全合併** | **仍為兩個節點**；**預設不自動加 `RELATED_TO`**（避免圖爆炸與偽連邊）。 |
| **需表達「可能相關」** | 僅在 **L0 有軟證據**（例如 crosswalk 標為 uncertain）或 **人工批准** 時，加 **低置信 `MAPS_TO` / `RELATED_TO`**，邊上必須帶 `confidence` / `source_ref`。 |

---

## 4. 與實作的銜接（摘要）

1. **先做 crosswalk（L0）**：小表或 JSON 映射優於全靠 name。
2. **自動合併預設只開 L0 + L1（exact name + 同 type）**；L2 只做候選不做 commit。
3. **合併後圖** 對外 API（`load_from_json` / merge 輸出）應能區分 **canonical id**、**別名**、**多來源 provenance**。

---

*本文件供審閱；定案後再改 `graph_builder.py` 或新增 `scripts/merge_kg.py` 實作。*

---

## 5. 現況確認（2026-03-28）

### L0 crosswalk 是否已存在？

**結論：目前不存在可直接使用的 crosswalk table。**

本次在 repo 的檢查結果：
- 未發現 `neijing_entity_id -> symmap_id` 類型映射檔（json/csv/tsv）。
- 未發現獨立「crosswalk 生成腳本」或既有管線輸出該映射。
- `data/graph/symmap/symmap_entities.json` 有 SymMap component 資訊；
  `data/graph/entities_partial.json` 為 Neijing 抽取節點（mention/source_ref），兩者尚未有正式橋接鍵。

### 這代表什麼？

在未建立 L0 之前，實際自動對齊會主要依賴 L1（同 type + normalized exact name），
覆蓋率與精度都會受限於術語差異（古典語彙 vs 現代資料集命名）。

另外，目前 `data/graph/symmap/symmap_entities.json` 仍包含大量示例命名（如「示例症狀*」「示例藥材*」），
即便走 L1 也幾乎無法對齊到 Neijing 實體。
因此 seed crosswalk 的人工審核需要在「真實 SymMap 匯出資料」上進行才有意義。

### 實作前決策（必選一）

1. **建 L0（推薦）**：先做 seed crosswalk（人工 + 規則輔助），再進 merge。
2. **暫不做 L0**：先上 L1-only merge，接受低覆蓋率，並把「未對齊率」列為驗收指標。
3. **混合方案**：先做小規模 L0（高頻症狀/藥物），其餘走 L1 並輸出候選審核清單。

> 建議採 **3（混合）**：最快落地且風險可控；等 seed crosswalk 穩定再擴成完整 L0。
