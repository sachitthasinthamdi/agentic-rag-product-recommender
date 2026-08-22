# 📖 เดินโค้ดทีละไฟล์ — ตามลำดับการทำงาน 1 turn

เอกสารนี้อธิบายโค้ด **แต่ละไฟล์อย่างละเอียด** เรียงตาม **ลำดับที่ถูกเรียกจริง** ตอนผู้ใช้พิมพ์ 1 ข้อความ
เปิดอ่านคู่โค้ดใน IDE · ภาพรวมความเชื่อมโยงดูที่ [`README.md`](README.md)

> ตัวอย่างตลอดเอกสาร: ผู้ใช้พิมพ์ **"หาเคสมือถือกันกระแทก งบไม่เกิน 15"** → เส้นทาง `recommend`

**ลำดับที่จะเดิน:**
```
(ตอนเปิดแอป)  config.py → catalog.build_index
(ต่อ 1 turn)  app.py → agent.py → memory.py → llm.py → planner.py
              → retriever.py → catalog.py → reranker.py → (agent: generator) → return
```

---

# 0. ก่อนเริ่ม (รันครั้งเดียวตอนเปิดแอป)

## `config.py` — ค่ากลาง
ไม่ import ใครเลย เป็นแค่ค่าคงที่ที่ทุกไฟล์อ่านร่วมกัน
```python
CHAT_MODEL  = "scb10x/llama3.1-typhoon2-8b-instruct"   # โมเดลคิด/เขียน
EMBED_MODEL = "bge-m3"                                   # โมเดลค้นหา
COLLECTION_NAME = "products"        # ชื่อ collection ใน ChromaDB
RETRIEVE_TOP_K  = 10                # ดึง candidate กี่ชิ้น
RERANK_TOP_N    = 3                 # คัดเหลือกี่ชิ้น
CHAT_OPTIONS = {"temperature": 0.7, "num_ctx": 4096}   # งานเขียน — สร้างสรรค์
JSON_OPTIONS = {"temperature": 0.1, "num_ctx": 4096}   # งานสกัด — นิ่ง
```
👉 อยากเปลี่ยนโมเดล/จำนวนผลลัพธ์/อุณหภูมิ → แก้ที่นี่ที่เดียว

## `catalog.build_index()` — เตรียมคลัง (ทำครั้งเดียว)
```python
def build_index(force=False):
    products = load_products()                          # โหลด products.json (1,800 ชิ้น)
    col = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
    if col.count() == len(products) and not force: return col.count()   # มีครบแล้วข้าม
    batch = 64
    for i in range(0, len(products), batch):            # ทำทีละ 64 (ยัดทีเดียว bge-m3 ล่ม)
        chunk = products[i:i+batch]
        vectors = llm.embed([build_document(p) for p in chunk])   # แปลงเป็นเวกเตอร์
        col.add(ids=[p["id"] for p in chunk], embeddings=vectors, documents=docs,
                metadatas=[{"price":float(p["price"]), "category":p["category"],
                            "json":json.dumps(p, ensure_ascii=False)} for p in chunk])
```
- `build_document(p)` รวม ชื่อ+หมวด+แบรนด์+ราคา+จุดเด่น+รายละเอียด เป็นข้อความเดียวก่อน embed
- **เก็บ 2 อย่างใน metadata:** `price`/`category` (scalar ไว้กรอง) + `json` (record เต็มไว้ดึงกลับ)

> จบ startup — ตั้งแต่นี้คือ **ต่อ 1 turn**

---

# 1. `app.py` / `cli.py` — หน้าบ้าน (จุดเริ่ม turn)

```python
# app.py — เมื่อผู้ใช้พิมพ์
if user_input := st.chat_input(...):
    st.session_state.messages.append({"role":"user","text":user_input})
    with st.spinner("กำลังคิด..."):
        trace = st.session_state.agent.respond(user_input)   # ← เรียก orchestrator
    st.session_state.messages.append(
        {"role":"assistant","text":trace["reply"],"picked":trace.get("picked",[])})
    st.rerun()
```
- เก็บ `picked` (พร้อมข้อมูลรูป) ลง `messages` → render การ์ดจากประวัติ (จึง persist หลัง rerun)
- `agent.respond()` คืน `trace` = ทุกอย่างของ turn นั้น (reply, action, profile, picked, ...)

---

# 2. `agent.py` — Orchestrator (หัวใจ)

## โครง `respond()` — เรียกลูกทีมตามลำดับ
```python
def respond(self, user_message):
    trace = {}
    self.memory.update_profile(user_message, self.last_recommendations, self.memory.recent())  # ①
    trace["profile"] = json.loads(json.dumps(self.memory.profile))          # snapshot
    plan = planner.decide(self.memory.profile, self.memory.recent(),        # ②
                          user_message, self.ask_count)
    trace["action"], trace["plan_reason"] = plan["action"], plan["reason"]
    if plan["action"] == "recommend":                    # แตก 3 ทาง
        reply, extra = self._recommend(user_message); self.ask_count = 0
    elif plan["action"] == "ask":
        reply, extra = self._ask(user_message), {}; self.ask_count += 1
    else:
        reply, extra = self._chat(user_message), {}
    trace.update(extra)
    self.memory.add("user", user_message); self.memory.add("assistant", reply)
    trace["reply"] = reply
    return trace
```
**บทบาท:** ไม่ลงมือค้น/คิดเอง แต่สั่ง memory → planner → (recommend/ask/chat) → บันทึก → คืน trace
**2 ตัวจำข้ามรอบ:** `ask_count` (กันถามวน), `last_recommendations` (ให้ memory เข้าใจ "ไม่เอาอันนี้")

## `_recommend()` — ค้น → คัด → เขียน (⑤ Generator อยู่ในนี้)
```python
def _recommend(self, user_message):
    profile = self.memory.profile
    query = retriever.build_query(profile, user_message)              # ③a
    exclude = self.memory.rejected_ids(self.products)                 # ③b
    candidates = retriever.retrieve(query, exclude_ids=exclude,       # ③c
                                    budget_max=profile.get("budget_max"))
    picked = reranker.rerank([dict(c) for c in candidates], profile, user_message)  # ④
    if not picked:
        return ("ขอโทษ...หาไม่เจอ...", {"query":query,"candidates":candidates,"picked":[]})
    context = "\n\n".join(f"ชิ้นที่ {i+1}: {m['title']}\n...จุดเด่น...\nเหตุผลที่ระบบคัดมา: {m['fit_reason']}"
                          for i,m in enumerate(picked))               # ⑤ ประกอบ context
    reply = llm.chat(self.memory.recent() + [{"role":"user","content":prompt}],
                     system=RECOMMEND_SYSTEM)                         # ⑤ เขียนคำตอบ
    self.last_recommendations = [m["title"] for m in picked]
    return reply, {"query":query, "candidates":candidates, "picked":picked}
```
- `[dict(c) for c in candidates]` = **copy** ก่อนส่ง reranker (reranker เติม `fit_score` — กันไปแก้ต้นฉบับใน trace)
- Generator ส่ง **ประวัติแชท + context 3 ชิ้น** → `RECOMMEND_SYSTEM` บังคับ ground กับของจริง

## `_ask()` / `_chat()` — อีก 2 โหมด
```python
def _ask(self, msg):   return llm.chat(recent + [prompt], system=ASK_SYSTEM)   # ถาม 1 คำถาม
def _chat(self, msg):                                                          # คุย + ค้นเบา
    hits = retriever.retrieve(msg, top_k=3)      # ค้น 3 ชิ้นเผื่อถามข้อมูล
    return llm.chat(recent + [context+prompt], system=CHAT_SYSTEM)
```

---

# 3. `memory.py` — ความจำ (ขั้น ①)

## โครง profile — 8 ช่องที่ทั้งระบบใช้ร่วม
```python
EMPTY_PROFILE = {"category":"", "budget_max":None, "brands_liked":[], "brands_disliked":[],
                 "features_wanted":[], "use_case":"", "rejected_titles":[], "other_preferences":""}
class SessionMemory:
    def __init__(self):
        self.history = []                              # ประวัติแชท (RAM)
        self.profile = json.loads(json.dumps(EMPTY_PROFILE))   # deep copy (กัน session แชร์กัน)
```

## `update_profile()` — จดโน้ต (เรียก llm)
```python
def update_profile(self, user_message, last_recommendations, recent_history=None):
    convo = ""
    if recent_history:                                 # ยัดคำถามก่อนหน้า → ตีความคำตอบสั้น
        convo = "บทสนทนาก่อนหน้า...:\n" + "\n".join(...) + "\n\n"
    prompt = (f"profile ปัจจุบัน:\n{json.dumps(self.profile,...)}\n\n"
              f"สินค้าที่เพิ่งแนะนำล่าสุด: {...}\n\n{convo}"
              f"ข้อความล่าสุดของผู้ใช้: \"{user_message}\"\n\nอัปเดต profile แล้วตอบเป็น JSON")
    result = llm.chat_json([{"role":"user","content":prompt}], system=EXTRACT_SYSTEM)
    for key, default in EMPTY_PROFILE.items():          # ★ กรองก่อนจด (whitelist by schema)
        if key not in result: continue
        val = result[key]
        if key == "budget_max":                         # งบ: รับเฉพาะตัวเลข
            if isinstance(val,(int,float)): self.profile[key] = val
        elif isinstance(val, type(default)):            # อื่นๆ: ชนิดต้องตรง default
            self.profile[key] = val
```
- **วน `EMPTY_PROFILE` ไม่ใช่ `result`** → AI แถม field แปลก/ผิดชนิดหลุดเข้าไม่ได้
- `EXTRACT_SYSTEM` มีกฎกัน hallucinate (ห้ามแต่งฟีเจอร์/ห้ามดึงคำจากคำถาม/"ไม่มี"=ไม่เพิ่ม)

## `rejected_ids()` — แปลงชื่อที่ปฏิเสธ → id
```python
def rejected_ids(self, catalog_products):
    fragments = [t.lower() for t in self.profile["rejected_titles"] if len(t) > 3]
    return {p["id"] for p in catalog_products if any(f in p["title"].lower() for f in fragments)}
```
จับแบบ substring (ชื่อสินค้ายาว) → ส่งให้ retriever กรองออก

---

# 4. `llm.py` — ประตูเดียวสู่ AI (ทุกขั้นเรียก)

```python
def chat(messages, system=None):                        # โหมดข้อความ (Generator)
    if system: messages = [{"role":"system","content":system}] + messages
    resp = ollama.chat(model=CHAT_MODEL, messages=messages, options=CHAT_OPTIONS)  # temp 0.7
    STATS["chat"]["calls"] += 1; ...จับเวลา...
    return resp["message"]["content"].strip()

def chat_json(messages, system=None):                   # โหมด JSON (memory/planner/reranker)
    if system: messages = [{"role":"system",...}] + messages
    resp = ollama.chat(model=CHAT_MODEL, messages=messages, format="json", options=JSON_OPTIONS)  # temp 0.1
    text = resp["message"]["content"]
    try:
        data = json.loads(text)
        if isinstance(data, dict): return data          # คืน dict
    except json.JSONDecodeError: pass
    STATS["chat_json"]["json_fail"] += 1
    return {}                                           # ★ พังก็คืนว่าง ไม่ crash

def embed(texts):                                       # โหมดเวกเตอร์ (Retriever)
    resp = ollama.embed(model=EMBED_MODEL, input=texts)  # ส่ง list → ได้ list เวกเตอร์
    return resp["embeddings"]
```
- **`chat_json` คือหัวใจความเสถียร:** สัญญา = "คืน dict เสมอ อาจว่าง" → ผู้เรียกทุกจุดมี fallback
- โครง response Ollama = `resp["message"]["content"]` · `system` เป็น message role ปกติ (prepend)
- `STATS` เก็บ latency + json_fail ให้ eval อ่าน

---

# 5. `planner.py` — ตัดสินใจ (ขั้น ②)

```python
def decide(profile, recent_history, user_message, ask_count):
    prompt = (f"profile ที่รู้แล้ว:\n{json.dumps(profile,...)}\n\n"
              f"บทสนทนาล่าสุด:\n{_format_history(recent_history)}\n\n"
              f"ข้อความล่าสุด: \"{user_message}\"\n\n"
              f"จำนวนครั้งที่ถามไปแล้ว: {ask_count}\nเลือก action")
    result = llm.chat_json([{"role":"user","content":prompt}], system=PLAN_SYSTEM)
    action = result.get("action", "")
    if action not in ("ask","recommend","chat"):        # fallback ① (AI ตอบเพี้ยน)
        has_pref = bool(profile["category"] or profile["use_case"] or
                        profile["features_wanted"] or profile["other_preferences"] or profile["budget_max"])
        action = "recommend" if has_pref or ask_count>=2 else "ask"
    if action == "ask" and ask_count >= 2:              # fallback ② (กันถามวน)
        action = "recommend"
    return {"action": action, "reason": ...}
```
2 ชั้นกันพัง: action นอก enum → เดาจากความครบของ profile · ถามเกิน 2 ครั้ง → บังคับ recommend

---

# 6. `retriever.py` — ค้นหา (ขั้น ③)

## `build_query()` — ประกอบประโยคค้น
```python
def build_query(profile, user_message):
    parts = []
    if profile.get("category"):  parts.append(profile["category"])   # ★ นำด้วยหมวด = anchor
    if profile.get("use_case"):  parts.append("ใช้สำหรับ: " + ...)
    if profile.get("features_wanted"): parts.append("ฟีเจอร์ที่อยากได้: " + ...)
    msg = user_message.strip()
    covered = (msg in category or msg in use_case or any(msg in f for f in features_wanted))
    if not covered: parts.append(msg)                   # ไม่ใส่คำตอบสั้นซ้ำ (กันกลบ anchor)
    return " | ".join(parts)
```
รวม preference สะสมทั้งหมด (ไม่ค้นจากข้อความล่าสุดอย่างเดียว) — นี่คือหัวใจ CRS

## `retrieve()` — embed + ค้น + กรองงบ
```python
def retrieve(query, top_k=10, exclude_ids=None, budget_max=None):
    col = catalog.get_collection()                      # เปิด ChromaDB (เรียก catalog)
    vec = llm.embed([query])[0]                         # ← bge-m3 → เวกเตอร์
    where = {"price": {"$lte": float(budget_max)}} if budget_max else None   # ★ กรองงบเป๊ะ
    n = min(top_k + len(exclude_ids or ()), col.count())   # ขอเผื่อของที่จะโดนตัด
    res = col.query(query_embeddings=[vec], n_results=n, where=where)
    out = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        product = json.loads(meta["json"])              # ดึง record เต็มกลับจาก JSON string
        if exclude_ids and product["id"] in exclude_ids: continue
        product["score"] = round(1 - dist, 3)           # cosine distance → similarity
        out.append(product)
    return out[:top_k]
```
- **hard vs soft:** งบ (`$lte`) กรองเป๊ะที่ metadata · ความหมาย (หมวด/ฟีเจอร์) ใช้เวกเตอร์
- `json.loads(meta["json"])` = ทริค ดึงข้อมูลสินค้าเต็ม (รวม `image`) กลับมาโดยไม่ต้องมี DB ที่สอง

---

# 7. `catalog.py` — คลังสินค้า (ถูก retriever เรียก)

ตอน turn ใช้แค่ `get_collection()` (เปิด ChromaDB ที่ index ไว้แล้ว):
```python
def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)   # คืน collection "products"

def load_products():                                       # ใช้ตอน agent init + rejected_ids
    path = PRODUCTS_FILE if PRODUCTS_FILE.exists() else PRODUCTS_SAMPLE_FILE
    return json.load(open(path, encoding="utf-8"))
```
(`build_document` / `build_index` รันตอน startup — ดูส่วน 0)

---

# 8. `reranker.py` — คัดตัวจริง (ขั้น ④)

```python
def rerank(candidates, profile, user_message, top_n=3):
    if not candidates: return []
    lines = [f'- id:{m["id"]} | {m["title"][:60]} | หมวด:{m["category"]} | ${m["price"]} | '
             f'แบรนด์:{m.get("brand","-")} | จุดเด่น:{feats}' for m in candidates]   # เขียนรายการ
    prompt = (f"profile:\n{json.dumps(profile,...)}\n\nคำขอ: \"{user_message}\"\n\n"
              f"ผู้เข้าชิง {len(candidates)} ชิ้น:\n" + "\n".join(lines) + "\n\nให้คะแนนทุกชิ้น")
    result = llm.chat_json([{"role":"user","content":prompt}], system=RERANK_SYSTEM)  # AI ให้คะแนน

    score_map = {}
    for row in result.get("scores", []):                # parse คะแนนเข้า map
        if isinstance(row, dict) and "id" in row:
            try: score_map[row["id"]] = (float(row["score"]), str(row["reason"]))
            except (TypeError, ValueError): continue    # score ไม่ใช่ตัวเลข → ข้าม
    if not score_map:                                   # AI ใช้ไม่ได้เลย → fallback vector
        for m in candidates: m["fit_score"], m["fit_reason"] = m["score"]*10, "เรียงตามความคล้าย"
        return candidates[:top_n]
    for m in candidates:                                # merge คะแนนกลับเข้าแต่ละชิ้น
        m["fit_score"], m["fit_reason"] = score_map.get(m["id"], (0.0, "ไม่ได้ให้คะแนน"))
    ranked = sorted(candidates, key=lambda m: m["fit_score"], reverse=True)
    ranked = [m for m in ranked if m["fit_score"] >= 3] or ranked[:1]   # ตัด<3 · never-empty
    return ranked[:top_n]
```
- แยก **parse** (สร้าง score_map) ออกจาก **merge** (attach กลับ) → id ที่ AI ลืมก็มี default
- `[... if >=3] or ranked[:1]` → กรองแล้วว่างก็ยังคืน 1 ชิ้น (กันหน้าจอว่าง)

---

# 9. กลับมาที่ `agent.respond()` — ปิด turn

```python
self.memory.add("user", user_message)      # เก็บลง history (RAM)
self.memory.add("assistant", reply)
trace["reply"] = reply
return trace                               # → app.py วาดคำตอบ + การ์ด (รูป/ราคา)
```

---

# 🔁 สรุปการเรียก AI ทั้ง turn (recommend)

| ลำดับ | ไฟล์ | ฟังก์ชัน llm | โมเดล | ผลลัพธ์ |
|---|---|---|---|---|
| ① | memory.py | `chat_json` | Typhoon2 | profile JSON |
| ② | planner.py | `chat_json` | Typhoon2 | action |
| ③ | retriever.py | `embed` | bge-m3 | เวกเตอร์ → candidates ×10 |
| ④ | reranker.py | `chat_json` | Typhoon2 | picked ×3 |
| ⑤ | agent.py | `chat` | Typhoon2 | reply |

**AI 5 ครั้ง/turn** (Typhoon2 ×4 + bge-m3 ×1) · ทุกครั้งผ่าน `llm.py` · ทุก dict ผ่าน type-guard + fallback

> ดูภาพประกอบ: [`file-map-turn.svg`](file-map-turn.svg) (ลำดับบนแผนผังไฟล์) · [`turn-flow.svg`](turn-flow.svg) (ข้อมูลเปลี่ยนรูป)
