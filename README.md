# 🛒 น้องช้อป — Conversational Product Recommendation with Agentic RAG on LLMs

ระบบแนะนำแบบสนทนาด้วย Agentic RAG บน LLM (อีคอมเมิร์ซหลากหมวด) — โปรเจกต์สอน/เดโมที่รวม 3 แนวคิดใหญ่ **รัน local 100% ฟรีทั้งหมด**:

| แนวคิด | คืออะไรในโปรเจกต์นี้ | ไฟล์ |
|---|---|---|
| **AI Agent** | ทุก turn ระบบ "ตัดสินใจก่อนตอบ" ว่าจะถามเพิ่ม/แนะนำ/คุยเฉยๆ | `src/planner.py`, `src/agent.py` |
| **Conversational RecSys (CRS)** | สะสม preference ระหว่างคุย เป็น profile ที่อัปเดตได้ เปลี่ยนใจ/ปฏิเสธได้ (หมวด/งบ/แบรนด์/ฟีเจอร์/การใช้งาน) | `src/memory.py` |
| **RAG** | คำแนะนำ ground กับแคตตาล็อกสินค้าจริงใน vector DB — ไม่มโนสินค้าที่ไม่มีอยู่ | `src/catalog.py`, `src/retriever.py`, `src/reranker.py` |

ข้อมูลสินค้าจริงจาก [Amazon Reviews 2023 (McAuley-Lab)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) · อิงงานวิจัย [Survey of LLM-powered Agents for RecSys (arXiv:2502.10050)](https://arxiv.org/abs/2502.10050), [ARAG (arXiv:2506.21931)](https://arxiv.org/abs/2506.21931), MACRS (dialog policy)

> **มิติเชิงพาณิชย์ที่ระบบให้ความสำคัญ:** งบประมาณเป็น *hard filter*, การเปรียบเทียบสเปค/แบรนด์, use-case ("ไว้เล่นเกม/เป็นของขวัญ") — reranker ต้องคัดหลายเงื่อนไขพร้อมกัน

## สถาปัตยกรรม

```
ผู้ใช้พิมพ์ภาษาไทย
   │
   ▼
[1] Memory & Profile (memory.py) ─ LLM สกัด preference จากข้อความ อัปเดต profile JSON
   │                               (หมวด, งบ, แบรนด์, ฟีเจอร์, use-case, สินค้าที่ปฏิเสธ)
   ▼
[2] Planner (planner.py) ─ ตัดสินใจ action ของ turn นี้
   │        ask = ข้อมูลไม่พอ ถามเพิ่ม │ recommend = ไปดึงสินค้า │ chat = คุยทั่วไป
   ▼ (ถ้า recommend)
[3] Retriever (retriever.py) ─ ประกอบ query จาก profile + ข้อความ → embed (bge-m3)
   │                           → ค้น ChromaDB (กรองงบ price ≤ budget แบบ hard filter) → top-10
   ▼
[4] Reranker (reranker.py) ─ ARAG-lite: LLM ให้คะแนน candidate ทีละชิ้นเทียบ profile
   │                         คัดเหลือ top-3 (กรองชิ้นที่ "คล้าย query แต่เกินงบ/ผิดแบรนด์")
   ▼
[5] Generator (agent.py) ─ Typhoon2 เขียนคำแนะนำภาษาไทย ground กับ metadata จริง
```

ตัวโมเดล:
- **LLM**: [Typhoon2 8B](https://ollama.com/scb10x/llama3.1-typhoon2-8b-instruct) (SCB 10X) — LLM ไทย open-source ผ่าน Ollama
- **Embedding**: [bge-m3](https://ollama.com/library/bge-m3) — multilingual รองรับไทยดีมาก
- **Vector DB**: ChromaDB (persistent ในโฟลเดอร์ `.chroma/`)

## วิธีติดตั้ง

```powershell
# 1. ติดตั้ง Ollama (https://ollama.com) แล้วดึงโมเดล
ollama pull scb10x/llama3.1-typhoon2-8b-instruct   # ~4.9GB
ollama pull bge-m3                                  # ~1.2GB
# เครื่อง RAM < 16GB: ใช้ scb10x/llama3.2-typhoon2-3b-instruct แล้วแก้ CHAT_MODEL ใน src/config.py

# 2. ติดตั้ง Python packages
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## วิธีรัน

```powershell
python cli.py            # คุยผ่าน terminal
python cli.py --debug    # + โชว์ trace: action, profile, คะแนน rerank ทุก turn
streamlit run app.py     # เว็บ UI พร้อม side panel "ความคิดของ agent"
python -m eval.test_dialogs   # รันชุดทดสอบบทสนทนาอัตโนมัติ 5 สถานการณ์
```

## วัดผล (Evaluation)

ตัวชี้วัดครอบคลุมทั้ง 3 ระบบ (RAG / CRS / Agent) — ผลเซฟเป็นรายงานที่ `eval/reports/*.md` และ `.csv`

```powershell
python -m eval.eval_products     # RAG: category-retrieval (P@10, NDCG) + budget-filter correctness
python -m eval.eval_simulator    # Dialogue: Success@t ด้วย user simulator (LLM สวมบทลูกค้า)
python -m eval.eval_system       # System: latency ต่อ turn/ต่อโมเดล + JSON failure rate
python -m eval.test_dialogs      # Behavioral: 5 สถานการณ์ (recommend/ask/reject/budget/chat)
python -m eval.eval_amazon_loo   # Recall@k แบบ leave-one-out จากรีวิว Amazon จริง
```

| สคริปต์ | วัดอะไร | ข้อมูลที่ใช้ |
|---|---|---|
| `eval_products.py` | ค้นตรงหมวดไหม + budget filter ถูกไหม | สินค้า Amazon ~10,000 ชิ้น (label หมวดมาตรฐาน) |
| `eval_simulator.py` | แนะนำตรงเป้าหมายภายในกี่รอบ (Success@t) | user simulator (LLM สวมบทลูกค้า) |
| `eval_system.py` | เร็ว/เสถียรแค่ไหน | จับเวลา + อ่าน `llm.STATS` |
| `test_dialogs.py` | พฤติกรรมถูก spec ไหม (grounded/ปฏิเสธซ้ำ/งบ) | 5 สถานการณ์ที่นิยามเอง |
| `eval_amazon_loo.py` | ทำนายชิ้นถัดไป (Recall@k, leave-one-out) | รีวิว Amazon จริง (catalog รีวิวหนาแน่น) |

**ผลจริง (รัน local, คลัง ~10,000 ชิ้น):** Precision@10 = 0.867, NDCG@10 = 0.848 · **budget-filter อยู่ในงบ 100%** · Success@3 = 100% · JSON failure = 0% · behavioral 11/11 · Recall@10 (LOO) = 0.23

ครั้งแรกระบบจะ index สินค้าจาก `data/products.sample.json` (หรือ `products.json` ถ้ามี) เข้า ChromaDB อัตโนมัติ

ตัวอย่างประโยคลองเล่น:
- "หาเคสมือถือกันกระแทก งบไม่เกิน 15" (ดู budget filter)
- "อยากได้ของใช้ในบ้านสักอย่าง" (ดู agent ถามเจาะว่าหมวด/งบ)
- "ไม่เอาอันนี้ ขออันอื่น" (ดู memory ทำงาน — ไม่แนะนำซ้ำ)
- "เพิ่มงบเป็น 50 ได้" (ดู profile อัปเดตงบกลางคัน)

## ขยาย/สร้างแคตตาล็อกสินค้าใหม่ (ไม่บังคับ)

```powershell
python data\build_products.py       # stream สินค้าจาก Amazon Reviews 2023 -> data/products.json
python -m src.catalog --force       # rebuild index
```

แก้ `CATEGORIES` / `N_PER_CAT` ใน `data/build_products.py` เพื่อเปลี่ยนหมวด/จำนวน

## โครงสร้างโค้ด (เรียงตามลำดับที่ควรอ่านตอนสอน)

```
src/config.py     ค่ากลางทั้งหมด — เปลี่ยนโมเดล/ค่า k ที่นี่
src/llm.py        จุดเดียวที่คุยกับ Ollama: chat / chat_json / embed
src/catalog.py    โหลดสินค้า → สร้าง "เอกสาร" → embed → เก็บ ChromaDB   [บทเรียน RAG: Indexing]
src/retriever.py  ค้นเวกเตอร์ + ประกอบ query + กรองงบ (hard filter)     [บทเรียน RAG: Retrieval]
src/memory.py     ความจำ 2 ชั้น: history + structured profile           [บทเรียน CRS]
src/planner.py    dialog policy: ask / recommend / chat                 [บทเรียน Agent]
src/reranker.py   LLM ประเมิน candidates เทียบ หมวด/งบ/แบรนด์ ก่อนใช้จริง  [บทเรียน Agentic RAG]
src/agent.py      orchestrator ร้อยทุกโมดูล + system prompts
cli.py / app.py   หน้าบ้าน (terminal / Streamlit)
eval/test_dialogs.py  ชุดทดสอบพฤติกรรม 5 สถานการณ์
```

## แบบฝึกหัดต่อยอด (สำหรับผู้เรียน)

1. **ง่าย**: เพิ่มหมวดสินค้าใหม่ใน `data/build_products.py` แล้ว rebuild — ระบบแนะนำถูกไหม?
2. **กลาง**: เพิ่ม field `in_stock` แล้วให้ retriever กรองเฉพาะที่มีของ
3. **กลาง**: เพิ่ม action `compare` — ผู้ใช้ถาม "อันไหนดีกว่ากัน" แล้ว agent เปรียบเทียบสเปค/ราคา
4. **ยาก**: เปลี่ยน retriever เป็น hybrid search (vector + keyword BM25) เทียบผลกัน
5. **ยาก**: ทำ leave-one-out จาก Amazon reviews จริง (Recall@k) — สร้าง catalog จากชุดสินค้าที่มีรีวิวหนาแน่น

## ข้อจำกัดที่ควรบอกผู้เรียน

- โมเดล 8B ตอบไม่เสถียรเท่า cloud LLM — JSON เพี้ยนได้ ระบบเลยต้องมี fallback ทุกจุด (ดู `planner.decide`, `reranker.rerank`)
- แคตตาล็อกเป็นสินค้าสุ่มจาก Amazon (title/ราคาเป็นอังกฤษ/ดอลลาร์) บางคำขอเฉพาะทางอาจหาไม่เจอ
- ความเร็วขึ้นกับ GPU: RTX 4060 ตอบเฉลี่ย ~20 วินาที/turn (โหมด recommend ~30 วินาที มี LLM call หลายครั้ง: extract → plan → rerank → generate)
