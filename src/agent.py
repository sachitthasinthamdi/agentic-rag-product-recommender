# -*- coding: utf-8 -*-
"""[5] Agent orchestrator — ประกอบทุกโมดูลเป็น loop เดียว

ลำดับต่อ 1 turn:
  user msg -> [1] memory.update_profile -> [2] planner.decide
           -> (ถ้า recommend) [3] retriever -> [4] reranker
           -> [5] generate คำตอบภาษาไทยแบบ ground กับข้อมูลสินค้าจริง

ทุกขั้นเก็บ trace ไว้ให้ UI โชว์ "ความคิด" ของ agent — มีค่ามากตอนสอน
"""
import json

from . import llm, catalog, memory, planner, retriever, reranker

CHAT_SYSTEM = """คุณคือ "น้องช้อป" ผู้ช่วยแนะนำสินค้า พูดไทยเป็นกันเอง สุภาพ กระชับ
คุยสนุกแต่ไม่ยืดยาว ถ้าผู้ใช้ถามข้อมูลสินค้าที่อยู่ในบริบทที่ให้ ให้ตอบจากบริบทเท่านั้น
ถ้าไม่รู้ให้บอกตรงๆ ห้ามแต่งข้อมูลเอง"""

ASK_SYSTEM = """คุณคือ "น้องช้อป" ผู้ช่วยแนะนำสินค้า ภาษาไทยเป็นกันเอง
ตอนนี้ข้อมูลยังไม่พอจะแนะนำ จงถามคำถามสั้นๆ 1 คำถามเพื่อเจาะ preference ที่ยังขาด
เช่น อยากได้สินค้าหมวดไหน งบประมาณเท่าไหร่ เอาไว้ใช้ทำอะไร มีแบรนด์หรือฟีเจอร์ที่ต้องการไหม
ถามอย่างเดียว อย่าเพิ่งแนะนำสินค้า และอย่าถามหลายเรื่องพร้อมกัน"""

RECOMMEND_SYSTEM = """คุณคือ "น้องช้อป" ผู้ช่วยแนะนำสินค้า พูดไทยเป็นกันเอง จริงใจ เหมือนเพื่อนที่รู้ใจ
แนะนำจาก "ตัวเลือกที่ระบบคัดมาแล้ว" เท่านั้น ห้ามแต่งสินค้า/ชื่อ/ราคา/สเปคที่ไม่ได้ให้มา
แนะนำครบทุกชิ้น เป็นลิสต์หมายเลข 1. 2. 3.

แต่ละชิ้นเขียน 2 บรรทัด:
- บรรทัดแรก: เลข. ชื่อสินค้า — $ราคา · แบรนด์  (ใช้ชื่อ/ราคา/แบรนด์ตามที่ให้มาเป๊ะ ห้ามใส่วงเล็บมุม < >)
- บรรทัดสอง: เหตุผล 2-3 ประโยคที่ต้องมีครบทั้ง 3 อย่าง:
   1) โยงกับสิ่งที่ "ผู้ใช้คนนี้" พิมพ์มาโดยตรง (การใช้งาน/ฟีเจอร์/งบ ที่เขาบอก)
   2) ยกจุดเด่นจริงของสินค้าชิ้นนั้น (จากข้อมูล "จุดเด่น" ที่ให้) มาอธิบายว่าตอบโจทย์ตรงไหน
   3) บอกว่าราคาอยู่ในงบ (ทุกชิ้นอยู่ในงบแล้ว — ห้ามบอกว่าเกินงบเด็ดขาด)
   ถ้าชิ้นไหนไม่ตรงเป๊ะ ให้บอกตามตรงว่าต่างตรงไหน แต่ทำไมยังน่าลอง

ห้าม: ประโยคกว้างๆ ลอยๆ เช่น "ตรงหมวด ตรงการใช้งาน" · พิมพ์ชื่อ/ราคาผิด · ใส่วงเล็บมุม < >
· บอกว่า "เป็นแบรนด์ที่คุณชอบ" ทั้งที่ผู้ใช้ไม่ได้บอกว่าชอบแบรนด์นั้น
ปิดท้ายด้วยถามสั้นๆ ว่าถูกใจชิ้นไหน หรืออยากปรับงบ/ฟีเจอร์อะไรไหม

ตัวอย่างที่ดี (ทำตามสไตล์นี้):
1. Adesso Xtream H4 Stereo Headset — $11.97 · Adesso
   คุณอยากได้หูฟังมีสายไว้เล่นเกม ตัวนี้เป็นเฮดเซ็ตมีสายพร้อมไมค์ในตัว คุยกับเพื่อนในเกมได้เลย เสียงสเตอริโอช่วยฟังทิศทางในเกม ราคา $11.97 อยู่ในงบ $40 สบายๆ"""


class ProductRecAgent:
    def __init__(self):
        self.memory = memory.SessionMemory()
        self.products = catalog.load_products()   # ไว้ map ชื่อสินค้า -> id
        self.ask_count = 0                        # กัน planner ถามวนไม่จบ
        self.last_recommendations: list[str] = []

    def respond(self, user_message: str) -> dict:
        """ประมวลผล 1 turn คืน {"reply", "action", "profile", "candidates", "picked", ...}"""
        trace: dict = {}

        # [1] อัปเดตความจำ + โปรไฟล์ (ส่งบทสนทนาก่อนหน้าไปช่วยตีความคำตอบสั้นๆ เช่น "มีสาย")
        self.memory.update_profile(user_message, self.last_recommendations,
                                   self.memory.recent())
        trace["profile"] = json.loads(json.dumps(self.memory.profile))

        # [2] วางแผน: turn นี้ควรทำอะไร
        plan = planner.decide(self.memory.profile, self.memory.recent(),
                              user_message, self.ask_count)
        trace["action"], trace["plan_reason"] = plan["action"], plan["reason"]

        # [3]-[5] ทำตามแผน
        if plan["action"] == "recommend":
            reply, extra = self._recommend(user_message)
            self.ask_count = 0
        elif plan["action"] == "ask":
            reply, extra = self._ask(user_message), {}
            self.ask_count += 1
        else:
            reply, extra = self._chat(user_message), {}
        trace.update(extra)

        # บันทึกบทสนทนา
        self.memory.add("user", user_message)
        self.memory.add("assistant", reply)
        trace["reply"] = reply
        return trace

    # ---------- actions ----------

    def _recommend(self, user_message: str) -> tuple[str, dict]:
        profile = self.memory.profile

        # [3] Retrieve: ค้นด้วย query ที่รวม preference สะสม
        # budget_max เป็น hard filter — เกินงบต้องไม่หลุดมา
        query = retriever.build_query(profile, user_message)
        exclude = self.memory.rejected_ids(self.products)
        candidates = retriever.retrieve(query, exclude_ids=exclude,
                                        budget_max=profile.get("budget_max"))

        # [4] Rerank: LLM คัดตัวจริงเทียบกับ profile (หัวใจ ARAG)
        picked = reranker.rerank([dict(c) for c in candidates], profile, user_message)

        if not picked:
            return ("ขอโทษด้วยนะคะ ในคลังตอนนี้ยังหาสินค้าที่ตรงใจไม่เจอเลย "
                    "ลองปรับงบ หรือบอกหมวด/การใช้งานเพิ่มได้ไหมคะ"), {
                        "query": query, "candidates": candidates, "picked": []}

        # [5] Generate: ตอบโดย ground กับ metadata จริงของสินค้าที่คัดมา
        context = "\n\n".join(
            f"ชิ้นที่ {i+1}: {m['title']}\n"
            f"หมวด: {m['category']} | ราคา ${m['price']} | แบรนด์: {m.get('brand', '-')} | "
            f"rating {m.get('rating', '-')} ({m.get('rating_count', 0)} รีวิว)\n"
            f"จุดเด่น: {', '.join(m.get('features', [])[:3])}\n"
            f"เหตุผลที่ระบบคัดมา: {m['fit_reason']}"
            for i, m in enumerate(picked)
        )
        prompt = (
            f"profile ผู้ใช้: {json.dumps(profile, ensure_ascii=False)}\n"
            f"คำขอล่าสุด: \"{user_message}\"\n\n"
            f"ตัวเลือกที่ระบบคัดมาแล้ว:\n{context}\n\n"
            "เขียนคำแนะนำ"
        )
        reply = llm.chat(self.memory.recent() + [{"role": "user", "content": prompt}],
                         system=RECOMMEND_SYSTEM)

        self.last_recommendations = [m["title"] for m in picked]
        return reply, {"query": query, "candidates": candidates, "picked": picked}

    def _ask(self, user_message: str) -> str:
        prompt = (
            f"profile ที่รู้แล้ว: {json.dumps(self.memory.profile, ensure_ascii=False)}\n"
            f"ผู้ใช้เพิ่งพิมพ์: \"{user_message}\"\n"
            "ถามคำถามเดียวเพื่อเติม preference ที่ยังขาด"
        )
        return llm.chat(self.memory.recent() + [{"role": "user", "content": prompt}],
                        system=ASK_SYSTEM)

    def _chat(self, user_message: str) -> str:
        # แถม retrieval เบาๆ ให้โหมดคุย — เผื่อผู้ใช้ถามข้อมูลสินค้าในคลัง
        hits = retriever.retrieve(user_message, top_k=3)
        context = "\n".join(
            f"- {m['title']} (${m['price']}, {m.get('brand', '')}): {m.get('description', '')[:110]}"
            for m in hits
        )
        prompt = (f"บริบทจากคลังสินค้า (ใช้เฉพาะที่เกี่ยวข้อง):\n{context}\n\n"
                  f"ข้อความผู้ใช้: \"{user_message}\"")
        return llm.chat(self.memory.recent() + [{"role": "user", "content": prompt}],
                        system=CHAT_SYSTEM)
