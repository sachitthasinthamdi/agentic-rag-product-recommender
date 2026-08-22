# -*- coding: utf-8 -*-
"""[1] Memory & Profile — ความจำของ agent

สองส่วนตามงานวิจัย CRS:
  - history  : ความจำระยะสั้น (บทสนทนาใน session นี้)
  - profile  : ความจำเชิงโครงสร้าง (preference สะสม) — LLM สกัดจากแต่ละ turn

profile คือสิ่งที่ทำให้ระบบ "จำได้" ว่าคุยอะไรกันมา ไม่ใช่ตอบทีละประโยคแบบไร้บริบท
(โดเมนสินค้า: มีมิติ งบประมาณ/แบรนด์/ฟีเจอร์/use-case เพิ่มเติม)
"""
import json

from . import llm

# โครง profile เริ่มต้น — field ตรงกับที่ planner/retriever/reranker ใช้
EMPTY_PROFILE = {
    "category": "",           # หมวด/ประเภทสินค้าที่สนใจ เช่น "หูฟัง", "โน้ตบุ๊ก"
    "budget_max": None,       # งบสูงสุด (ตัวเลข ดอลลาร์) — ใช้เป็น hard filter
    "brands_liked": [],       # แบรนด์ที่ชอบ
    "brands_disliked": [],    # แบรนด์ที่ไม่เอา
    "features_wanted": [],    # ฟีเจอร์ที่อยากได้ เช่น ["ไร้สาย", "กันน้ำ"]
    "use_case": "",           # ใช้ทำอะไร เช่น "เล่นเกม", "ทำงาน", "ของขวัญ"
    "rejected_titles": [],    # สินค้าที่ปฏิเสธ — ห้ามแนะนำซ้ำ
    "other_preferences": "",  # อื่นๆ เช่น "ขอรีวิวเยอะๆ", "สีดำ"
}

EXTRACT_SYSTEM = """คุณคือระบบสกัดข้อมูล preference จากบทสนทนาแนะนำสินค้า
หน้าที่: อ่านข้อความล่าสุดของผู้ใช้ แล้วอัปเดต profile เดิมให้เป็นปัจจุบัน

กติกา:
- คงข้อมูลเดิมไว้ เพิ่ม/แก้เฉพาะสิ่งที่ข้อความล่าสุดบอก
- ใส่เฉพาะสิ่งที่ "ผู้ใช้" พูดจริงเท่านั้น ห้ามเดาหรือแต่งฟีเจอร์เพิ่มเอง
  และห้ามดึงคำจาก "คำถามของผู้ช่วย" มาใส่ (คำถามเป็นแค่บริบทช่วยตีความคำตอบ ไม่ใช่ความต้องการของผู้ใช้)
- ถ้าผู้ใช้ตอบ "ไม่มี"/"ไม่รู้"/"อะไรก็ได้" = ไม่มีข้อมูลใหม่ อย่าเพิ่มฟีเจอร์ใดๆ
- budget_max = ตัวเลขล้วน (ดอลลาร์) เช่น "งบไม่เกิน 50" -> 50 ; ถ้าไม่ระบุคงเป็น null
- ถ้าผู้ใช้ปฏิเสธสินค้าที่เพิ่งแนะนำ (เช่น "ไม่เอาอันนี้") ให้เพิ่มชื่อสินค้านั้นใน rejected_titles
- ถ้าผู้ใช้บอกแบรนด์ที่ไม่ชอบ ให้ใส่ brands_disliked
- category/use_case เขียนทับได้ถ้าผู้ใช้บอกใหม่
- ถ้าข้อความล่าสุดเป็นคำตอบสั้นๆ ให้ดู "บทสนทนาก่อนหน้า" เพื่อตีความ
  เช่น ผู้ช่วยถาม "มีสายหรือไร้สาย" แล้วผู้ใช้ตอบ "มีสาย" -> เพิ่ม "มีสาย" ใน features_wanted
  (ถ้าตอบ "ไร้สาย" ให้เอา "มีสาย" ออกแล้วใส่ "ไร้สาย" แทน — อย่าเก็บทั้งคู่ที่ขัดกัน)
- เขียนค่าข้อความเป็น "ภาษาไทย" (ยกเว้นชื่อแบรนด์/รุ่นที่เป็นอังกฤษได้)
- ตอบเป็น JSON โครงเดียวกับ profile เดิมเท่านั้น ห้ามเพิ่ม field ใหม่

ตัวอย่าง: "หาหูฟังไร้สายไว้ออกกำลังกาย งบไม่เกิน 40" ต้องได้
{"category": "หูฟังไร้สาย", "budget_max": 40, "brands_liked": [], "brands_disliked": [],
 "features_wanted": ["ไร้สาย", "กันเหงื่อ"], "use_case": "ออกกำลังกาย",
 "rejected_titles": [], "other_preferences": ""}"""


class SessionMemory:
    def __init__(self):
        self.history: list[dict] = []            # [{"role","content"}, ...]
        self.profile: dict = json.loads(json.dumps(EMPTY_PROFILE))

    # ---------- history (ความจำระยะสั้น) ----------

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def recent(self, n: int = 8) -> list[dict]:
        """คืน n ข้อความล่าสุด — จำกัดไว้กัน context ยาวเกิน"""
        return self.history[-n:]

    # ---------- profile (ความจำเชิงโครงสร้าง) ----------

    def update_profile(self, user_message: str, last_recommendations: list[str],
                       recent_history: list[dict] | None = None):
        """ให้ LLM อ่านข้อความล่าสุด แล้วอัปเดต profile (JSON in -> JSON out)

        last_recommendations: ชื่อสินค้าที่เพิ่งแนะนำไป turn ก่อน
        ช่วยให้ LLM รู้ว่า "ไม่เอาอันนี้" หมายถึงสินค้าไหน
        recent_history: บทสนทนาก่อนหน้า — ช่วยตีความคำตอบสั้นๆ เช่น "มีสาย" (ตอบคำถามของ agent)
        """
        convo = ""
        if recent_history:
            label = {"user": "ผู้ใช้", "assistant": "ผู้ช่วย"}
            lines = "\n".join(f"{label.get(h['role'], h['role'])}: {h['content']}"
                              for h in recent_history[-4:])
            convo = f"บทสนทนาก่อนหน้า (ใช้ตีความคำตอบสั้นๆ):\n{lines}\n\n"
        prompt = (
            f"profile ปัจจุบัน:\n{json.dumps(self.profile, ensure_ascii=False)}\n\n"
            f"สินค้าที่เพิ่งแนะนำล่าสุด: {json.dumps(last_recommendations, ensure_ascii=False)}\n\n"
            f"{convo}"
            f"ข้อความล่าสุดของผู้ใช้: \"{user_message}\"\n\n"
            "อัปเดต profile แล้วตอบเป็น JSON"
        )
        result = llm.chat_json(
            [{"role": "user", "content": prompt}], system=EXTRACT_SYSTEM
        )
        # รับเฉพาะ field ที่รู้จัก + ชนิดข้อมูลถูกต้อง — LLM อาจตอบเพี้ยนได้เสมอ
        for key, default in EMPTY_PROFILE.items():
            if key not in result:
                continue
            val = result[key]
            if key == "budget_max":               # อัปเดตเฉพาะเมื่อเป็นตัวเลข
                if isinstance(val, (int, float)):  # กัน LLM ลบงบเดิมด้วย null ตอน turn ต่อไป
                    self.profile[key] = val
            elif isinstance(val, type(default)):
                self.profile[key] = val

    def rejected_ids(self, catalog_products: list[dict]) -> set[str]:
        """แปลง rejected_titles เป็น id ในแคตตาล็อก เพื่อกรองตอน retrieve

        ชื่อสินค้ายาว — จับแบบ substring (ชิ้นส่วนที่ปฏิเสธอยู่ในชื่อสินค้าไหม)
        """
        rejected = set()
        fragments = [t.lower() for t in self.profile["rejected_titles"] if len(t) > 3]
        for p in catalog_products:
            name = p["title"].lower()
            if any(frag in name for frag in fragments):
                rejected.add(p["id"])
        return rejected
