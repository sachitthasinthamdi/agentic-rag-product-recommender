# -*- coding: utf-8 -*-
"""[2] Planner — dialog policy ของ agent (หัวใจของความเป็น "Agent")

ทุก turn LLM ตัดสินใจก่อนว่าจะทำ action ไหน (แบบ MACRS: ask / recommend / chat)
ไม่ใช่สักแต่ตอบ — นี่คือความต่างระหว่าง chatbot ธรรมดากับ agent ที่มี policy

  ask       = preference ยังไม่พอ ถามเจาะเพิ่มก่อน (ถามได้ไม่เกิน 1-2 รอบ)
  recommend = ข้อมูลพอแล้ว หรือผู้ใช้ขอตรงๆ -> ไปดึงสินค้ามาแนะนำ
  chat      = คุยทั่วไป / ถามข้อมูลสินค้าเฉยๆ ไม่ได้ขอคำแนะนำ
"""
import json

from . import llm

PLAN_SYSTEM = """คุณคือ Planner ของระบบแนะนำสินค้าแบบสนทนา
หน้าที่: เลือก action ของ turn นี้จากสามตัวเลือก

- "ask": ผู้ใช้อยากได้คำแนะนำแต่ข้อมูลยังน้อยเกิน (ยังไม่รู้หมวด/งบ/การใช้งาน)
  และเรายังถามไปไม่เกิน 2 ครั้ง — ถามเพิ่มเพื่อให้แนะนำได้แม่นขึ้น
- "recommend": ข้อมูลพอแนะนำได้ (รู้หมวดหรือการใช้งานอย่างน้อยหนึ่งอย่าง)
  หรือผู้ใช้ขอให้แนะนำเลย / ปฏิเสธของเดิมแล้วขอชิ้นใหม่
- "chat": ผู้ใช้ทักทาย คุยเล่น หรือถามข้อมูลสินค้า (เช่น "อันนี้กันน้ำไหม") ไม่ได้ขอคำแนะนำ

หลักคิด: อย่าถามพร่ำเพรื่อ ถ้าพอแนะนำได้ให้ recommend เลย ผู้ใช้รำคาญการถามซ้ำ
ตอบ JSON: {"action": "...", "reason": "เหตุผลสั้นๆ ภาษาไทย"}"""


def decide(profile: dict, recent_history: list[dict], user_message: str,
           ask_count: int) -> dict:
    """เลือก action ของ turn นี้ คืน {"action": ..., "reason": ...}"""
    prompt = (
        f"profile ที่รู้แล้ว:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"บทสนทนาล่าสุด:\n{_format_history(recent_history)}\n\n"
        f"ข้อความล่าสุดของผู้ใช้: \"{user_message}\"\n\n"
        f"จำนวนครั้งที่เราถาม preference ไปแล้ว: {ask_count}\n"
        "เลือก action"
    )
    result = llm.chat_json([{"role": "user", "content": prompt}], system=PLAN_SYSTEM)

    action = result.get("action", "")
    if action not in ("ask", "recommend", "chat"):
        # fallback แบบ heuristic ถ้า LLM ตอบเพี้ยน:
        # มี preference บ้างแล้ว -> แนะนำเลย, ไม่มีเลย -> ถาม
        has_pref = bool(profile["category"] or profile["use_case"]
                        or profile["features_wanted"] or profile["other_preferences"]
                        or profile["budget_max"])
        action = "recommend" if has_pref or ask_count >= 2 else "ask"
        result = {"action": action, "reason": "fallback: ตัดสินจากความครบของ profile"}

    # กันถามวนไม่จบ — ถามเกิน 2 ครั้งบังคับไปแนะนำ
    if action == "ask" and ask_count >= 2:
        result = {"action": "recommend", "reason": "ถามมาพอแล้ว ลุยแนะนำจากที่รู้"}
    return result


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(เพิ่งเริ่มคุย)"
    label = {"user": "ผู้ใช้", "assistant": "ผู้ช่วย"}
    return "\n".join(f"{label.get(h['role'], h['role'])}: {h['content']}" for h in history)
