# -*- coding: utf-8 -*-
"""[4] Reranker — แนวคิด ARAG แบบย่อ (arXiv:2506.21931)

ปัญหาของ RAG ธรรมดา: vector search วัดแค่ "ความคล้ายข้อความ"
สินค้าที่คล้าย query ที่สุดอาจขัด preference อื่น เช่น เกินงบ หรือเป็นแบรนด์ที่ไม่ชอบ

ทางแก้แบบ ARAG: ให้ LLM ทำหน้าที่ NLI — อ่าน candidate ทีละชิ้น
เทียบกับ profile ทั้งชุด (หมวด/งบ/แบรนด์/ฟีเจอร์) แล้วให้คะแนน "ความเข้ากัน" ใหม่
โดเมนสินค้ามีเงื่อนไขหลายอย่างพร้อมกัน -> reranker ยิ่งจำเป็น
"""
import json

from . import config, llm

RERANK_SYSTEM = """คุณคือผู้เชี่ยวชาญประเมินว่าสินค้าแต่ละชิ้น "ตรงใจ" ผู้ใช้แค่ไหน
ให้คะแนน 0-10 ทีละชิ้น โดยเทียบกับ profile และคำขอล่าสุด:
- ตรงหมวด / การใช้งาน / ฟีเจอร์ที่ต้องการ = คะแนนสูง
- ผิดหมวด หรือเป็นของคนละประเภท (เช่น อะไหล่/สายไฟ แทนที่จะเป็นตัวสินค้า) = คะแนนต่ำมาก
- เป็นแบรนด์ใน brands_disliked = คะแนนต่ำ · อยู่ใน rejected_titles = ให้ 0
- rating สูงหรือรีวิวเยอะ = บวกเล็กน้อย
(หมายเหตุ: ระบบกรองราคา <= งบ ให้ก่อนแล้ว ทุกชิ้นอยู่ในงบ — ห้ามบอกว่า "เกินงบ")

reason: ประโยคเต็ม 1-2 ประโยค พูดถึง "สินค้าชิ้นนั้นโดยเฉพาะ" (ประเภท/จุดเด่นของมัน)
ว่าตรงหรือขัด preference ข้อไหน — ห้ามตอบกว้างๆ ลอยๆ ว่า "ตรงหมวด ตรงการใช้งาน"
เช่น "เฮดเซ็ตมีสายมีไมค์ เหมาะเล่นเกมตามที่ขอ ราคาในงบ" ไม่ใช่แค่ "ตรงใจ/ตรงหมวด"
ตอบ JSON: {"scores": [{"id": "...", "score": 0-10, "reason": "..."}]}"""


def rerank(candidates: list[dict], profile: dict, user_message: str,
           top_n: int = config.RERANK_TOP_N) -> list[dict]:
    """คัด candidate จาก vector search เหลือตัวจริง top_n เรียงตามความตรงใจ

    คืน candidate เดิมที่เพิ่ม field: fit_score, fit_reason
    ถ้า LLM ตอบเพี้ยน -> fallback เรียงตาม vector score เดิม (ระบบไม่ล่มกลางบทสนทนา)
    """
    if not candidates:
        return []

    lines = []
    for m in candidates:
        feats = ", ".join(m.get("features", [])[:3])[:90]
        lines.append(
            f'- id: {m["id"]} | {m["title"][:60]} | หมวด: {m["category"]} | '
            f'${m["price"]} | แบรนด์: {m.get("brand", "-")} | '
            f'rating {m.get("rating", "-")} ({m.get("rating_count", 0)} รีวิว) | จุดเด่น: {feats}'
        )
    prompt = (
        f"profile ผู้ใช้:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"คำขอล่าสุด: \"{user_message}\"\n\n"
        f"ผู้เข้าชิง {len(candidates)} ชิ้น:\n" + "\n".join(lines) + "\n\nให้คะแนนทุกชิ้น"
    )
    result = llm.chat_json([{"role": "user", "content": prompt}], system=RERANK_SYSTEM)

    score_map = {}
    for row in result.get("scores", []):
        if isinstance(row, dict) and "id" in row:
            try:
                score_map[row["id"]] = (float(row.get("score", 0)), str(row.get("reason", "")))
            except (TypeError, ValueError):
                continue

    if not score_map:  # LLM ตอบใช้ไม่ได้ -> ใช้ลำดับ vector search เดิม
        for m in candidates:
            m["fit_score"], m["fit_reason"] = m["score"] * 10, "เรียงตามความคล้ายข้อความ"
        return candidates[:top_n]

    for m in candidates:
        score, reason = score_map.get(m["id"], (0.0, "โมเดลไม่ได้ให้คะแนน"))
        m["fit_score"], m["fit_reason"] = score, reason

    ranked = sorted(candidates, key=lambda m: m["fit_score"], reverse=True)
    # ตัดชิ้นที่คะแนนต่ำมาก (ขัด preference ชัดเจน) ออกไปเลย
    ranked = [m for m in ranked if m["fit_score"] >= 3] or ranked[:1]
    return ranked[:top_n]
