# -*- coding: utf-8 -*-
"""[3] Retriever — ขั้น R ของ RAG

รับ query ภาษาไทย -> embed -> ค้น ChromaDB -> คืน candidate top-k
"""
import json

from . import config, llm, catalog


def retrieve(query: str, top_k: int = config.RETRIEVE_TOP_K,
             exclude_ids: set[str] | None = None,
             budget_max: float | None = None) -> list[dict]:
    """ค้นสินค้าที่ใกล้เคียง query ที่สุด

    exclude_ids: สินค้าที่ผู้ใช้ปฏิเสธ จะถูกกรองออก (ข้อมูลจาก memory)
    budget_max: งบสูงสุด -> กรองราคาแบบ hard filter ที่ vector DB เลย ($lte)
                (บทเรียนสำคัญ: เงื่อนไขตัวเลขชัดเจนควรกรองด้วย metadata
                 ไม่ใช่หวังให้ vector similarity เดางบเอาเอง)
    คืน list ของ product dict เต็ม พร้อม similarity score (0-1 ยิ่งสูงยิ่งตรง)
    """
    col = catalog.get_collection()
    vec = llm.embed([query])[0]

    where = {"price": {"$lte": float(budget_max)}} if budget_max else None
    # ขอเผื่อไว้ เพราะบางชิ้นอาจโดนกรองออกทีหลัง
    n = min(top_k + len(exclude_ids or ()), col.count())
    res = col.query(query_embeddings=[vec], n_results=n, where=where)

    out = []
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        product = json.loads(meta["json"])
        if exclude_ids and product["id"] in exclude_ids:
            continue
        product["score"] = round(1 - dist, 3)  # cosine distance -> similarity
        out.append(product)
    return out[:top_k]


def build_query(profile: dict, user_message: str) -> str:
    """ประกอบ query สำหรับ vector search จาก profile + ข้อความล่าสุด

    จุดสำคัญของ CRS: ไม่ได้ค้นจากข้อความล่าสุดอย่างเดียว
    แต่รวม preference ที่สะสมมาทั้งบทสนทนา (จาก memory) ด้วย
    (งบ budget_max ไม่ใส่ใน query แต่ไปกรองที่ metadata แทน — แม่นกว่า)
    """
    parts = []
    if profile.get("category"):
        parts.append(profile["category"])   # นำด้วยหมวด = anchor หลักของการค้น
    if profile.get("use_case"):
        parts.append("ใช้สำหรับ: " + profile["use_case"])
    if profile.get("features_wanted"):
        parts.append("ฟีเจอร์ที่อยากได้: " + ", ".join(profile["features_wanted"]))
    if profile.get("brands_liked"):
        parts.append("แบรนด์ที่ชอบ: " + ", ".join(profile["brands_liked"]))
    if profile.get("other_preferences"):
        parts.append(profile["other_preferences"])
    # ต่อท้ายด้วยข้อความล่าสุด เฉพาะเมื่อยังไม่ถูกครอบใน profile
    # (กันคำตอบสั้นๆ เช่น "มีสาย" ปรากฏซ้ำจนกลบ anchor หมวด)
    msg = user_message.strip()
    covered = (msg and (msg in (profile.get("category") or "")
                        or msg in (profile.get("use_case") or "")
                        or any(msg in f for f in profile.get("features_wanted", []))))
    if not covered:
        parts.append(msg)
    return " | ".join(parts) if parts else msg
