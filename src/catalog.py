# -*- coding: utf-8 -*-
"""จัดการแคตตาล็อกสินค้า: โหลด products.json -> index เข้า ChromaDB

หัวใจของ RAG ฝั่ง "R" (Retrieval):
แต่ละสินค้าถูกแปลงเป็น "เอกสาร" ข้อความเดียว (build_document)
แล้ว embed ด้วย bge-m3 เก็บใน ChromaDB — ตอนผู้ใช้ถาม เราค้นด้วยเวกเตอร์
(ข้อมูล: Amazon Reviews 2023, McAuley-Lab — ราคาเป็นดอลลาร์)
"""
import json

import chromadb

from . import config, llm


def load_products() -> list[dict]:
    """โหลดแคตตาล็อก — ใช้ products.json ถ้ามี ไม่งั้น fallback ไปไฟล์ sample"""
    path = (config.PRODUCTS_FILE if config.PRODUCTS_FILE.exists()
            else config.PRODUCTS_SAMPLE_FILE)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_document(p: dict) -> str:
    """รวม metadata เป็นข้อความเดียวสำหรับ embed

    ใส่ทั้งชื่อ หมวด แบรนด์ ราคา ฟีเจอร์ และรายละเอียด เพื่อให้เวกเตอร์จับความหมายครบ
    ผู้ใช้อาจถามด้วยหมวด ("มีหูฟังไหม"), ฟีเจอร์ ("แบบไร้สาย"), หรือ use-case ("ไว้เล่นเกม")
    """
    parts = [
        p["title"],
        f"หมวด: {p['category']}",
    ]
    if p.get("brand"):
        parts.append(f"แบรนด์: {p['brand']}")
    parts.append(f"ราคา: {p['price']} ดอลลาร์")
    if p.get("features"):
        parts.append(f"จุดเด่น: {', '.join(p['features'])}")
    if p.get("description"):
        parts.append(f"รายละเอียด: {p['description']}")
    return "\n".join(parts)


def get_collection() -> chromadb.Collection:
    """เปิด collection ที่ index แล้ว (ต้องรัน build_index ก่อน)"""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def build_index(force: bool = False) -> int:
    """สร้าง index ใน ChromaDB จากแคตตาล็อก คืนจำนวนสินค้าที่ index

    ทำครั้งเดียวพอ (persistent) — รันซ้ำจะข้ามถ้ามีครบแล้ว เว้นแต่ force=True
    """
    products = load_products()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    if force:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass

    col = client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if col.count() == len(products) and not force:
        return col.count()

    # embed ทีละ batch — ยัดทีเดียว 600 ชิ้นทำ bge-m3 runner ล่มได้
    print(f"กำลัง embed {len(products)} สินค้าด้วย {config.EMBED_MODEL} ...")
    batch = 64
    for i in range(0, len(products), batch):
        chunk = products[i:i + batch]
        docs = [build_document(p) for p in chunk]
        vectors = llm.embed(docs)
        col.add(
            ids=[p["id"] for p in chunk],
            embeddings=vectors,
            documents=docs,
            # เก็บ scalar ที่ใช้ filter (price/category) + record เต็มเป็น JSON string
            metadatas=[{
                "title": p["title"],
                "category": p["category"],
                "brand": p.get("brand", ""),
                "price": float(p["price"]),   # ใช้กรองงบด้วย $lte
                "rating": float(p.get("rating") or 0),
                "json": json.dumps(p, ensure_ascii=False),
            } for p in chunk],
        )
        print(f"  indexed {min(i + batch, len(products))}/{len(products)}", flush=True)
    return len(products)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # กัน print ไทยล่มบน console ที่ไม่ใช่ utf-8
    n = build_index(force="--force" in sys.argv)
    print(f"index พร้อมใช้: {n} สินค้า")
