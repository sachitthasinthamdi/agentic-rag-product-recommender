# -*- coding: utf-8 -*-
"""สคริปต์เปิดดูข้อมูลใน ChromaDB — รัน: python peek_chroma.py"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from src import config

client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

print("=== ทุก collection ในฐานข้อมูล ===")
for c in client.list_collections():
    print(f"  - {c.name}  (จำนวน record: {c.count()})")

print("\n=== เปิด collection 'products' ===")
col = client.get_collection(config.COLLECTION_NAME)
print("จำนวนสินค้าทั้งหมด:", col.count())

print("\n=== ดึงมา 2 record แรก ===")
res = col.get(limit=2, include=["documents", "metadatas"])
for i in range(len(res["ids"])):
    print(f"\n--- record {i+1} ---")
    print("id       :", res["ids"][i])
    print("document :", res["documents"][i][:200])
    print("metadata :", res["metadatas"][i])

print("\n=== ทดลองค้นหาจริง (ต้องเปิด Ollama ก่อน): 'หูฟังไร้สายกันน้ำ' ===")
# หมายเหตุ: ต้อง embed ด้วย bge-m3 เอง ห้ามใช้ query_texts (Chroma จะใช้ตัว embed ผิดตัว)
try:
    from src import retriever
    for i, p in enumerate(retriever.retrieve("หูฟังไร้สายกันน้ำ", top_k=3), 1):
        print(f"  {i}. {p['id']}  |  {p['title'][:55]}  |  ${p['price']}  (score {p['score']})")
except Exception as e:
    print("  ข้ามการค้นหา (Ollama ยังไม่เปิด?):", e)
