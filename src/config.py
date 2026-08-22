# -*- coding: utf-8 -*-
"""ค่าคอนฟิกกลางของโปรเจกต์ — แก้ที่นี่ที่เดียว"""
from pathlib import Path

# ---------- โมเดล (รันผ่าน Ollama ทั้งหมด) ----------
# ถ้าเครื่อง RAM/VRAM ไม่พอ ให้เปลี่ยนเป็น "scb10x/llama3.2-typhoon2-3b-instruct"
CHAT_MODEL = "scb10x/llama3.1-typhoon2-8b-instruct"
EMBED_MODEL = "bge-m3"

# ---------- Path ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRODUCTS_FILE = DATA_DIR / "products.json"          # ไฟล์เต็ม (สร้างจาก build_products.py)
PRODUCTS_SAMPLE_FILE = DATA_DIR / "products.sample.json"  # ตัวอย่างติดโปรเจกต์
CHROMA_DIR = PROJECT_ROOT / ".chroma"           # ที่เก็บ vector DB

# ---------- Retrieval ----------
COLLECTION_NAME = "products"
RETRIEVE_TOP_K = 10   # ดึง candidate จาก vector search กี่ชิ้น (คลัง ~10k; top-10 แม่นพอ P@10=0.87 + เดโมเร็ว)
RERANK_TOP_N = 3      # คัดเหลือแนะนำจริงกี่ชิ้น

# ---------- LLM options ----------
CHAT_OPTIONS = {"temperature": 0.7, "num_ctx": 4096}
JSON_OPTIONS = {"temperature": 0.1, "num_ctx": 4096}  # งานสกัดข้อมูล ใช้ temperature ต่ำ
