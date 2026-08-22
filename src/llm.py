# -*- coding: utf-8 -*-
"""Wrapper บางๆ รอบ Ollama — จุดเดียวที่คุยกับ LLM

สามฟังก์ชันที่เหลือทั้งโปรเจกต์เรียกใช้:
  chat(messages)        -> ข้อความตอบกลับ (ภาษาไทย)
  chat_json(messages)   -> dict ที่ parse จาก JSON mode ของโมเดล
  embed(texts)          -> list ของเวกเตอร์ จาก bge-m3

เก็บสถิติการเรียกไว้ใน STATS ให้สคริปต์ eval อ่าน: จำนวนครั้ง, เวลารวม, JSON ที่พัง
(เป็น global แบบง่าย พอสำหรับงานสอน — ถ้าใช้ concurrent ต้องเปลี่ยนเป็น thread-safe)
"""
import json
import time

import ollama

from . import config

# ---------- สถิติสำหรับ eval (latency + JSON failure) ----------
STATS = {
    "chat":      {"calls": 0, "seconds": 0.0},
    "chat_json": {"calls": 0, "seconds": 0.0, "json_fail": 0},
    "embed":     {"calls": 0, "seconds": 0.0},
}


def reset_stats() -> None:
    """ล้างสถิติ — เรียกก่อนเริ่มวัดรอบใหม่"""
    STATS["chat"].update(calls=0, seconds=0.0)
    STATS["chat_json"].update(calls=0, seconds=0.0, json_fail=0)
    STATS["embed"].update(calls=0, seconds=0.0)


def chat(messages: list[dict], system: str | None = None) -> str:
    """คุยกับโมเดลแบบปกติ คืนข้อความล้วน"""
    if system:
        messages = [{"role": "system", "content": system}] + messages
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=config.CHAT_MODEL,
        messages=messages,
        options=config.CHAT_OPTIONS,
    )
    STATS["chat"]["calls"] += 1
    STATS["chat"]["seconds"] += time.perf_counter() - t0
    return resp["message"]["content"].strip()


def chat_json(messages: list[dict], system: str | None = None) -> dict:
    """คุยกับโมเดลใน JSON mode — ใช้กับงานสกัดข้อมูล/ตัดสินใจ

    Ollama จะบังคับให้โมเดลตอบเป็น JSON ที่ parse ได้ (format="json")
    ถ้า parse ไม่ได้จริงๆ คืน dict ว่าง (นับเป็น json_fail) ให้ caller มี fallback เสมอ
    """
    if system:
        messages = [{"role": "system", "content": system}] + messages
    t0 = time.perf_counter()
    resp = ollama.chat(
        model=config.CHAT_MODEL,
        messages=messages,
        format="json",
        options=config.JSON_OPTIONS,
    )
    STATS["chat_json"]["calls"] += 1
    STATS["chat_json"]["seconds"] += time.perf_counter() - t0

    text = resp["message"]["content"]
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    STATS["chat_json"]["json_fail"] += 1
    return {}


def embed(texts: list[str]) -> list[list[float]]:
    """แปลงข้อความเป็นเวกเตอร์ด้วย bge-m3 (รองรับภาษาไทยดี)"""
    t0 = time.perf_counter()
    resp = ollama.embed(model=config.EMBED_MODEL, input=texts)
    STATS["embed"]["calls"] += 1
    STATS["embed"]["seconds"] += time.perf_counter() - t0
    return resp["embeddings"]


def check_models() -> list[str]:
    """เช็คว่าโมเดลที่ต้องใช้ถูก pull มาแล้วหรือยัง คืนรายชื่อที่ยังขาด"""
    installed = {m.model for m in ollama.list().models}
    missing = []
    for name in (config.CHAT_MODEL, config.EMBED_MODEL):
        # ollama เก็บชื่อพร้อม tag เช่น "bge-m3:latest" — เทียบแบบขึ้นต้น
        if not any(inst.startswith(name) for inst in installed):
            missing.append(name)
    return missing
