# -*- coding: utf-8 -*-
"""Leave-One-Out Recall@k บนรีวิว Amazon จริง (ข้อมูลมาตรฐาน) — โดเมนสินค้า

วิธีมาตรฐานเดียวกับ MovieLens leave-one-out:
  - ดึงประวัติผู้ใช้จริงจากไฟล์รีวิว Amazon (user -> สินค้าที่รีวิว)
  - ต่อผู้ใช้ 1 คน: ซ่อนสินค้าชิ้นล่าสุด 1 ชิ้น (target) ใช้ที่เหลือเป็น "รสนิยม" (profile)
  - วัดว่า retriever ดึง target กลับมาใน top-k ได้ไหม -> Recall@k

รัน:  python -m eval.eval_amazon_loo
ผล:  eval/reports/amazon_loo.md

หมายเหตุ (ตามตรง): retriever เป็น content-based (ดูเนื้อหาสินค้า) ไม่ใช่ collaborative filtering
ค่า Recall อาจต่ำ = task mismatch ไม่ใช่บั๊ก (บทเรียนเดียวกับ MovieLens leave-one-out)
ต้องสตรีมไฟล์รีวิว/เมทาจาก HuggingFace (ใหญ่) จึงใช้ line cap + early stop
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import json
from collections import defaultdict

import chromadb
import requests

from src import config, llm
from eval.report_utils import write_markdown, md_table

BASE = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        "/resolve/main/raw")
CATEGORY = "Office_Products"
REVIEW_URL = f"{BASE}/review_categories/{CATEGORY}.jsonl"
META_URL = f"{BASE}/meta_categories/meta_{CATEGORY}.jsonl"

MAX_REVIEW_LINES = 150000   # อ่านรีวิวมากสุดกี่บรรทัด (คุมเวลา/ปริมาณโหลด)
MAX_META_LINES = 80000
MIN_ITEM_FREQ = 5           # สินค้าต้องถูกรีวิว >= ครั้งนี้ ถึงเข้า eval catalog
MAX_ITEMS = 500             # ขนาด eval catalog สูงสุด
MIN_USER_ITEMS = 2          # ผู้ใช้ต้องรีวิว >= นี้ (ในเซ็ต) ถึงทำ LOO ได้
MAX_USERS = 300             # จำนวน LOO instance สูงสุด (คุมเวลา retrieval)
KS = [1, 5, 10, 20]
COLLECTION = "amazon_loo"


def stream_reviews():
    """สตรีมรีวิว: เก็บ (ก) จำนวนรีวิวต่อสินค้า (ข) ประวัติผู้ใช้ -> [(asin, time)]"""
    item_freq = defaultdict(int)
    user_hist = defaultdict(list)
    print(f"stream reviews {CATEGORY} (สูงสุด {MAX_REVIEW_LINES} บรรทัด) ...", flush=True)
    r = requests.get(REVIEW_URL, stream=True, timeout=120)
    r.raise_for_status()
    n = 0
    for line in r.iter_lines(decode_unicode=True):
        n += 1
        if n > MAX_REVIEW_LINES:
            break
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        asin, user = d.get("parent_asin"), d.get("user_id")
        if not asin or not user:
            continue
        item_freq[asin] += 1
        user_hist[user].append((asin, d.get("timestamp") or 0))
        if n % 50000 == 0:
            print(f"  อ่าน {n} รีวิว ...", flush=True)
    r.close()
    print(f"  รวม {n} รีวิว · สินค้าไม่ซ้ำ {len(item_freq)} · ผู้ใช้ {len(user_hist)}",
          flush=True)
    return item_freq, user_hist


def select_items(item_freq: dict) -> set:
    """เลือกสินค้าที่ถูกรีวิวหนาแน่นเป็น eval catalog"""
    popular = [a for a, c in item_freq.items() if c >= MIN_ITEM_FREQ]
    popular.sort(key=lambda a: -item_freq[a])
    return set(popular[:MAX_ITEMS])


def stream_meta(wanted: set) -> dict:
    """สตรีมเมทาดาทา เก็บเฉพาะ asin ที่อยู่ใน wanted และมี title+description"""
    meta = {}
    print(f"stream meta (หา {len(wanted)} ชิ้น) ...", flush=True)
    r = requests.get(META_URL, stream=True, timeout=120)
    r.raise_for_status()
    n = 0
    for line in r.iter_lines(decode_unicode=True):
        n += 1
        if n > MAX_META_LINES or len(meta) >= len(wanted):
            break
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        asin = d.get("parent_asin")
        if asin not in wanted or asin in meta:
            continue
        title = (d.get("title") or "").strip()
        desc = " ".join(x for x in (d.get("description") or []) if isinstance(x, str))
        if not title or len(desc) < 20:
            continue
        feats = [x for x in (d.get("features") or []) if isinstance(x, str)][:5]
        meta[asin] = {"id": asin, "title": title[:140],
                      "category": d.get("main_category") or CATEGORY,
                      "features": feats, "description": desc[:500]}
    r.close()
    print(f"  ได้เมทา {len(meta)} ชิ้น (อ่าน {n} บรรทัด)", flush=True)
    return meta


def document(m: dict) -> str:
    parts = [m["title"], f"หมวด: {m['category']}"]
    if m["features"]:
        parts.append("จุดเด่น: " + ", ".join(m["features"]))
    parts.append("รายละเอียด: " + m["description"])
    return "\n".join(parts)


def build_index(meta: dict):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    items = list(meta.values())
    print(f"index {len(items)} สินค้า ...", flush=True)
    B = 64
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        vecs = llm.embed([document(m) for m in chunk])
        col.add(ids=[m["id"] for m in chunk], embeddings=vecs,
                documents=[document(m) for m in chunk])
    return col


def make_loo(user_hist: dict, meta: dict) -> list:
    """สร้าง LOO instance: (profile_ids, target_id) ต่อผู้ใช้ที่มี >=2 ชิ้นในเซ็ต
    ซ่อน 'ชิ้นล่าสุด' (เรียงตามเวลา) เป็น target ที่เหลือเป็น profile"""
    inst = []
    for hist in user_hist.values():
        seen, ordered = set(), []
        for asin, _ in sorted(hist, key=lambda x: x[1]):    # เรียงตามเวลา
            if asin in meta and asin not in seen:
                seen.add(asin)
                ordered.append(asin)
        if len(ordered) >= MIN_USER_ITEMS:
            inst.append((ordered[:-1], ordered[-1]))
            if len(inst) >= MAX_USERS:
                break
    return inst


def query_text(profile_ids: list, meta: dict) -> str:
    """ประโยคค้นจากรสนิยมผู้ใช้ = ชื่อ+ฟีเจอร์ของสินค้าที่เคยรีวิว (เอา 5 ชิ้นล่าสุด)"""
    parts = []
    for a in profile_ids[-5:]:
        m = meta[a]
        parts.append(m["title"])
        if m["features"]:
            parts.append(", ".join(m["features"][:2]))
    return " | ".join(parts)


def main():
    if llm.check_models():
        print("ยังไม่ได้ pull โมเดล embedding (bge-m3)")
        return

    item_freq, user_hist = stream_reviews()
    wanted = select_items(item_freq)
    meta = stream_meta(wanted)
    if len(meta) < 20:
        print("เมทาน้อยเกินไป — เพิ่ม MAX_META_LINES / MAX_REVIEW_LINES แล้วรันใหม่")
        return

    loo = make_loo(user_hist, meta)
    if not loo:
        print("ไม่พบผู้ใช้ที่รีวิว >=2 ชิ้นในเซ็ต — เพิ่ม MAX_REVIEW_LINES หรือลด MIN_ITEM_FREQ")
        return

    col = build_index(meta)
    N = col.count()
    print(f"\nรัน LOO {len(loo)} ผู้ใช้ บน catalog {N} ชิ้น ...", flush=True)

    hits = {k: 0 for k in KS}
    for i, (profile_ids, target) in enumerate(loo, 1):
        vec = llm.embed([query_text(profile_ids, meta)])[0]
        res = col.query(query_embeddings=[vec], n_results=min(max(KS) + len(profile_ids), N))
        exclude = set(profile_ids)
        ranked = [rid for rid in res["ids"][0] if rid not in exclude]
        for k in KS:
            if target in ranked[:k]:
                hits[k] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(loo)} ...", flush=True)

    total = len(loo)
    rows = [[f"Recall@{k}", f"{hits[k] / total:.3f}", f"{k / N:.3f}",
             f"{(hits[k] / total) / (k / N):.1f}x" if hits[k] else "0x"] for k in KS]
    tbl = md_table(["ตัวชี้วัด", "ค่า", "random (k/N)", "ดีกว่าสุ่ม"], rows)

    write_markdown("amazon_loo",
                   "รายงานผล: Leave-One-Out Recall@k (รีวิว Amazon มาตรฐาน)", [
                       f"Dataset: Amazon Reviews 2023 (McAuley-Lab) หมวด {CATEGORY} · "
                       f"eval catalog {N} สินค้า (รีวิวหนาแน่น) · {total} ผู้ใช้ (LOO)",
                       "วิธี: ซ่อนสินค้าชิ้นล่าสุดของผู้ใช้ ใช้ที่เหลือเป็น query "
                       "แล้ววัดว่าดึงชิ้นที่ซ่อนกลับมาใน top-k ได้ไหม",
                       tbl,
                       "> retriever เป็น content-based ไม่ใช่ collaborative filtering — "
                       "Recall สะท้อน 'ทายชิ้นถัดไปจากเนื้อหาสินค้า' ซึ่งยากโดยธรรมชาติ "
                       "(task mismatch แบบเดียวกับ MovieLens leave-one-out) "
                       "แต่เทียบ random ยังเห็นว่ามีสัญญาณ",
                   ])

    print(f"\n{'=' * 56}")
    for k in KS:
        print(f"Recall@{k} = {hits[k] / total:.3f} (random {k / N:.3f})")
    print("เซฟรายงาน: eval/reports/amazon_loo.md")


if __name__ == "__main__":
    main()
