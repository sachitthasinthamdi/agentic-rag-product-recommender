# -*- coding: utf-8 -*-
"""สร้าง product catalog จาก Amazon Reviews 2023 (McAuley-Lab, HuggingFace)

วิธี: stream ไฟล์ metadata ทีละบรรทัด (JSONL) เก็บเฉพาะสินค้าที่มีราคา/ชื่อ/คำอธิบาย
หยุดเมื่อครบ N ต่อหมวด — ไม่ต้องโหลดไฟล์ทั้ง GB และไม่ต้องพึ่ง datasets lib
(dataset: CC BY 4.0 / research use — cite McAuley-Lab Amazon Reviews 2023)

รัน:  python data/build_products.py
ได้:  data/products.json (เต็ม) + data/products.sample.json (~80 สำหรับเดโม)
"""
import json
import sys
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BASE = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        "/resolve/main/raw/meta_categories")
OUT_DIR = Path(__file__).resolve().parent

# หมวดอีคอมเมิร์ซทั่วไปที่มักมีราคา + ฟีเจอร์ (ชื่อไฟล์ตาม dataset)
CATEGORIES = [
    "Electronics", "Home_and_Kitchen", "Office_Products",
    "Toys_and_Games", "Sports_and_Outdoors", "Cell_Phones_and_Accessories",
]
N_PER_CAT = 1700         # เก็บกี่ชิ้นต่อหมวด (×6 หมวด ≈ 10,000 ชิ้น หลัง dedup)
MAX_LINES = 400000       # อ่านมากสุดต่อหมวด (กันหมวดที่ราคาว่างเยอะ; loop เบรกเองเมื่อครบ N)


def parse_price(p):
    """ราคาอาจเป็น None / ตัวเลข / สตริง '$29.99' — คืน float หรือ None"""
    if p is None:
        return None
    if isinstance(p, (int, float)):
        return float(p) if p > 0 else None
    s = str(p).replace("$", "").replace(",", "").strip()
    if not s:
        return None
    try:
        v = float(s.split()[0])
        return v if 0 < v < 100000 else None
    except ValueError:
        return None


def clean_list(x, limit):
    if not isinstance(x, list):
        return []
    return [str(s).strip() for s in x if str(s).strip()][:limit]


def parse_image(images) -> str:
    """ดึง URL รูปหลักจาก field images ของ Amazon (list ของ dict) — เลือก variant MAIN
    รูปเป็น URL บน Amazon CDN ไว้ให้ Streamlit โหลดมาแสดงในการ์ด"""
    if not isinstance(images, list) or not images:
        return ""
    main = next((im for im in images
                 if isinstance(im, dict) and im.get("variant") == "MAIN"), None)
    if main is None and isinstance(images[0], dict):
        main = images[0]
    if not isinstance(main, dict):
        return ""
    return (main.get("large") or main.get("hi_res") or main.get("thumb") or "").strip()


def normalize(d: dict) -> dict | None:
    title = (d.get("title") or "").strip()
    price = parse_price(d.get("price"))
    desc = " ".join(clean_list(d.get("description"), 10))
    if not title or price is None or len(desc) < 20:
        return None                    # ต้องมี ชื่อ+ราคา+คำอธิบาย
    return {
        "id": d.get("parent_asin"),
        "title": title[:140],
        "category": d.get("main_category") or "",
        "subcategories": clean_list(d.get("categories"), 3),
        "price": round(price, 2),
        "brand": (d.get("store") or "").strip()[:60],
        "rating": d.get("average_rating"),
        "rating_count": d.get("rating_number") or 0,
        "features": clean_list(d.get("features"), 5),
        "description": desc[:500],
        "image": parse_image(d.get("images")),
    }


def stream_category(cat: str, n: int) -> list[dict]:
    url = f"{BASE}/meta_{cat}.jsonl"
    print(f"  stream {cat} ...", flush=True)
    out, lines = [], 0
    try:
        r = requests.get(url, stream=True, timeout=90)
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            lines += 1
            if lines > MAX_LINES:
                break
            if not line:
                continue
            try:
                prod = normalize(json.loads(line))
            except json.JSONDecodeError:
                continue
            if prod and prod["id"]:
                prod["category"] = cat.replace("_", " ")
                out.append(prod)
                if len(out) >= n:
                    break
        r.close()
    except requests.RequestException as e:
        print(f"    ! {cat} error: {e}", flush=True)
    print(f"    got {len(out)} (อ่าน {lines} บรรทัด)", flush=True)
    return out


def main():
    all_products = []
    for cat in CATEGORIES:
        all_products += stream_category(cat, N_PER_CAT)

    # กัน id ซ้ำข้ามหมวด
    seen, unique = set(), []
    for p in all_products:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)

    (OUT_DIR / "products.json").write_text(
        json.dumps(unique, ensure_ascii=False, indent=1), encoding="utf-8")

    # sample: เอาคละหมวด ~80 ชิ้น (เดโมได้โดยไม่ต้อง stream)
    by_cat = {}
    for p in unique:
        by_cat.setdefault(p["category"], []).append(p)
    sample = []
    for cat, items in by_cat.items():
        sample += items[:14]
    (OUT_DIR / "products.sample.json").write_text(
        json.dumps(sample[:84], ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nรวม {len(unique)} สินค้า ({len(by_cat)} หมวด) -> products.json")
    print(f"sample {len(sample[:84])} ชิ้น -> products.sample.json")
    for cat, items in by_cat.items():
        print(f"  {cat}: {len(items)}")


if __name__ == "__main__":
    main()
