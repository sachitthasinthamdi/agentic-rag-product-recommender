# -*- coding: utf-8 -*-
"""วัด RAG-Retrieval บนแคตตาล็อกสินค้า Amazon Reviews 2023 (มาตรฐาน)

รัน:  python -m eval.eval_products

2 มุมมอง (ทั้งคู่ใช้ข้อมูล/label มาตรฐานของ Amazon ไม่ใช่เราตั้งเอง):
  1. Category-retrieval — query = หมวด, relevant = สินค้าที่ Amazon ติด label หมวดนั้น
     วัด Precision@10, NDCG@10 (เหมาะกับ content-based retriever)
  2. Budget-filter correctness — เฉพาะโดเมนสินค้า: ตรวจว่า hard filter งบทำงานถูก
     (ค้นด้วยงบสูงสุด -> ทุกชิ้นต้องไม่เกินงบ)
ผลเซฟที่ eval/reports/products.md
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import json
import math
from collections import defaultdict

from src import llm, catalog
from eval.report_utils import write_markdown, write_csv, md_table


def ndcg_binary(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(i + 1)
              for i, r in enumerate(ranked[:k], 1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, k + 1))
    return dcg / ideal if ideal else 0.0


def eval_category(products, col, n_cat):
    by_cat = defaultdict(set)
    for p in products:
        by_cat[p["category"]].add(p["id"])

    agg = {"p10": 0.0, "ndcg10": 0.0, "rand": 0.0}
    rows = []
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        relevant = by_cat[cat]
        vec = llm.embed([cat])[0]
        res = col.query(query_embeddings=[vec], n_results=10)
        ranked = res["ids"][0]
        p10 = sum(1 for r in ranked[:10] if r in relevant) / 10
        ndcg = ndcg_binary(ranked, relevant, 10)
        rand = len(relevant) / n_cat
        agg["p10"] += p10; agg["ndcg10"] += ndcg; agg["rand"] += rand
        rows.append([cat, len(relevant), f"{p10:.2f}", f"{ndcg:.2f}", f"{rand:.3f}"])
    for k in agg:
        agg[k] /= len(by_cat)
    return agg, rows


def eval_subcategory(products, col, n_total, min_items=30):
    """Retrieval eval ระดับ subcategory (query จำนวนมาก + วัด variance)

    query = ชื่อ subcategory ที่เฉพาะเจาะจงที่สุด (สมาชิกท้ายของ subcategories)
    relevant = สินค้าที่ Amazon ติด label subcategory นั้น (เฉลยจาก taxonomy จริง)
    เลือกเฉพาะ subcategory ที่มีสินค้า >= min_items เพื่อให้เฉลยหนาแน่นพอ
    คืนค่าเฉลี่ย + ส่วนเบี่ยงเบน + ต่ำสุด/สูงสุด ของ P@10, NDCG@10
    """
    import statistics as stt
    by_sub = defaultdict(set)
    for p in products:
        sc = p.get("subcategories") or []
        if len(sc) >= 2:
            by_sub[sc[-1]].add(p["id"])
    subs = [(s, ids) for s, ids in by_sub.items() if len(ids) >= min_items]

    p10s, ndcgs, rands, rows = [], [], [], []
    for sub, relevant in sorted(subs, key=lambda x: -len(x[1])):
        vec = llm.embed([sub])[0]
        res = col.query(query_embeddings=[vec], n_results=10)
        ranked = res["ids"][0]
        p10 = sum(1 for r in ranked[:10] if r in relevant) / 10
        ndcg = ndcg_binary(ranked, relevant, 10)
        rand = len(relevant) / n_total
        p10s.append(p10); ndcgs.append(ndcg); rands.append(rand)
        rows.append([sub, len(relevant), f"{p10:.2f}", f"{ndcg:.2f}", f"{rand:.3f}"])

    agg = {"n": len(subs),
           "p10_mean": stt.mean(p10s), "p10_std": stt.pstdev(p10s),
           "p10_min": min(p10s), "p10_max": max(p10s),
           "ndcg_mean": stt.mean(ndcgs), "ndcg_std": stt.pstdev(ndcgs),
           "rand_mean": stt.mean(rands)}
    return agg, rows


def eval_budget(col):
    """ตรวจ hard filter งบ — ค้นด้วย budget_max แล้วทุกชิ้นต้อง <= งบ"""
    tests = [("cheap electronics accessory", 10),
             ("kitchen home gadget", 25),
             ("sports fitness gear", 20),
             ("kids toy gift", 30),
             ("phone case cover", 15),
             ("wireless bluetooth headphones", 35),
             ("office desk organizer", 20),
             ("water bottle for gym", 15),
             ("laptop stand aluminum", 40),
             ("board game for family", 25),
             ("smartwatch band strap", 12),
             ("notebook and pens set", 10),
             ("coffee mug ceramic", 18),
             ("usb charging cable", 8),
             ("yoga mat exercise", 30),
             ("action figure collectible", 22),
             ("wall art poster decor", 20),
             ("computer mouse wireless", 25),
             ("cooking utensil set", 28),
             ("fishing tackle kit", 35)]
    total, within, detail = 0, 0, []
    for q, budget in tests:
        vec = llm.embed([q])[0]
        res = col.query(query_embeddings=[vec], n_results=10,
                        where={"price": {"$lte": float(budget)}})
        prices = [json.loads(m["json"])["price"] for m in res["metadatas"][0]]
        ok = sum(1 for pr in prices if pr <= budget)
        total += len(prices); within += ok
        detail.append([q, f"${budget}", len(prices), ok,
                       f"${max(prices):.2f}" if prices else "-"])
    return (within / total if total else 0.0), detail


def main():
    missing = llm.check_models()
    if missing:
        print("ยังไม่ได้ pull โมเดล:", ", ".join(missing))
        return

    n_cat = catalog.build_index()
    products = catalog.load_products()
    col = catalog.get_collection()
    llm.reset_stats()

    print(f"catalog {n_cat} สินค้า\n[1] category-retrieval (6 query) ...", flush=True)
    c_agg, c_rows = eval_category(products, col, n_cat)

    print("[2] subcategory-retrieval (query จำนวนมาก + variance) ...", flush=True)
    s_agg, s_rows = eval_subcategory(products, col, n_cat, min_items=30)

    print("[3] budget-filter correctness (20 query) ...", flush=True)
    budget_rate, b_detail = eval_budget(col)

    cat_tbl = md_table(
        ["มุมมอง", "Precision@10", "NDCG@10", "random baseline"],
        [["Category-retrieval (เฉลี่ย 6 หมวด)", f"{c_agg['p10']:.3f}",
          f"{c_agg['ndcg10']:.3f}", f"{c_agg['rand']:.3f}"]])
    per_cat = md_table(["หมวด", "#สินค้า", "P@10", "NDCG@10", "random"], c_rows)
    sub_tbl = md_table(
        ["ตัวชี้วัด", "ค่าเฉลี่ย", "ส่วนเบี่ยงเบน (SD)", "ต่ำสุด", "สูงสุด"],
        [["Precision@10", f"{s_agg['p10_mean']:.3f}", f"{s_agg['p10_std']:.3f}",
          f"{s_agg['p10_min']:.2f}", f"{s_agg['p10_max']:.2f}"],
         ["NDCG@10", f"{s_agg['ndcg_mean']:.3f}", f"{s_agg['ndcg_std']:.3f}", "-", "-"]])
    per_sub = md_table(["subcategory", "#สินค้า", "P@10", "NDCG@10", "random"], s_rows)
    budget_tbl = md_table(
        ["query", "งบ", "คืนมา", "อยู่ในงบ", "แพงสุด"], b_detail)

    write_markdown("products", "รายงานผล: RAG-Retrieval บนสินค้า Amazon (มาตรฐาน)", [
        f"Dataset: Amazon Reviews 2023 (McAuley-Lab) · {n_cat} สินค้า 6 หมวด · "
        f"label หมวด/subcategory จาก Amazon (ไม่ใช่เราตั้งเอง)",
        "## มุมมองที่ 1 — Category-retrieval (6 query)",
        "query = ชื่อหมวด · relevant = สินค้าที่ Amazon ติด label หมวดนั้น",
        cat_tbl,
        f"> retriever ค้นสินค้าตรงหมวดได้ P@10 = {c_agg['p10']:.2f} "
        f"(ดีกว่าสุ่ม {c_agg['p10']/c_agg['rand']:.1f} เท่า)",
        "### แยกตามหมวด",
        per_cat,
        f"## มุมมองที่ 2 — Subcategory-retrieval ({s_agg['n']} query + variance)",
        f"query = ชื่อ subcategory เฉพาะเจาะจง ({s_agg['n']} subcategory ที่มีสินค้า >= 30 ชิ้น) · "
        "relevant = สินค้าที่ติด label subcategory นั้น — sample ใหญ่กว่า จึงน่าเชื่อถือกว่า",
        sub_tbl,
        f"> เฉลี่ย P@10 = {s_agg['p10_mean']:.3f} (SD {s_agg['p10_std']:.3f}, "
        f"ช่วง {s_agg['p10_min']:.2f}–{s_agg['p10_max']:.2f}) จาก {s_agg['n']} query "
        f"· ดีกว่าสุ่ม {s_agg['p10_mean']/s_agg['rand_mean']:.0f} เท่า",
        "### แยกราย subcategory (บางส่วน)",
        per_sub,
        "## มุมมองที่ 3 — Budget-filter correctness (20 query)",
        "ค้นด้วยงบสูงสุด (hard filter) แล้วทุกชิ้นต้องไม่เกินงบ",
        budget_tbl,
        f"> **{budget_rate:.0%} ของสินค้าที่คืนมาอยู่ในงบ** — hard filter ราคาทำงานถูกต้อง "
        "(deterministic โดยโค้ด ไม่ใช่การประมาณเชิงสถิติ)",
    ])
    write_csv("products_category", ["category", "n", "p@10", "ndcg@10", "random"], c_rows)
    write_csv("products_subcategory", ["subcategory", "n", "p@10", "ndcg@10", "random"], s_rows)

    print(f"\n{'=' * 56}")
    print(f"[category 6q]     P@10 = {c_agg['p10']:.3f} · NDCG@10 = {c_agg['ndcg10']:.3f}")
    print(f"[subcategory {s_agg['n']}q] P@10 = {s_agg['p10_mean']:.3f} "
          f"(SD {s_agg['p10_std']:.3f}, {s_agg['p10_min']:.2f}-{s_agg['p10_max']:.2f}) · "
          f"NDCG@10 = {s_agg['ndcg_mean']:.3f}")
    print(f"[budget 20q]      อยู่ในงบ {budget_rate:.0%}")
    print("เซฟรายงาน: eval/reports/products.md")


if __name__ == "__main__":
    main()
