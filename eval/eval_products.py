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


def eval_budget(col):
    """ตรวจ hard filter งบ — ค้นด้วย budget_max แล้วทุกชิ้นต้อง <= งบ"""
    tests = [("cheap electronics accessory", 10),
             ("kitchen home gadget", 25),
             ("sports fitness gear", 20),
             ("kids toy gift", 30),
             ("phone case cover", 15)]
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

    print(f"catalog {n_cat} สินค้า\n[1] category-retrieval ...", flush=True)
    c_agg, c_rows = eval_category(products, col, n_cat)

    print("[2] budget-filter correctness ...", flush=True)
    budget_rate, b_detail = eval_budget(col)

    cat_tbl = md_table(
        ["มุมมอง", "Precision@10", "NDCG@10", "random baseline"],
        [["Category-retrieval (เฉลี่ยทุกหมวด)", f"{c_agg['p10']:.3f}",
          f"{c_agg['ndcg10']:.3f}", f"{c_agg['rand']:.3f}"]])
    per_cat = md_table(["หมวด", "#สินค้า", "P@10", "NDCG@10", "random"], c_rows)
    budget_tbl = md_table(
        ["query", "งบ", "คืนมา", "อยู่ในงบ", "แพงสุด"], b_detail)

    write_markdown("products", "รายงานผล: RAG-Retrieval บนสินค้า Amazon (มาตรฐาน)", [
        f"Dataset: Amazon Reviews 2023 (McAuley-Lab) · {n_cat} สินค้า 6 หมวด · "
        f"label หมวดจาก Amazon (ไม่ใช่เราตั้งเอง)",
        "## มุมมองที่ 1 — Category-retrieval",
        "query = ชื่อหมวด · relevant = สินค้าที่ Amazon ติด label หมวดนั้น",
        cat_tbl,
        f"> retriever ค้นสินค้าตรงหมวดได้ P@10 = {c_agg['p10']:.2f} "
        f"(ดีกว่าสุ่ม {c_agg['p10']/c_agg['rand']:.1f} เท่า) = จับ category semantics ได้",
        "### แยกตามหมวด",
        per_cat,
        "## มุมมองที่ 2 — Budget-filter correctness (เฉพาะโดเมนสินค้า)",
        "ค้นด้วยงบสูงสุด (hard filter) แล้วทุกชิ้นต้องไม่เกินงบ",
        budget_tbl,
        f"> **{budget_rate:.0%} ของสินค้าที่คืนมาอยู่ในงบ** — hard filter ราคาทำงานถูกต้อง "
        "(งบเป็นเงื่อนไขตัวเลขเฉพาะโดเมนสินค้า ต้องกรองแม่นยำ 100%)",
    ])
    write_csv("products_category", ["category", "n", "p@10", "ndcg@10", "random"], c_rows)

    print(f"\n{'=' * 56}")
    print(f"[category] P@10 = {c_agg['p10']:.3f} · NDCG@10 = {c_agg['ndcg10']:.3f} "
          f"(random {c_agg['rand']:.3f})")
    print(f"[budget]   อยู่ในงบ {budget_rate:.0%}")
    print("เซฟรายงาน: eval/reports/products.md")


if __name__ == "__main__":
    main()
