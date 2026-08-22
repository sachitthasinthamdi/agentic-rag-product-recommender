# -*- coding: utf-8 -*-
"""วัดคุณภาพระดับบทสนทนา (CRS) ด้วย User Simulator — โดเมนสินค้า

รัน:  python -m eval.eval_simulator [จำนวนเป้าหมาย] [max_turns]
      เช่น  python -m eval.eval_simulator 6 4     (ค่าเริ่มต้น 6 เป้าหมาย, 4 turn)

แนวคิด (มาตรฐานงานวิจัย CRS): ให้ LLM อีกตัวสวมบท "ลูกค้า" ที่มีสินค้าเป้าหมายในใจ
แต่ห้ามพูดชื่อ/แบรนด์/รุ่นตรงๆ ต้องบรรยายความต้องการ (หมวด/การใช้งาน/ฟีเจอร์/งบ) ให้ระบบเดา
เราวัดว่า:
  - Success@t      : ระบบแนะนำสินค้า "ตรงหมวด + อยู่ในงบ" ภายใน t turn กี่ %
  - Average Turns  : เฉลี่ยกี่ turn กว่าจะสำเร็จ (ยิ่งน้อย = planner ถามเก่ง)
  - Exact-item hit : โบนัส — เจอ "ชิ้นเป้าหมายเป๊ะ" กี่ % (พินพอยต์ SKU เดียวใน 600 ชิ้น = ยากโดยธรรมชาติ)
ผลเซฟที่ eval/reports/simulator.md

หมายเหตุ: ใช้ LLM เยอะ (sim 1 + agent ~4 ต่อ turn) จึงกินเวลา — ปรับจำนวนได้ทาง argv
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import math
import time
from collections import defaultdict

from src import catalog, llm
from src.agent import ProductRecAgent
from eval.report_utils import write_markdown, md_table

SIM_SYSTEM = """คุณกำลังสวมบทเป็น "ลูกค้า" ที่อยากซื้อสินค้าชิ้นหนึ่ง กำลังคุยกับผู้ช่วยแนะนำสินค้า (น้องช้อป)

สินค้าที่คุณอยากได้ (ใช้เป็นแนวทางบรรยายเท่านั้น):
- หมวด: {category}
- จุดเด่นที่คุณต้องการ: {features}
- งบของคุณ: ไม่เกิน {budget} ดอลลาร์
- รายละเอียดสินค้า: {description}

กติกาสำคัญ:
- ห้ามพูดชื่อสินค้า แบรนด์ หรือรุ่นนี้ออกมาเด็ดขาด
- เริ่มด้วยความต้องการกว้างๆ (เอาไปใช้ทำอะไร) ก่อน แล้วค่อยเผยรายละเอียด (ฟีเจอร์/งบ) เมื่อผู้ช่วยถาม
- ตอบสั้นๆ เป็นธรรมชาติ ภาษาไทย 1-2 ประโยค เหมือนลูกค้าจริง
- ถ้าผู้ช่วยถามกลับ ให้ตอบเพิ่มตามความต้องการที่สอดคล้องกับสินค้าชิ้นนี้
- อย่าเพิ่งบอกว่าพอใจหรือจบบทสนทนาเอง ให้คุยต่อเรื่อยๆ
พิมพ์เฉพาะข้อความของคุณ (ลูกค้า) เท่านั้น"""


def budget_for(price: float) -> int:
    """งบของลูกค้า = ปัดราคาขึ้นเป็นพหุคูณของ 5 แล้วบวก headroom อีก 5
    การันตีว่าสินค้าเป้าหมายไม่เกินงบเสมอ (budget >= price)"""
    return int(math.ceil(price / 5) * 5) + 5


def pick_targets(products: list[dict], n: int) -> list[dict]:
    """เลือกสินค้าเป้าหมายแบบไดนามิกจากแคตตาล็อก คละหมวด

    ไม่ hardcode id เพราะ ASIN ของ Amazon เปลี่ยนทุกครั้งที่ rebuild แคตตาล็อก
    เกณฑ์: มีฟีเจอร์ + คำอธิบายพอบรรยายได้ + ราคาไม่แพงเกิน (เดโมง่าย)
    """
    by_cat = defaultdict(list)
    for p in products:
        if (p.get("features") and len(p.get("description", "")) >= 30
                and 5 <= p["price"] <= 80):
            by_cat[p["category"]].append(p)
    cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))
    targets, i = [], 0
    while len(targets) < n and cats:
        cat = cats[i % len(cats)]
        if by_cat[cat]:
            targets.append(by_cat[cat].pop(0))   # หยิบหนึ่งชิ้นต่อรอบ วนหมวด
            i += 1
        else:
            cats.remove(cat)                     # หมวดนี้หมดแล้ว เอาออก
    return targets


def sim_message(target: dict, budget: int, history: list[tuple[str, str]]) -> str:
    """ให้ LLM (บทลูกค้า) พิมพ์ข้อความถัดไป จากบทสนทนาที่ผ่านมา"""
    sys_prompt = SIM_SYSTEM.format(
        category=target["category"],
        features=", ".join(target.get("features", [])[:3]) or "-",
        budget=budget,
        description=target.get("description", "")[:150],
    )
    if not history:
        user = "เริ่มบทสนทนา: พิมพ์สิ่งที่คุณอยากได้ (กว้างๆ สั้นๆ)"
    else:
        convo = "\n".join(
            (f"ผู้ช่วย: {t}" if role == "assistant" else f"คุณ: {t}")
            for role, t in history)
        user = convo + "\n\nพิมพ์ข้อความถัดไปของคุณ:"
    return llm.chat([{"role": "user", "content": user}], system=sys_prompt)


def run_one(target: dict, budget: int, max_turns: int):
    """คุย 1 บทสนทนา คืน (success_turn|None, exact_turn|None, log)

    success = แนะนำสินค้าตรงหมวด + อยู่ในงบ (จบบทสนทนาเมื่อสำเร็จ)
    exact   = เจอชิ้นเป้าหมายเป๊ะ (โบนัส — บันทึกถ้าเกิดในเทิร์นที่สำเร็จ)
    """
    agent = ProductRecAgent()
    history: list[tuple[str, str]] = []
    log = []
    success_turn = exact_turn = None
    for turn in range(1, max_turns + 1):
        user_msg = sim_message(target, budget, history)
        history.append(("user", user_msg))
        trace = agent.respond(user_msg)
        history.append(("assistant", trace["reply"]))
        picked = trace.get("picked", [])
        log.append((turn, trace["action"], user_msg[:38],
                    " / ".join(m["title"][:24] for m in picked)[:60]))
        if success_turn is None and any(
                m["category"] == target["category"] and m["price"] <= budget
                for m in picked):
            success_turn = turn
        if exact_turn is None and any(m["id"] == target["id"] for m in picked):
            exact_turn = turn
        if success_turn is not None:      # สำเร็จตามเป้าหลักแล้ว จบบทสนทนา
            break
    return success_turn, exact_turn, log


def main():
    missing = llm.check_models()
    if missing:
        print("ยังไม่ได้ pull โมเดล:", ", ".join(missing))
        return

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    max_turns = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    catalog.build_index()
    products = catalog.load_products()
    targets = pick_targets(products, n)

    print(f"User simulator (สินค้า): {len(targets)} เป้าหมาย · "
          f"สูงสุด {max_turns} turn/บทสนทนา\n")
    t0 = time.perf_counter()

    results = []       # (label, success_turn|None, exact_turn|None)
    rows = []          # per-conversation สำหรับรายงาน
    for target in targets:
        budget = budget_for(target["price"])
        label = f'{target["title"][:26]} ({target["category"]}, งบ ${budget})'
        print(f"เป้าหมาย: {label} ...", flush=True)
        success_turn, exact_turn, log = run_one(target, budget, max_turns)
        results.append((label, success_turn, exact_turn))
        if success_turn:
            status = f"สำเร็จ turn {success_turn}" + (" (เป๊ะ)" if exact_turn else "")
        else:
            status = "ไม่สำเร็จ"
        rows.append([label, status, " | ".join(f"t{t}:{a}" for t, a, _, _ in log)])
        print(f"  -> {status}")

    wall = time.perf_counter() - t0

    # ---------- คำนวณตัวชี้วัด ----------
    total = len(results)

    def success_at(t):
        return (sum(1 for _, st, _ in results if st is not None and st <= t) / total
                if total else 0.0)

    successes = [st for _, st, _ in results if st is not None]
    avg_turns = sum(successes) / len(successes) if successes else 0.0
    exact_rate = (sum(1 for _, _, ex in results if ex is not None) / total
                  if total else 0.0)

    metric_tbl = md_table(
        ["ตัวชี้วัด", "ค่า"],
        [["Success@1", f"{success_at(1):.1%}"],
         [f"Success@{min(3, max_turns)}", f"{success_at(min(3, max_turns)):.1%}"],
         [f"Success@{max_turns} (รวม)", f"{success_at(max_turns):.1%}"],
         ["Average Turns to success", f"{avg_turns:.2f}" if successes else "—"],
         ["Exact-item hit (โบนัส)", f"{exact_rate:.1%}"],
         ["จำนวนบทสนทนา", total]])

    detail_tbl = md_table(["สินค้าเป้าหมาย", "ผล", "action แต่ละ turn"], rows)

    write_markdown("simulator", "รายงานผล: CRS User-Simulator Metrics (สินค้า)", [
        f"จำลอง {total} บทสนทนา · สูงสุด {max_turns} turn · เวลารวม {wall / 60:.1f} นาที",
        "## 1. ตัวชี้วัดหลัก",
        metric_tbl,
        "> **Success@t** = แนะนำสินค้าตรงหมวด+อยู่ในงบ ภายใน t turn · "
        "**Average Turns** ยิ่งน้อย = planner ถามตรงจุด · "
        "**Exact-item hit** = พินพอยต์ชิ้นเป้าหมายเป๊ะ (ยากเมื่อมีสินค้าคล้ายกันเยอะใน 600 ชิ้น)",
        "## 2. รายละเอียดแต่ละบทสนทนา",
        detail_tbl,
        "> ลูกค้าจำลอง (LLM) ห้ามเอ่ยชื่อ/แบรนด์ ต้องบรรยายความต้องการให้ระบบเดา — "
        "จึงทดสอบ retriever + reranker + planner พร้อมกัน "
        "(งบเป็น hard filter → สินค้าที่แนะนำอยู่ในงบเสมอ)",
    ])

    print(f"\n{'=' * 56}")
    print(f"Success@{max_turns}: {success_at(max_turns):.1%} | "
          f"Success@1: {success_at(1):.1%} | Avg turns: {avg_turns:.2f} | "
          f"Exact-hit: {exact_rate:.1%}")
    print("เซฟรายงาน: eval/reports/simulator.md")


if __name__ == "__main__":
    main()
