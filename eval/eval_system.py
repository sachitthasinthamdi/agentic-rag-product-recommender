# -*- coding: utf-8 -*-
"""วัดตัวชี้วัดระบบ: latency ต่อ turn/ต่อชนิด call + JSON failure rate

รัน:  python -m eval.eval_system

รันชุดข้อความตัวแทน (ครอบคลุมทั้ง recommend/ask/chat) ผ่าน agent จริง
แล้ววัดเวลาต่อ turn + แยกเวลาต่อชนิดการเรียกโมเดล + นับ JSON ที่พัง
ผลเซฟที่ eval/reports/system.md
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import time

from src import catalog, llm
from src.agent import ProductRecAgent
from eval.report_utils import write_markdown, md_table

# ข้อความตัวแทนแต่ละ action (ใช้ agent ใหม่ต่อข้อความ เพื่อวัด latency สะอาด)
PROMPTS = [
    ("หาหูฟังไร้สายไว้ออกกำลังกาย งบไม่เกิน 40", "คาดว่า recommend"),
    ("ขอลำโพงบลูทูธพกพา งบไม่เกิน 30", "คาดว่า recommend"),
    ("อยากได้ของขวัญสักอย่าง ยังไม่รู้จะเอาอะไร", "คาดว่า ask"),
    ("ร้านนี้ขายอะไรบ้างครับ", "คาดว่า chat"),
    ("หาเคสมือถือกันกระแทก งบไม่เกิน 15", "คาดว่า recommend"),
]


def main():
    missing = llm.check_models()
    if missing:
        print("ยังไม่ได้ pull โมเดล:", ", ".join(missing))
        return

    catalog.build_index()
    print(f"วัดระบบบน {len(PROMPTS)} ข้อความ ...\n")

    llm.reset_stats()
    per_turn = []
    total_t0 = time.perf_counter()

    for prompt, note in PROMPTS:
        agent = ProductRecAgent()        # เริ่มใหม่ทุกครั้ง วัด latency สะอาด
        t0 = time.perf_counter()
        trace = agent.respond(prompt)
        dt = time.perf_counter() - t0
        per_turn.append([prompt[:26], trace["action"], f"{dt:.1f}"])
        print(f"  {trace['action']:10} {dt:5.1f} วิ  {prompt[:30]}")

    wall = time.perf_counter() - total_t0
    s = llm.STATS

    def avg(kind):
        return s[kind]["seconds"] / s[kind]["calls"] if s[kind]["calls"] else 0.0

    turn_tbl = md_table(["ข้อความ", "action", "เวลา (วิ)"], per_turn)

    call_tbl = md_table(
        ["ชนิดการเรียก", "โมเดล", "จำนวนครั้ง", "เวลาเฉลี่ย (วิ)"],
        [["chat (เขียน/ถาม/คุย)", "Typhoon2", s["chat"]["calls"], f"{avg('chat'):.2f}"],
         ["chat_json (สกัด/วางแผน/ให้คะแนน)", "Typhoon2", s["chat_json"]["calls"], f"{avg('chat_json'):.2f}"],
         ["embed (ค้น)", "bge-m3", s["embed"]["calls"], f"{avg('embed'):.2f}"]])

    total_calls = s["chat_json"]["calls"]
    fails = s["chat_json"]["json_fail"]
    fail_rate = fails / total_calls if total_calls else 0.0
    n_turns = len(PROMPTS)

    write_markdown("system", "รายงานผล: System Metrics (latency + JSON failure)", [
        f"จำนวน turn ที่วัด: {n_turns} · เวลารวม {wall:.1f} วิ · "
        f"เฉลี่ย {wall / n_turns:.1f} วิ/turn",
        "## 1. เวลาต่อ turn (แยกตาม action)",
        turn_tbl,
        "> ask/chat เร็วกว่า recommend เพราะเรียกโมเดลน้อยขั้นกว่า",
        "## 2. เวลาเฉลี่ยต่อชนิดการเรียกโมเดล",
        call_tbl,
        "## 3. JSON failure rate (เฉพาะ chat_json)",
        f"- เรียก chat_json ทั้งหมด: {total_calls} ครั้ง\n"
        f"- ตอบ JSON พังจนต้องใช้ fallback: {fails} ครั้ง\n"
        f"- **JSON failure rate = {fail_rate:.1%}**",
        "> ยิ่งต่ำยิ่งดี — เป็นตัวชี้วัดความเสถียรเฉพาะของ local model "
        "(ระบบมี fallback ทุกจุดจึงไม่ล่มแม้ค่านี้ > 0)",
    ])

    print(f"\n{'=' * 56}")
    print(f"เฉลี่ย {wall / n_turns:.1f} วิ/turn | JSON failure rate {fail_rate:.1%} "
          f"({fails}/{total_calls})")
    print("เซฟรายงาน: eval/reports/system.md")


if __name__ == "__main__":
    main()
