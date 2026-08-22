# -*- coding: utf-8 -*-
"""ทดสอบบทสนทนาอัตโนมัติ (โดเมนสินค้า) — รันสถานการณ์จริงผ่าน agent แล้วเช็คพฤติกรรม

รัน:  python -m eval.test_dialogs

ตรวจ 3 อย่างต่อสถานการณ์:
  1. action ที่ planner เลือก อยู่ในเซ็ตที่ยอมรับได้
  2. ถ้าแนะนำ ต้องแนะนำจากแคตตาล็อกจริง + อยู่ในงบ (budget filter)
  3. เงื่อนไขเฉพาะ เช่น ปฏิเสธแล้วห้ามซ้ำ, profile จับ budget/category ได้
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from src import catalog
from src.agent import ProductRecAgent

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
passed = failed = 0


def check(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  {GREEN}PASS{RESET} {name}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET} {name} {detail}")


def titles(trace) -> list[str]:
    return [m["title"] for m in trace.get("picked", [])]


def run():
    n = catalog.build_index()
    all_ids = {p["id"] for p in catalog.load_products()}
    print(f"คลังสินค้า {n} ชิ้น\n")

    # ---------- 1: ขอชัด + งบ -> recommend ในงบ จาก catalog ----------
    print("[1] หาเคสมือถือกันกระแทก งบไม่เกิน 15")
    a = ProductRecAgent()
    t = a.respond("หาเคสมือถือกันกระแทก งบไม่เกิน 15 ดอลลาร์")
    check("action = recommend หรือ ask", t["action"] in ("recommend", "ask"),
          f"(ได้ {t['action']})")
    check("profile จับ budget=15", t["profile"].get("budget_max") == 15,
          f"(ได้ {t['profile'].get('budget_max')})")
    if t["action"] == "recommend":
        check("ทุกชิ้นอยู่ใน catalog จริง",
              all(m["id"] in all_ids for m in t["picked"]))
        check("ทุกชิ้นอยู่ในงบ $15",
              all(m["price"] <= 15 for m in t["picked"]),
              f"(ราคา {[m['price'] for m in t['picked']]})")

    # ---------- 2: กำกวม -> เติม -> recommend ----------
    print("\n[2] กำกวม -> เติม preference -> recommend")
    a = ProductRecAgent()
    t1 = a.respond("อยากได้ของใช้ในบ้านสักอย่าง")
    check("turn แรก action สมเหตุผล", t1["action"] in ("ask", "chat", "recommend"))
    t2 = a.respond("เอาของแต่งครัว งบไม่เกิน 40")
    check("หลังบอกชัด -> recommend", t2["action"] == "recommend",
          f"(ได้ {t2['action']})")
    if t2["action"] == "recommend":
        check("แนะนำในงบ $40", all(m["price"] <= 40 for m in t2["picked"]))

    # ---------- 3: ปฏิเสธ -> ห้ามซ้ำ ----------
    print("\n[3] ปฏิเสธสินค้าที่แนะนำ -> ห้ามเสนอซ้ำ")
    a = ProductRecAgent()
    t1 = a.respond("หาลูกบอลโยคะออกกำลังกาย งบไม่เกิน 30")
    if t1["action"] != "recommend":      # planner อาจถามก่อน — ดันให้แนะนำ
        t1 = a.respond("แนะนำมาเลย")
    first = titles(t1)
    if first:
        reject = first[0]
        short = reject[:20]
        t2 = a.respond(f"ไม่เอา \"{short}\" ขออันอื่น")
        check("ไม่แนะนำชิ้นเดิมซ้ำ", reject not in titles(t2),
              f"(ได้ {[x[:20] for x in titles(t2)]})")
    else:
        check("turn แรกควรได้คำแนะนำ", False, f"(action={t1['action']})")

    # ---------- 4: เปลี่ยนงบกลางคัน ----------
    print("\n[4] เปลี่ยนงบ: ตั้ง 20 -> ขยับเป็น 50")
    a = ProductRecAgent()
    a.respond("หาลำโพงบลูทูธ งบไม่เกิน 20")
    t2 = a.respond("เพิ่มงบเป็น 50 ได้")
    check("profile อัปเดต budget=50", t2["profile"].get("budget_max") == 50,
          f"(ได้ {t2['profile'].get('budget_max')})")
    if t2["action"] == "recommend" and t2["picked"]:
        check("แนะนำในงบใหม่ $50", all(m["price"] <= 50 for m in t2["picked"]))

    # ---------- 5: ถามข้อมูล -> chat ----------
    print("\n[5] ถามข้อมูลสินค้า -> chat")
    a = ProductRecAgent()
    t = a.respond("สวัสดีครับ ร้านนี้ขายอะไรบ้าง")
    check("action = chat", t["action"] == "chat", f"(ได้ {t['action']})")

    print(f"\n{'=' * 40}\nสรุป: ผ่าน {passed} / {passed + failed}")
    if failed:
        print(f"{RED}มี {failed} ข้อไม่ผ่าน — LLM ตัวเล็กอาจตอบไม่เสถียร ลองรันซ้ำ{RESET}")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
