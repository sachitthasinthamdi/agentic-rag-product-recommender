# -*- coding: utf-8 -*-
"""โหมดคุยผ่าน terminal — ไว้เดโม/ดีบักเร็วๆ

รัน:  python cli.py            (คุยปกติ)
      python cli.py --debug    (โชว์ trace: action, profile, candidates ทุก turn)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json

from src import catalog, llm
from src.agent import ProductRecAgent


def main():
    debug = "--debug" in sys.argv

    missing = llm.check_models()
    if missing:
        print("ยังไม่ได้ดาวน์โหลดโมเดล:", ", ".join(missing))
        print("รัน:  ollama pull " + "  &&  ollama pull ".join(missing))
        return

    n = catalog.build_index()
    print(f"คลังสินค้าพร้อม: {n} ชิ้น")
    print("=" * 50)
    print("AI Assistant: สวัสดีค่า~ วันนี้อยากได้สินค้าอะไรดีคะ บอกหมวด งบ หรือการใช้งานมาได้เลย")
    print("(พิมพ์ 'ออก' เพื่อจบ)")
    print("=" * 50)

    agent = ProductRecAgent()
    while True:
        try:
            user = input("\nคุณ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("ออก", "exit", "quit", "q"):
            print("AI Assistant: ขอให้ช้อปสนุกนะคะ~")
            break

        result = agent.respond(user)

        if debug:
            print(f"\n  [action] {result['action']} — {result['plan_reason']}")
            print(f"  [profile] {json.dumps(result['profile'], ensure_ascii=False)}")
            if result.get("picked"):
                print("  [picked] " + ", ".join(
                    f"{m['title'][:30]} (${m['price']}, fit {m['fit_score']})"
                    for m in result["picked"]))

        print(f"\nAI Assistant: {result['reply']}")


if __name__ == "__main__":
    main()
