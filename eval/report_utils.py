# -*- coding: utf-8 -*-
"""ตัวช่วยเขียนรายงานผล eval เป็นไฟล์ Markdown + CSV (เก็บใน eval/reports/)"""
import csv
import datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def md_table(headers: list[str], rows: list[list]) -> str:
    """สร้างตาราง Markdown จาก header + rows"""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_markdown(name: str, title: str, sections: list[str]) -> Path:
    """เขียนไฟล์ .md — sections คือ list ของ block ข้อความ/ตาราง"""
    REPORTS_DIR.mkdir(exist_ok=True)
    lines = [f"# {title}", "", f"_สร้างเมื่อ {_timestamp()}_", ""]
    for sec in sections:
        lines.append(sec)
        lines.append("")
    path = REPORTS_DIR / f"{name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_csv(name: str, headers: list[str], rows: list[list]) -> Path:
    """เขียนไฟล์ .csv (utf-8-sig เพื่อให้ Excel เปิดภาษาไทยได้)"""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path
