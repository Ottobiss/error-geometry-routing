#!/usr/bin/env python3
"""Проверяет карточки literature/papers/*.md и собирает literature/index.md.

    python scripts/lit_index.py           # проверить и собрать index.md
    python scripts/lit_index.py --check   # только проверить (код 1 при ошибках)

Нужен pyyaml: pip install pyyaml
"""
import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "literature" / "papers"
INDEX = ROOT / "literature" / "index.md"

STATUS = {"to-read": "не читали", "abstract": "аннотация", "full": "полный текст"}
USEFUL = ["yes", "no", "?"]
REQUIRED = ["title", "url", "added_by", "status"]


def parse(path):
    text = path.read_text(encoding="utf-8")
    try:
        _, head, _ = text.split("---", 2)
        meta = yaml.safe_load(head) or {}
    except Exception:  # noqa: BLE001
        return None, ["нет или не читается YAML-шапка"]

    errors = [f"пустое поле `{k}`" for k in REQUIRED if not meta.get(k)]
    if meta.get("status") and meta["status"] not in STATUS:
        errors.append(f"`status: {meta['status']}` — допустимо: {', '.join(STATUS)}")
    meta["useful"] = str(meta.get("useful") or "?")
    if meta["useful"] not in USEFUL:
        errors.append(f"`useful: {meta['useful']}` — допустимо: {', '.join(USEFUL)}")
    rb = meta.get("read_by") or []
    meta["read_by"] = rb if isinstance(rb, list) else [rb]
    if meta.get("status") in ("abstract", "full") and not meta["read_by"]:
        errors.append("статья прочитана, но `read_by` пуст")
    meta["_file"] = path.name
    return meta, errors


def cell(v):
    if isinstance(v, list):
        v = ", ".join(map(str, v))
    return str(v or "").replace("|", "\\|")


def build(papers):
    order = {"yes": 0, "?": 1, "no": 2}
    papers = sorted(papers, key=lambda p: (order.get(p["useful"], 3), p["_file"]))
    link = lambda p: f"[{cell(p['title'])}](papers/{p['_file']})"  # noqa: E731

    st = Counter(p.get("status") for p in papers)
    out = [
        "# Литература",
        "",
        f"_Собрано `scripts/lit_index.py` {date.today().isoformat()}, руками не править._",
        "",
        f"Всего: **{len(papers)}** — "
        + ", ".join(f"{name}: {st.get(k, 0)}" for k, name in STATUS.items()),
        "",
    ]

    readers = Counter(r for p in papers for r in p["read_by"])
    if readers:
        out += ["**Кто сколько прочитал:** "
                + ", ".join(f"{r} — {n}" for r, n in sorted(readers.items())), ""]

    todo = [p for p in papers if p["useful"] == "yes" and p.get("status") != "full"]
    if todo:
        out += ["**Полезные, но не прочитаны полностью:**", ""]
        out += [f"- {link(p)}" for p in todo]
        out.append("")

    out += ["| Статья | Статус | Полезно | Кто читал | Добавил |",
            "|---|---|---|---|---|"]
    for p in papers:
        out.append(f"| {link(p)} ([src]({p['url']})) | {STATUS.get(p.get('status'), '?')} "
                   f"| {p['useful']} | {cell(p['read_by'])} | {cell(p.get('added_by'))} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="только проверка")
    args = ap.parse_args()

    papers, bad = [], 0
    for path in sorted(PAPERS.glob("*.md")):
        meta, errors = parse(path)
        for e in errors:
            print(f"{path.relative_to(ROOT)}: {e}")
        bad += bool(errors)
        if meta:
            papers.append(meta)

    print(f"Карточек: {len(papers)}, с ошибками: {bad}")
    if args.check:
        sys.exit(1 if bad else 0)
    INDEX.write_text(build(papers), encoding="utf-8")
    print(f"Записан {INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
