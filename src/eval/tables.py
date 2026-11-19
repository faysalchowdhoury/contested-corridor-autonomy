"""Export benchmark tables as Markdown + CSV + JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence


def write_table(out_dir: Path, name: str, rows: List[Dict], columns: Sequence[str],
                title: str = "", float_fmt: str = "{:.3f}") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)

    # Markdown
    md = [f"### {title}" if title else f"### {name}", ""]
    md.append("| " + " | ".join(columns) + " |")
    md.append("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        md.append("| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |")
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # CSV
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})

    # JSON
    (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str))


def print_table(title: str, rows: List[Dict], columns: Sequence[str]) -> None:
    print(f"\n{title}")
    widths = {c: max(len(c), *(len(_f(r.get(c, ''))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(_f(r.get(c, "")).ljust(widths[c]) for c in columns))


def _f(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)
