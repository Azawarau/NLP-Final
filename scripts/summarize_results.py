#!/usr/bin/env python
"""将 basic_results.json 打印为 Markdown 表格，便于填入报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, default=Path("results/basic/basic_results.json"), nargs="?")
    args = parser.parse_args()
    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    runs = data.get("runs", data)

    tasks = set()
    for metrics in runs.values():
        tasks.update(metrics.keys())
    tasks = sorted(tasks)

    header = "| Run | " + " | ".join(tasks) + " |"
    sep = "|" + "---|" * (len(tasks) + 1)
    print(header)
    print(sep)
    for run_id, metrics in sorted(runs.items()):
        cells = [run_id]
        for t in tasks:
            m = metrics.get(t, {})
            val = m.get("ndcg@10", m.get("ndcg@1", "-"))
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
