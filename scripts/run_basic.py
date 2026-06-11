#!/usr/bin/env python
"""基础实验：PromptEOL vs mean-pooling，三数据集 MTEB 评估。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import run_basic_experiment  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP 基础实验：PromptEOL & mean-pooling")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "basic.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖 config 中的 model_name",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="使用 Qwen2-1.5B-Instruct（显存不足时，作业会扣分）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="覆盖结果输出目录",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["prompteol", "mean"],
        default=None,
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="抽取层索引，-1 为最后一层，1..N 为第 k 个 Transformer 层",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="MTEB 任务名，默认 QMSum / 2Wiki / ArguAna",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model_name"] = args.model
    if args.fallback:
        config["model_name"] = "Qwen/Qwen2-1.5B-Instruct"
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.methods:
        config["methods"] = args.methods
    if args.layers:
        config["layers"] = args.layers
    if args.tasks:
        config["tasks"] = args.tasks
    if args.max_length:
        config["max_length"] = args.max_length
    if args.batch_size:
        config["batch_size"] = args.batch_size

    results = run_basic_experiment(config)
    print("\n========== 基础实验汇总 ==========")
    for run_id, metrics in results["runs"].items():
        print(f"\n[{run_id}]")
        for task, scores in metrics.items():
            print(f"  {task}: {scores}")


if __name__ == "__main__":
    main()
