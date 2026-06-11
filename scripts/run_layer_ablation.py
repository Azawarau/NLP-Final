#!/usr/bin/env python
"""层数消融：比较不同 Transformer 层上的 PromptEOL / mean-pooling 表示质量。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import run_basic_experiment  # noqa: E402
from src.llm_encoder import LLMEmbeddingEncoder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def default_layer_grid(num_layers: int) -> list[int]:
    """在 25%/50%/75%/100% 深度及中间层采样。"""
    candidates = sorted(
        {
            max(1, num_layers // 4),
            max(1, num_layers // 2),
            max(1, (3 * num_layers) // 4),
            num_layers,
            1,
        }
    )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="不同 hidden layer 嵌入质量消融")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "basic.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="指定层列表；默认按模型深度自动生成",
    )
    parser.add_argument("--output-dir", type=str, default="results/layer_ablation")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.3")
    if args.model:
        model_name = args.model
    if args.fallback:
        model_name = "Qwen/Qwen2-1.5B-Instruct"

    if args.layers:
        layers = args.layers
    else:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        layers = default_layer_grid(cfg.num_hidden_layers)
        logging.info("Auto layer grid for %d layers: %s", cfg.num_hidden_layers, layers)

    config["model_name"] = model_name
    config["layers"] = layers
    config["output_dir"] = args.output_dir
    config["methods"] = config.get("methods", ["prompteol", "mean"])

    results = run_basic_experiment(config)
    print("\n========== 层数消融汇总 ==========")
    for run_id, metrics in results["runs"].items():
        print(f"\n[{run_id}]")
        for task, scores in metrics.items():
            ndcg = scores.get("ndcg@10", scores.get("ndcg@1", "N/A"))
            print(f"  {task}: ndcg@10={ndcg}")


if __name__ == "__main__":
    main()
