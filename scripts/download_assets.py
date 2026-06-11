#!/usr/bin/env python
"""下载基础实验所需的数据集与 Mistral-7B-Instruct-v0.3 模型。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

LONGEMBED_SUBSETS = ("qmsum", "2wikimqa")
LONGEMBED_SPLITS = ("corpus", "queries", "qrels")


def download_datasets() -> None:
    from datasets import load_dataset

    logger.info("=== LongEmbed: QMSum & 2WikiMultihop ===")
    for name in LONGEMBED_SUBSETS:
        for split in LONGEMBED_SPLITS:
            logger.info("Loading dwzhu/LongEmbed name=%s split=%s", name, split)
            ds = load_dataset("dwzhu/LongEmbed", name=name, split=split)
            logger.info("  -> %d rows", len(ds))

    logger.info("=== ArguAna (mteb/arguana) ===")
    ds = load_dataset("mteb/arguana")
    logger.info("  -> splits: %s", list(ds.keys()))


def download_model(model_id: str, local_dir: Path | None) -> Path:
    from huggingface_hub import snapshot_download

    logger.info("=== Model: %s ===", model_id)
    target = local_dir or (ROOT / "models" / model_id.replace("/", "--"))
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=model_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    logger.info("Model saved to: %s", path)
    return Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "Mistral-7B-Instruct-v0.3",
        help="模型本地目录（默认项目 models/ 下）",
    )
    args = parser.parse_args()

    if not args.skip_datasets:
        download_datasets()
    if not args.skip_model:
        download_model(args.model, args.model_dir)

    logger.info("All downloads finished.")


if __name__ == "__main__":
    main()
