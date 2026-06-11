"""Run MTEB retrieval evaluation for basic experiments."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.llm_encoder import LLMEmbeddingEncoder

logger = logging.getLogger(__name__)

# LongEmbed tasks + ArguAna (BEIR)
BASIC_TASKS = [
    "LEMBQMSumRetrieval",
    "LEMBWikimQARetrieval",
    "ArguAna",
]


def _extract_metrics_from_scores(scores: dict[str, Any]) -> dict[str, float]:
    """Pull common retrieval metrics from a scores dict."""
    metrics: dict[str, float] = {}
    key_map = {
        "ndcg_at_1": "ndcg@1",
        "ndcg_at_10": "ndcg@10",
        "map_at_1": "map@1",
        "map_at_10": "map@10",
        "mrr_at_10": "mrr@10",
        "recall_at_1": "recall@1",
        "recall_at_10": "recall@10",
    }
    for src, dst in key_map.items():
        if src in scores:
            metrics[dst] = float(scores[src])
    return metrics


def _extract_metrics(task_result: dict[str, Any] | Any) -> dict[str, float]:
    """Pull metrics from MTEB 1.x dict or 2.x TaskResult."""
    if hasattr(task_result, "scores"):
        scores_dict = task_result.scores
        split = "test" if "test" in scores_dict else next(iter(scores_dict))
        # scores[split] is list of subset score dicts; take first or average
        subsets = scores_dict[split]
        if subsets:
            return _extract_metrics_from_scores(subsets[0])
        return {}
    split = "test" if "test" in task_result else "validation"
    return _extract_metrics_from_scores(task_result[split])


def _resolve_task_names(tasks: list[str]) -> list[str]:
    """Map friendly names to installed MTEB task registry names."""
    try:
        import mteb

        available = {t.metadata.name for t in mteb.get_tasks()}
    except Exception:
        return tasks

    aliases = {
        "ArguAna": ["ArguAna", "ArguAnaRetrieval"],
        "LEMBWikimQARetrieval": ["LEMBWikimQARetrieval", "LEMB2WikiMultihopRetrieval"],
    }
    resolved: list[str] = []
    for task in tasks:
        candidates = aliases.get(task, [task])
        picked = next((c for c in candidates if c in available), task)
        if picked not in available:
            logger.warning("Task %s not in MTEB registry; trying as-is", task)
        resolved.append(picked)
    return resolved


def run_mteb_evaluation(
    model: LLMEmbeddingEncoder,
    tasks: list[str],
    output_dir: str,
    batch_size: int | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Evaluate encoder on MTEB retrieval tasks (mteb 1.x and 2.x)."""
    from src.mteb_wrapper import wrap_for_mteb

    task_names = _resolve_task_names(tasks)
    os.makedirs(output_dir, exist_ok=True)
    bs = batch_size or model.batch_size
    wrapped = wrap_for_mteb(model)

    try:
        import mteb

        mteb_tasks = mteb.get_tasks(tasks=task_names)
        strategy = "always" if overwrite else "only-missing"
        model_result = mteb.evaluate(
            wrapped,
            tasks=mteb_tasks,
            encode_kwargs={"batch_size": bs},
            overwrite_strategy=strategy,
            prediction_folder=output_dir,
            show_progress_bar=True,
        )
        summary: dict[str, Any] = {}
        for task_result in model_result.task_results:
            summary[task_result.task_name] = _extract_metrics(task_result)
        return summary
    except TypeError:
        # Fallback: deprecated MTEB().run for older installs
        from mteb import MTEB

        evaluation = MTEB(tasks=task_names)
        results = evaluation.run(
            model,
            output_folder=output_dir,
            overwrite_results=overwrite,
            batch_size=bs,
            verbosity=1,
        )
        summary = {}
        for task_name, task_result in results.items():
            summary[task_name] = _extract_metrics(task_result)
        return summary


def run_basic_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run PromptEOL / mean-pooling and optional layer ablation."""
    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    methods: list[str] = config.get("methods", ["prompteol", "mean"])
    layers: list[int] = config.get("layers", [-1])
    tasks: list[str] = config.get("tasks", BASIC_TASKS)
    all_results: dict[str, Any] = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "runs": {},
    }

    for method in methods:
        for layer in layers:
            run_id = f"{method}_layer{layer}"
            logger.info("=== Run: %s ===", run_id)
            encoder = LLMEmbeddingEncoder(
                model_name_or_path=config.get(
                    "model_name", "mistralai/Mistral-7B-Instruct-v0.3"
                ),
                method=method,  # type: ignore[arg-type]
                layer=layer,
                max_length=config.get("max_length", 8192),
                batch_size=config.get("batch_size", 8),
                device=config.get("device"),
                torch_dtype=config.get("torch_dtype", "auto"),
                normalize=config.get("normalize", True),
                use_chat_template=config.get("use_chat_template", False),
                load_in_4bit=config.get("load_in_4bit", False),
                load_in_8bit=config.get("load_in_8bit", False),
            )

            run_dir = output_root / run_id
            try:
                metrics = run_mteb_evaluation(
                    encoder,
                    tasks=tasks,
                    output_dir=str(run_dir),
                    batch_size=config.get("batch_size"),
                    overwrite=config.get("overwrite", True),
                )
            finally:
                del encoder
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_results["runs"][run_id] = metrics
            _save_json(output_root / f"{run_id}.json", metrics)

    _save_json(output_root / "basic_results.json", all_results)
    return all_results


def _save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", path)
