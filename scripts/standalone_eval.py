#!/usr/bin/env python
"""独立评估管线：绕过 MTEB，手动编码 + 检索 + 指标计算。

重要：datasets 必须在 torch 之前导入（DLL 冲突规避）。
"""

from __future__ import annotations

# *** 导入顺序关键：datasets 必须在 torch 之前 ***
from datasets import load_dataset  # noqa: E402

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_encoder import LLMEmbeddingEncoder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------

def load_qmsum():
    """QMSum: dwzhu/LongEmbed, name=qmsum"""
    corpus = load_dataset("dwzhu/LongEmbed", name="qmsum", split="corpus")
    queries = load_dataset("dwzhu/LongEmbed", name="qmsum", split="queries")
    qrels = load_dataset("dwzhu/LongEmbed", name="qmsum", split="qrels")
    # Build corpus texts
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [str(doc.get("doc_id", doc.get("_id", str(i)))) for i, doc in enumerate(corpus)]
    query_texts = [q["text"] for q in queries]
    query_ids = [str(q.get("qid", q.get("_id", str(i)))) for i, q in enumerate(queries)]
    # Build qrels dict: query_id -> set of relevant corpus_ids
    qrels_map = defaultdict(set)
    for row in qrels:
        qrels_map[str(row.get("qid", row.get("_id", "")))].add(
            str(row.get("doc_id", row.get("docid", "")))
        )
    return {
        "name": "QMSum",
        "corpus_texts": corpus_texts,
        "corpus_ids": corpus_ids,
        "query_texts": query_texts,
        "query_ids": query_ids,
        "qrels": dict(qrels_map),
    }


def load_2wikimultihop():
    """2WikiMultihop: dwzhu/LongEmbed, name=2wikimqa"""
    corpus = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="corpus")
    queries = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="queries")
    qrels = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="qrels")
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [str(doc.get("doc_id", doc.get("_id", str(i)))) for i, doc in enumerate(corpus)]
    query_texts = [q["text"] for q in queries]
    query_ids = [str(q.get("qid", q.get("_id", str(i)))) for i, q in enumerate(queries)]
    qrels_map = defaultdict(set)
    for row in qrels:
        qrels_map[str(row.get("qid", row.get("_id", "")))].add(
            str(row.get("doc_id", row.get("docid", "")))
        )
    return {
        "name": "2WikiMultihop",
        "corpus_texts": corpus_texts,
        "corpus_ids": corpus_ids,
        "query_texts": query_texts,
        "query_ids": query_ids,
        "qrels": dict(qrels_map),
    }


def load_arguana():
    """ArguAna: mteb/arguana with config='corpus'/'queries' for texts, default for qrels."""
    # Load texts from dedicated configs
    corpus_ds = load_dataset("mteb/arguana", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/arguana", "queries", split="queries")
    qrels_ds = load_dataset("mteb/arguana", split="test")  # default config

    # Build corpus: title + text
    corpus_texts = []
    corpus_ids = []
    for doc in corpus_ds:
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        full_text = f"{title}\n{text}".strip() if title else text
        corpus_texts.append(full_text)
        corpus_ids.append(str(doc["_id"]))

    # Build queries
    query_texts = [q["text"] for q in queries_ds]
    query_ids = [str(q["_id"]) for q in queries_ds]

    # Build qrels from test split (score > 0 = relevant)
    qrels_map = defaultdict(set)
    for row in qrels_ds:
        if float(row["score"]) > 0:
            qrels_map[str(row["query-id"])].add(str(row["corpus-id"]))

    return {
        "name": "ArguAna",
        "corpus_texts": corpus_texts,
        "corpus_ids": corpus_ids,
        "query_texts": query_texts,
        "query_ids": query_ids,
        "qrels": dict(qrels_map),
    }


# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    qrels: dict[str, set[str]],
    query_ids: list[str],
    corpus_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """计算 nDCG, Recall, MAP, MRR 等检索指标."""
    if k_values is None:
        k_values = [1, 10]

    # Normalize embeddings
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9)
    c_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-9)

    # Cosine similarity -> scores
    scores = q_norm @ c_norm.T  # (n_queries, n_corpus)

    # Build corpus_id -> index mapping
    cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    max_k = max(k_values)
    metrics: dict[str, float] = {}
    total_queries = 0

    # Accumulators
    ndcg_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    precision_at_k = {k: [] for k in k_values}
    map_scores = []
    mrr_scores = []

    for qi, qid in enumerate(query_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        total_queries += 1
        relevant = qrels[qid]

        # Get top-k indices sorted by similarity score (descending)
        q_scores = scores[qi]
        top_indices = np.argsort(q_scores)[::-1][:max_k]

        # MRR
        rr = 0.0
        for rank, idx in enumerate(top_indices):
            if corpus_ids[idx] in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)

        # MAP
        ap = 0.0
        relevant_count = 0
        for rank, idx in enumerate(top_indices):
            if corpus_ids[idx] in relevant:
                relevant_count += 1
                ap += relevant_count / (rank + 1)
        if relevant_count > 0:
            ap /= min(len(relevant), max_k)
        map_scores.append(ap)

        # nDCG, Recall, Precision at each k
        for k in k_values:
            top_k_indices = top_indices[:k]
            # Relevance labels: 1 if relevant, 0 otherwise
            y_true = np.zeros(k)
            y_score = np.zeros(k)
            for rank, idx in enumerate(top_k_indices):
                cid = corpus_ids[idx]
                if cid in relevant:
                    y_true[rank] = 1.0
                y_score[k - 1 - rank] = 1.0 / np.log2(rank + 2)  # DCG-like for nDCG

            # nDCG
            dcg = sum((2 ** y_true[i] - 1) / np.log2(i + 2) for i in range(len(y_true)))
            ideal_y = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
            idcg = sum((2 ** ideal_y[i] - 1) / np.log2(i + 2) for i in range(len(ideal_y)))
            ndcg_val = dcg / idcg if idcg > 0 else 0.0
            ndcg_at_k[k].append(ndcg_val)

            # Recall
            recall_val = sum(1 for idx in top_k_indices if corpus_ids[idx] in relevant) / min(len(relevant), k) if len(relevant) > 0 else 0
            recall_at_k[k].append(recall_val)

            # Precision
            precision_val = sum(1 for idx in top_k_indices if corpus_ids[idx] in relevant) / k
            precision_at_k[k].append(precision_val)

    # Average over queries
    for k in k_values:
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_at_k[k])) if ndcg_at_k[k] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0
        metrics[f"precision@{k}"] = float(np.mean(precision_at_k[k])) if precision_at_k[k] else 0.0
    metrics["mrr@10"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    metrics["map@10"] = float(np.mean(map_scores)) if map_scores else 0.0
    metrics["num_queries_evaluated"] = total_queries

    return metrics


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------

DATASET_LOADERS = {
    "QMSum": load_qmsum,
    "2WikiMultihop": load_2wikimultihop,
    "ArguAna": load_arguana,
}


def run_experiment(
    model_path: str,
    method: str,
    layer: int,
    max_length: int,
    batch_size: int,
    output_dir: Path,
    datasets: list[str] | None = None,
    load_in_4bit: bool = True,
) -> dict:
    """Run a single experiment (method × layer) on specified datasets."""
    if datasets is None:
        datasets = list(DATASET_LOADERS.keys())

    logger.info("Loading encoder: %s method=%s layer=%d", model_path, method, layer)
    t0 = time.time()
    enc = LLMEmbeddingEncoder(
        model_name_or_path=model_path,
        method=method,  # type: ignore[arg-type]
        layer=layer,
        max_length=max_length,
        batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        normalize=True,
    )
    logger.info("Encoder loaded in %.1fs. VRAM: %.2f GB", time.time() - t0, torch.cuda.memory_allocated() / 1e9)

    all_metrics: dict[str, dict] = {}

    for ds_name in datasets:
        logger.info("=== Dataset: %s ===", ds_name)
        loader = DATASET_LOADERS[ds_name]
        data = loader()
        logger.info(
            "  corpus: %d, queries: %d, qrels: %d",
            len(data["corpus_texts"]),
            len(data["query_texts"]),
            len(data["qrels"]),
        )

        # Encode corpus
        logger.info("  Encoding corpus (%d texts)...", len(data["corpus_texts"]))
        t1 = time.time()
        corpus_emb = enc.encode(data["corpus_texts"], batch_size=batch_size)
        logger.info("  Corpus encoded in %.1fs, shape=%s", time.time() - t1, corpus_emb.shape)

        # Encode queries
        logger.info("  Encoding queries (%d texts)...", len(data["query_texts"]))
        t2 = time.time()
        query_emb = enc.encode(data["query_texts"], batch_size=batch_size)
        logger.info("  Queries encoded in %.1fs, shape=%s", time.time() - t2, query_emb.shape)

        # Compute metrics
        logger.info("  Computing retrieval metrics...")
        metrics = compute_retrieval_metrics(
            query_embeddings=query_emb,
            corpus_embeddings=corpus_emb,
            qrels=data["qrels"],
            query_ids=data["query_ids"],
            corpus_ids=data["corpus_ids"],
            k_values=[1, 10],
        )
        all_metrics[ds_name] = metrics
        logger.info(
            "  %s: nDCG@10=%.4f, Recall@10=%.4f, MRR@10=%.4f, MAP@10=%.4f",
            ds_name,
            metrics.get("ndcg@10", 0),
            metrics.get("recall@10", 0),
            metrics.get("mrr@10", 0),
            metrics.get("map@10", 0),
        )

    del enc
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_path,
        "method": method,
        "layer": layer,
        "max_length": max_length,
        "load_in_4bit": load_in_4bit,
        "datasets": all_metrics,
    }
    out_path = output_dir / f"{method}_layer{layer}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", out_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Standalone retrieval evaluation")
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--methods", nargs="+", default=["prompteol", "mean"])
    parser.add_argument("--layers", nargs="+", type=int, default=[-1])
    parser.add_argument("--datasets", nargs="+", default=["QMSum", "2WikiMultihop", "ArguAna"])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="results/standalone")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_results = {}

    for method in args.methods:
        for layer in args.layers:
            # Ensure clean GPU state before each run
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("VRAM free before run: %.2f GB", torch.cuda.memory_allocated() / 1e9)

            run_id = f"{method}_layer{layer}"
            logger.info("=" * 60)
            logger.info("RUN: %s", run_id)
            logger.info("=" * 60)
            try:
                result = run_experiment(
                    model_path=args.model,
                    method=method,
                    layer=layer,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    output_dir=output_dir,
                    datasets=args.datasets,
                    load_in_4bit=not args.no_4bit,
                )
                all_results[run_id] = result["datasets"]
            except Exception as e:
                logger.error("Run %s failed: %s", run_id, e, exc_info=True)
                all_results[run_id] = {"error": str(e)}

    # Save summary
    summary_path = output_dir / "all_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    for run_id, datasets_metrics in all_results.items():
        print(f"\n--- {run_id} ---")
        for ds_name, metrics in datasets_metrics.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                print(
                    f"  {ds_name:20s}  nDCG@10={metrics.get('ndcg@10',0):.4f}  "
                    f"Recall@10={metrics.get('recall@10',0):.4f}  "
                    f"MRR@10={metrics.get('mrr@10',0):.4f}  "
                    f"MAP@10={metrics.get('map@10',0):.4f}"
                )
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
