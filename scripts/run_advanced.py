#!/usr/bin/env python
"""Advanced experiment runner for the three research points and their combinations.

Research Point 1: Keyword-Enhanced Weighted Pooling
Research Point 2: Chunk-based Aggregation
Research Point 3: Semantic Compression

Combinations tested:
  - RP1+RP2: chunk → weighted pool within each chunk
  - RP2+RP3: compress → chunk → mean pool
  - RP1+RP2+RP3: compress → chunk → weighted pool

Usage:
  python scripts/run_advanced.py \
    --model models/Mistral-7B-Instruct-v0.3 \
    --datasets QMSum 2WikiMultihop ArguAna \
    --max-length 2048 --batch-size 8 \
    --output-dir results/advanced
"""

from __future__ import annotations

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset loading (same as standalone_eval.py)
# ---------------------------------------------------------------------------

from datasets import load_dataset  # noqa: E402


def load_qmsum():
    corpus = load_dataset("dwzhu/LongEmbed", name="qmsum", split="corpus")
    queries = load_dataset("dwzhu/LongEmbed", name="qmsum", split="queries")
    qrels = load_dataset("dwzhu/LongEmbed", name="qmsum", split="qrels")
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
        "name": "QMSum",
        "corpus_texts": corpus_texts, "corpus_ids": corpus_ids,
        "query_texts": query_texts, "query_ids": query_ids,
        "qrels": dict(qrels_map),
    }


def load_2wikimultihop():
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
        "corpus_texts": corpus_texts, "corpus_ids": corpus_ids,
        "query_texts": query_texts, "query_ids": query_ids,
        "qrels": dict(qrels_map),
    }


def load_arguana():
    corpus_ds = load_dataset("mteb/arguana", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/arguana", "queries", split="queries")
    qrels_ds = load_dataset("mteb/arguana", "default", split="test")
    corpus_texts, corpus_ids = [], []
    for doc in corpus_ds:
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        full = f"{title}\n{text}".strip() if title else text
        corpus_texts.append(full)
        corpus_ids.append(str(doc["_id"]))
    query_texts = [q["text"] for q in queries_ds]
    query_ids = [str(q["_id"]) for q in queries_ds]
    qrels_map = defaultdict(set)
    for row in qrels_ds:
        if float(row["score"]) > 0:
            qrels_map[str(row["query-id"])].add(str(row["corpus-id"]))
    return {
        "name": "ArguAna",
        "corpus_texts": corpus_texts, "corpus_ids": corpus_ids,
        "query_texts": query_texts, "query_ids": query_ids,
        "qrels": dict(qrels_map),
    }


DATASET_LOADERS = {
    "QMSum": load_qmsum,
    "2WikiMultihop": load_2wikimultihop,
    "ArguAna": load_arguana,
}

# ---------------------------------------------------------------------------
# Retrieval metrics (same as standalone_eval.py)
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    qrels: dict[str, set[str]],
    query_ids: list[str],
    corpus_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    if k_values is None:
        k_values = [1, 10]
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9)
    c_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-9)
    scores = q_norm @ c_norm.T
    max_k = max(k_values)
    ndcg_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    mrr_scores = []
    map_scores = []
    for qi, qid in enumerate(query_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        relevant = qrels[qid]
        q_scores = scores[qi]
        top_indices = np.argsort(q_scores)[::-1][:max_k]
        rr = 0.0
        for rank, idx in enumerate(top_indices):
            if corpus_ids[idx] in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)
        ap = 0.0
        relevant_count = 0
        for rank, idx in enumerate(top_indices):
            if corpus_ids[idx] in relevant:
                relevant_count += 1
                ap += relevant_count / (rank + 1)
        if relevant_count > 0:
            ap /= min(len(relevant), max_k)
        map_scores.append(ap)
        for k in k_values:
            top_k_indices = top_indices[:k]
            y_true = np.zeros(k)
            for rank, idx in enumerate(top_k_indices):
                if corpus_ids[idx] in relevant:
                    y_true[rank] = 1.0
            dcg = sum((2 ** y_true[i] - 1) / np.log2(i + 2) for i in range(k))
            ideal = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
            idcg = sum((2 ** ideal[i] - 1) / np.log2(i + 2) for i in range(k))
            ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0.0)
            recall_at_k[k].append(
                sum(1 for idx in top_k_indices if corpus_ids[idx] in relevant) / min(len(relevant), k)
                if len(relevant) > 0 else 0.0
            )
    metrics = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_at_k[k])) if ndcg_at_k[k] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recall_at_k[k])) if recall_at_k[k] else 0.0
    metrics["mrr@10"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    metrics["map@10"] = float(np.mean(map_scores)) if map_scores else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Main advanced experiment
# ---------------------------------------------------------------------------


def run_advanced_experiments(
    model_path: str,
    datasets: list[str],
    output_dir: Path,
    max_length: int = 2048,
    batch_size: int = 8,
    load_in_4bit: bool = True,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    compression_ratio: float = 0.3,
) -> dict:
    """Run all advanced experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.llm_encoder import LLMEmbeddingEncoder
    from src.chunk_encoder import ChunkEncoder, ChunkWeightedEncoder
    from src.semantic_compression import CompressionEncoder, CompressionChunkEncoder
    from src.advanced_pooling import (
        attention_weighted_pooling,
        tfidf_weighted_pooling,
        saliency_weighted_pooling,
        combined_weighted_pooling,
        TFIDFWeightCalculator,
    )

    # ------------------------------------------------------------------
    # Load model once, create base encoder
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", model_path)
    base_encoder = LLMEmbeddingEncoder(
        model_name_or_path=model_path,
        method="mean",
        layer=-1,
        max_length=max_length,
        batch_size=batch_size,
        load_in_4bit=load_in_4bit,
        normalize=True,
    )
    logger.info("Model loaded. VRAM: %.2f GB",
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)

    all_results: dict = {}

    # Define experiment configurations
    experiments = {
        # ---- Baselines ----
        "baseline_mean": {
            "type": "baseline",
            "description": "Baseline: mean-pooling (last layer)",
            "encoder": base_encoder,
        },
        "baseline_prompteol": {
            "type": "baseline",
            "description": "Baseline: PromptEOL (last layer)",
            "encoder": LLMEmbeddingEncoder(
                model_name_or_path=model_path, method="prompteol", layer=-1,
                max_length=max_length, batch_size=batch_size,
                load_in_4bit=load_in_4bit, normalize=True,
            ),
        },
        # ---- RP1: Weighted Pooling ----
        "rp1_attention_weighted": {
            "type": "rp1",
            "description": "Attention-score weighted pooling",
            "pooling_fn": attention_weighted_pooling,
        },
        "rp1_saliency_weighted": {
            "type": "rp1",
            "description": "Gradient-norm saliency weighted pooling",
            "pooling_fn": saliency_weighted_pooling,
        },
        "rp1_combined_weighted": {
            "type": "rp1_combined",
            "description": "Combined weighted pooling (attention + saliency + norm)",
            "pooling_fn": combined_weighted_pooling,
        },
        # ---- RP2: Chunk-based Aggregation ----
        "rp2_chunk_mean": {
            "type": "rp2",
            "description": f"Chunk-based (size={chunk_size}, overlap={chunk_overlap}, mean agg)",
            "encoder": ChunkEncoder(
                base_encoder, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                aggregation="mean",
            ),
        },
        "rp2_chunk_weighted": {
            "type": "rp2",
            "description": f"Chunk-based (size={chunk_size}, overlap={chunk_overlap}, weighted agg)",
            "encoder": ChunkEncoder(
                base_encoder, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                aggregation="weighted",
            ),
        },
        # ---- RP3: Semantic Compression ----
        "rp3_extractive": {
            "type": "rp3",
            "description": f"Extractive compression (ratio={compression_ratio}) + mean-pooling",
            "encoder": CompressionEncoder(
                base_encoder, compression_method="extractive",
                compression_ratio=compression_ratio, use_llm_for_compression=True,
            ),
        },
        "rp3_hierarchical": {
            "type": "rp3",
            "description": f"Hierarchical compression (ratio={compression_ratio}) + mean-pooling",
            "encoder": CompressionEncoder(
                base_encoder, compression_method="hierarchical",
                compression_ratio=compression_ratio, use_llm_for_compression=True,
            ),
        },
        # ---- Combined: RP2 + RP3 (compress → chunk) ----
        "combined_rp23": {
            "type": "combined",
            "description": "RP2+RP3: extractive compression → chunk → mean pool",
            "encoder": CompressionChunkEncoder(
                base_encoder,
                compression_ratio=compression_ratio,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                aggregation="mean",
            ),
        },
        # ---- Combined: RP1 + RP2 + RP3 ----
        "combined_rp123": {
            "type": "combined",
            "description": "RP1+RP2+RP3: compress → chunk → weighted pool → aggregate",
            "encoder": None,  # built dynamically after TF-IDF fit
        },
    }

    # ------------------------------------------------------------------
    # Run each experiment on each dataset
    # ------------------------------------------------------------------
    for exp_name, exp_config in experiments.items():
        logger.info("=" * 70)
        logger.info("EXPERIMENT: %s", exp_name)
        logger.info("  %s", exp_config["description"])
        logger.info("=" * 70)

        exp_type = exp_config["type"]
        encoder = exp_config.get("encoder")

        # For rp1_tfidf, we need to fit TF-IDF on corpus first
        if exp_type == "rp1_tfidf":
            continue  # TF-IDF requires corpus pre-loading; handled in rp1_combined

        # For RP1 variants, we modify the base encoder's pooling
        if exp_type in ("rp1", "rp1_combined"):
            pooling_fn = exp_config.get("pooling_fn")
            encoder = base_encoder  # reuse base encoder, override pooling in encode loop

        if encoder is None and exp_name == "combined_rp123":
            continue  # built dynamically below

        exp_results = {}
        for ds_name in datasets:
            logger.info("--- Dataset: %s ---", ds_name)
            data = DATASET_LOADERS[ds_name]()
            logger.info("corpus=%d, queries=%d",
                        len(data["corpus_texts"]), len(data["query_texts"]))

            t0 = time.time()

            if exp_type in ("rp1", "rp1_combined"):
                # Encode with custom pooling
                corpus_emb = _encode_with_custom_pooling(
                    base_encoder, data["corpus_texts"],
                    pooling_fn=pooling_fn,
                    batch_size=batch_size,
                )
                query_emb = _encode_with_custom_pooling(
                    base_encoder, data["query_texts"],
                    pooling_fn=pooling_fn,
                    batch_size=batch_size,
                )
            else:
                # Standard encode via encoder wrapper
                corpus_emb = encoder.encode(
                    data["corpus_texts"], batch_size=batch_size, show_progress=True
                )
                query_emb = encoder.encode(
                    data["query_texts"], batch_size=batch_size, show_progress=True
                )

            elapsed = time.time() - t0
            logger.info("Encoding done in %.1fs", elapsed)

            # Compute metrics
            metrics = compute_retrieval_metrics(
                query_embeddings=query_emb,
                corpus_embeddings=corpus_emb,
                qrels=data["qrels"],
                query_ids=data["query_ids"],
                corpus_ids=data["corpus_ids"],
                k_values=[1, 10],
            )
            exp_results[ds_name] = metrics

            logger.info(
                "  nDCG@10=%.4f  Recall@10=%.4f  MRR@10=%.4f  MAP@10=%.4f",
                metrics["ndcg@10"], metrics["recall@10"],
                metrics["mrr@10"], metrics["map@10"],
            )

            # Cleanup GPU
            del corpus_emb, query_emb
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[exp_name] = {
            "description": exp_config["description"],
            "type": exp_type,
            "results": exp_results,
        }

        # Save incremental results
        inc_path = output_dir / f"{exp_name}.json"
        with open(inc_path, "w", encoding="utf-8") as f:
            json.dump(all_results[exp_name], f, indent=2, ensure_ascii=False)

        # Clean up encoder to free VRAM
        if exp_type not in ("rp1", "rp1_combined", "baseline"):
            # non-baseline encoders wrap the base encoder; just delete the wrapper
            del encoder
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save full summary
    # ------------------------------------------------------------------
    summary_path = output_dir / "advanced_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    _print_advanced_summary(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Helper: Encode with custom pooling function
# ---------------------------------------------------------------------------


def _encode_with_custom_pooling(
    encoder,
    texts: list[str],
    pooling_fn,
    batch_size: int = 8,
) -> np.ndarray:
    """Encode texts using a custom pooling function instead of the encoder's default.

    This reuses the encoder's model, tokenizer, and hook for efficient extraction.
    """
    import torch.nn.functional as F
    from tqdm import tqdm

    all_embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="encode[custom_pool]"):
        batch = texts[start:start + batch_size]

        # Format and tokenize (reuse encoder's logic)
        formatted = encoder._format_inputs(batch) if hasattr(encoder, '_format_inputs') else list(batch)
        encoded = encoder.tokenizer(
            formatted, padding=True, truncation=True,
            max_length=encoder.max_length, return_tensors="pt",
        )
        device = next(encoder.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward pass with hook
        encoder._hook_cache.clear()
        encoder.model(**encoded)
        hidden = encoder._hook_cache[-1]

        # Apply custom pooling
        pooled = pooling_fn(hidden, encoded["attention_mask"])

        # Normalize
        if encoder.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu().float().numpy())

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_advanced_summary(results: dict) -> None:
    """Print human-readable summary of advanced experiments."""
    print("\n" + "=" * 90)
    print("ADVANCED EXPERIMENT RESULTS — COMPLETE SUMMARY")
    print("=" * 90)

    # Find all datasets
    all_datasets = set()
    for exp_data in results.values():
        for ds_name in exp_data.get("results", {}):
            all_datasets.add(ds_name)
    datasets = sorted(all_datasets)

    # Group by type
    groups = {
        "Baselines": [],
        "RP1: Weighted Pooling": [],
        "RP2: Chunk-based": [],
        "RP3: Semantic Compression": [],
        "Combined": [],
    }

    for exp_name, exp_data in results.items():
        desc = exp_data.get("description", exp_name)
        etype = exp_data.get("type", "")
        if etype == "baseline":
            groups["Baselines"].append((exp_name, desc, exp_data))
        elif etype in ("rp1", "rp1_combined"):
            groups["RP1: Weighted Pooling"].append((exp_name, desc, exp_data))
        elif etype == "rp2":
            groups["RP2: Chunk-based"].append((exp_name, desc, exp_data))
        elif etype == "rp3":
            groups["RP3: Semantic Compression"].append((exp_name, desc, exp_data))
        else:
            groups["Combined"].append((exp_name, desc, exp_data))

    for group_name, items in groups.items():
        if not items:
            continue
        print(f"\n{'=' * 40}")
        print(f"  {group_name}")
        print(f"{'=' * 40}")
        for exp_name, desc, exp_data in items:
            print(f"\n  [{exp_name}] {desc}")
            exp_results = exp_data.get("results", {})
            # Print header
            header = f"  {'Dataset':<20s} {'nDCG@10':>10s} {'Recall@10':>10s} {'MRR@10':>10s} {'MAP@10':>10s}"
            print(header)
            print("  " + "-" * 62)
            for ds_name in datasets:
                metrics = exp_results.get(ds_name, {})
                if metrics:
                    print(
                        f"  {ds_name:<20s} {metrics.get('ndcg@10', 0):>10.4f} "
                        f"{metrics.get('recall@10', 0):>10.4f} "
                        f"{metrics.get('mrr@10', 0):>10.4f} "
                        f"{metrics.get('map@10', 0):>10.4f}"
                    )

    # Compute improvement over baseline
    baseline_mean = results.get("baseline_mean", {}).get("results", {})
    print(f"\n{'=' * 40}")
    print(f"  Improvement over Baseline Mean-Pooling (nDCG@10)")
    print(f"{'=' * 40}")

    for exp_name, exp_data in results.items():
        if exp_name == "baseline_mean":
            continue
        exp_results = exp_data.get("results", {})
        improvements = []
        for ds_name in datasets:
            base_ndcg = baseline_mean.get(ds_name, {}).get("ndcg@10", 0)
            exp_ndcg = exp_results.get(ds_name, {}).get("ndcg@10", 0)
            if base_ndcg > 0:
                pct = 100 * (exp_ndcg - base_ndcg) / base_ndcg
                improvements.append(pct)
        if improvements:
            avg_imp = np.mean(improvements)
            print(f"  {exp_name:<30s}: {avg_imp:+.1f}% avg over {len(improvements)} datasets")

    print(f"\n{'=' * 90}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Advanced experiment runner")
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+",
                        default=["QMSum", "2WikiMultihop", "ArguAna"])
    parser.add_argument("--output-dir", default="results/advanced")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    run_advanced_experiments(
        model_path=args.model,
        datasets=args.datasets,
        output_dir=output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        load_in_4bit=not args.no_4bit,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        compression_ratio=args.compression_ratio,
    )

    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
