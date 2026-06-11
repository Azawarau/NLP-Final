#!/usr/bin/env python
"""Fast experiment runner — loads model once, runs all methods/datasets."""
from __future__ import annotations
import sys, json, time, logging, gc
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_encoder import LLMEmbeddingEncoder
from src.pooling import mean_pooling, last_token_pooling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import datasets after torch (DLL order)
from datasets import load_dataset

OUTPUT_DIR = ROOT / "results" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["prompteol", "mean"]
MAX_LENGTH = 1024  # reduced for speed on RTX 4060
BATCH_SIZE = 2
MODEL_PATH = str(ROOT / "models" / "Mistral-7B-Instruct-v0.3")


def load_dataset_safe(name: str):
    """Load dataset with error handling."""
    if name == "QMSum":
        c = load_dataset("dwzhu/LongEmbed", name="qmsum", split="corpus")
        q = load_dataset("dwzhu/LongEmbed", name="qmsum", split="queries")
        qr = load_dataset("dwzhu/LongEmbed", name="qmsum", split="qrels")
        c_texts = [d["text"] for d in c]
        c_ids = [str(d.get("doc_id", d.get("_id", str(i)))) for i, d in enumerate(c)]
        q_texts = [d["text"] for d in q]
        q_ids = [str(d.get("qid", d.get("_id", str(i)))) for i, d in enumerate(q)]
        qrels = {}
        for row in qr:
            qid = str(row.get("qid", row.get("_id", "")))
            did = str(row.get("doc_id", row.get("docid", "")))
            qrels.setdefault(qid, set()).add(did)
        return {"name": "QMSum", "c_texts": c_texts, "c_ids": c_ids,
                "q_texts": q_texts, "q_ids": q_ids, "qrels": qrels}
    elif name == "2WikiMultihop":
        c = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="corpus")
        q = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="queries")
        qr = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="qrels")
        c_texts = [d["text"] for d in c]
        c_ids = [str(d.get("doc_id", d.get("_id", str(i)))) for i, d in enumerate(c)]
        q_texts = [d["text"] for d in q]
        q_ids = [str(d.get("qid", d.get("_id", str(i)))) for i, d in enumerate(q)]
        qrels = {}
        for row in qr:
            qid = str(row.get("qid", row.get("_id", "")))
            did = str(row.get("doc_id", row.get("docid", "")))
            qrels.setdefault(qid, set()).add(did)
        return {"name": "2WikiMultihop", "c_texts": c_texts, "c_ids": c_ids,
                "q_texts": q_texts, "q_ids": q_ids, "qrels": qrels}
    elif name == "ArguAna":
        c = load_dataset("mteb/arguana", "corpus", split="corpus")
        q = load_dataset("mteb/arguana", "queries", split="queries")
        qr = load_dataset("mteb/arguana", split="test")
        c_texts = []
        c_ids = []
        for doc in c:
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            c_texts.append(f"{title}\n{text}".strip() if title else text)
            c_ids.append(str(doc["_id"]))
        q_texts = [d["text"] for d in q]
        q_ids = [str(d["_id"]) for d in q]
        qrels = {}
        for row in qr:
            if float(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
        return {"name": "ArguAna", "c_texts": c_texts, "c_ids": c_ids,
                "q_texts": q_texts, "q_ids": q_ids, "qrels": qrels}


def compute_metrics(q_emb, c_emb, qrels, q_ids, c_ids):
    """Compute retrieval metrics."""
    k_values = [1, 10]
    q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    c_norm = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
    scores = q_norm @ c_norm.T

    cid_to_idx = {cid: i for i, cid in enumerate(c_ids)}
    max_k = max(k_values)
    ndcg_k = {k: [] for k in k_values}
    recall_k = {k: [] for k in k_values}
    mrr_scores = []
    total = 0

    for qi, qid in enumerate(q_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        total += 1
        relevant = qrels[qid]
        top_indices = np.argsort(scores[qi])[::-1][:max_k]

        # MRR
        rr = 0.0
        for rank, idx in enumerate(top_indices):
            if c_ids[idx] in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)

        for k in k_values:
            top_k = top_indices[:k]
            # nDCG
            y_true = np.zeros(k)
            for rank, idx in enumerate(top_k):
                if c_ids[idx] in relevant:
                    y_true[rank] = 1.0
            dcg = sum((2**y_true[i] - 1) / np.log2(i + 2) for i in range(k))
            ideal = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - min(len(relevant), k)), reverse=True)
            idcg = sum((2**ideal[i] - 1) / np.log2(i + 2) for i in range(k))
            ndcg_k[k].append(dcg / idcg if idcg > 0 else 0.0)
            # Recall
            rel_found = sum(1 for idx in top_k if c_ids[idx] in relevant)
            recall_k[k].append(rel_found / min(len(relevant), k) if len(relevant) > 0 else 0)

    metrics = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_k[k])) if ndcg_k[k] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recall_k[k])) if recall_k[k] else 0.0
    metrics["mrr@10"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    metrics["num_queries"] = total
    return metrics


def main():
    datasets = ["QMSum", "2WikiMultihop", "ArguAna"]
    all_results = {}

    for method in METHODS:
        logger.info("=" * 60)
        logger.info("Loading encoder: method=%s layer=-1 max_length=%d", method, MAX_LENGTH)
        t0 = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        enc = LLMEmbeddingEncoder(
            model_name_or_path=MODEL_PATH,
            method=method,
            layer=-1,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            load_in_4bit=True,
            normalize=True,
        )
        logger.info("Encoder loaded in %.1fs. VRAM: %.2f GB",
                    time.time() - t0, torch.cuda.memory_allocated() / 1e9)

        for ds_name in datasets:
            run_id = f"{method}_layer-1"
            logger.info("--- Dataset: %s ---", ds_name)
            data = load_dataset_safe(ds_name)
            logger.info("corpus=%d queries=%d qrels=%d",
                        len(data["c_texts"]), len(data["q_texts"]), len(data["qrels"]))

            # Encode corpus
            t1 = time.time()
            logger.info("Encoding corpus...")
            c_emb = enc.encode(data["c_texts"], batch_size=BATCH_SIZE)
            logger.info("Corpus: %s in %.1fs", c_emb.shape, time.time() - t1)

            # Encode queries
            t2 = time.time()
            logger.info("Encoding queries...")
            q_emb = enc.encode(data["q_texts"], batch_size=BATCH_SIZE)
            logger.info("Queries: %s in %.1fs", q_emb.shape, time.time() - t2)

            # Compute metrics
            t3 = time.time()
            logger.info("Computing metrics...")
            metrics = compute_metrics(q_emb, c_emb, data["qrels"], data["q_ids"], data["c_ids"])
            logger.info("Metrics in %.1fs", time.time() - t3)

            logger.info("  nDCG@10=%.4f  Recall@10=%.4f  MRR@10=%.4f",
                        metrics.get("ndcg@10", 0),
                        metrics.get("recall@10", 0),
                        metrics.get("mrr@10", 0))

            # Save per-dataset result
            result = {
                "model": MODEL_PATH,
                "method": method,
                "layer": -1,
                "max_length": MAX_LENGTH,
                "dataset": ds_name,
                "metrics": metrics,
            }
            out_path = OUTPUT_DIR / f"{method}_layer-1_{ds_name}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Saved: %s", out_path)

            all_results.setdefault(run_id, {})[ds_name] = metrics

        del enc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Encoder freed.")

    # Save summary
    summary = {"config": {"model": MODEL_PATH, "methods": METHODS, "max_length": MAX_LENGTH},
               "results": all_results}
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for run_id, ds_metrics in all_results.items():
        print(f"\n--- {run_id} ---")
        for ds, m in ds_metrics.items():
            print(f"  {ds:20s}  nDCG@10={m.get('ndcg@10',0):.4f}  "
                  f"Recall@10={m.get('recall@10',0):.4f}  MRR@10={m.get('mrr@10',0):.4f}")
    print(f"\nSummary saved to {OUTPUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
