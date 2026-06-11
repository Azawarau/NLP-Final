#!/usr/bin/env python
"""层数消融：单次编码，多层抽取 — 比逐层重编码高效 32 倍。

重要：datasets 必须在 torch 之前导入（DLL 冲突规避）。
"""

from __future__ import annotations

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
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pooling import last_token_pooling, mean_pooling  # noqa: E402
from src.prompts import build_prompteol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据集加载（与 standalone_eval.py 相同）
# ---------------------------------------------------------------------------

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
    qrels_ds = load_dataset("mteb/arguana", split="test")
    corpus_texts = []
    corpus_ids = []
    for doc in corpus_ds:
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        full_text = f"{title}\n{text}".strip() if title else text
        corpus_texts.append(full_text)
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

        for k in k_values:
            top_k_indices = top_indices[:k]
            y_true = np.zeros(k)
            for rank, idx in enumerate(top_k_indices):
                if corpus_ids[idx] in relevant:
                    y_true[rank] = 1.0
            dcg = sum((2 ** y_true[i] - 1) / np.log2(i + 2) for i in range(len(y_true)))
            ideal = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
            idcg = sum((2 ** ideal[i] - 1) / np.log2(i + 2) for i in range(len(ideal)))
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
# 多层编码器：单次前传，多 layer 抽取
# ---------------------------------------------------------------------------

class MultiLayerEncoder:
    """Encodes texts once and extracts embeddings from specified layers."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        batch_size: int = 1,
        load_in_4bit: bool = True,
    ):
        logger.info("Loading model: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        load_kwargs: dict = {
            "trust_remote_code": True,
            "output_hidden_states": True,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.max_length = max_length
        self.batch_size = batch_size
        logger.info("Model loaded. %d layers, VRAM: %.2f GB", self.num_layers, torch.cuda.memory_allocated() / 1e9)

    def _resolve_layer_idx(self, layer: int) -> int:
        """Map user layer index to hidden_states tuple index."""
        n = self.num_layers
        if layer == -1 or layer == n:
            return n
        if layer < 0:
            return n + layer + 1
        if 1 <= layer <= n:
            return layer
        raise ValueError(f"Invalid layer: {layer}")

    @torch.inference_mode()
    def encode_all_layers(
        self,
        texts: list[str],
        method: str,
        layers: list[int],
    ) -> dict[int, np.ndarray]:
        """
        Encode texts once, extract from all requested layers.

        Returns: {layer: embeddings_ndarray}
        """
        layer_indices = [self._resolve_layer_idx(l) for l in layers]
        all_embeddings: dict[int, list[np.ndarray]] = {l: [] for l in layers}

        for start in tqdm(range(0, len(texts), self.batch_size), desc=f"encode[{method}]"):
            batch = texts[start : start + self.batch_size]

            # Format input
            if method == "prompteol":
                batch_inputs = [build_prompteol(t) for t in batch]
            else:
                batch_inputs = list(batch)

            encoded = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = self.model(**encoded, output_hidden_states=True)

            for layer, layer_idx in zip(layers, layer_indices):
                hidden = outputs.hidden_states[layer_idx]
                if method == "prompteol":
                    pooled = last_token_pooling(hidden, encoded["attention_mask"])
                else:
                    pooled = mean_pooling(hidden, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                all_embeddings[layer].append(pooled.cpu().float().numpy())

        return {l: np.vstack(embs) for l, embs in all_embeddings.items()}

    def cleanup(self):
        del self.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Efficient layer ablation")
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--methods", nargs="+", default=["prompteol", "mean"])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 8, 16, 24, 32])
    parser.add_argument("--datasets", nargs="+", default=["QMSum", "2WikiMultihop", "ArguAna"])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", default="results/layer_ablation")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    encoder = MultiLayerEncoder(
        model_path=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        load_in_4bit=not args.no_4bit,
    )
    logger.info("Model layers available: %d", encoder.num_layers)

    all_results: dict = {}

    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds_name)
        data = DATASET_LOADERS[ds_name]()
        logger.info("corpus=%d, queries=%d", len(data["corpus_texts"]), len(data["query_texts"]))

        ds_results: dict = {}

        for method in args.methods:
            logger.info("--- Method: %s ---", method)
            t0 = time.time()

            # Encode once, extract from all layers
            corpus_embs = encoder.encode_all_layers(data["corpus_texts"], method, args.layers)
            query_embs = encoder.encode_all_layers(data["query_texts"], method, args.layers)
            logger.info("Encoding done in %.1fs", time.time() - t0)

            for layer in args.layers:
                key = f"{method}_layer{layer}"
                logger.info("Computing metrics for %s...", key)
                metrics = compute_retrieval_metrics(
                    query_embeddings=query_embs[layer],
                    corpus_embeddings=corpus_embs[layer],
                    qrels=data["qrels"],
                    query_ids=data["query_ids"],
                    corpus_ids=data["corpus_ids"],
                    k_values=[1, 10],
                )
                ds_results[key] = metrics
                logger.info(
                    "  %s: nDCG@10=%.4f, Recall@10=%.4f, MRR@10=%.4f, MAP@10=%.4f",
                    key,
                    metrics.get("ndcg@10", 0),
                    metrics.get("recall@10", 0),
                    metrics.get("mrr@10", 0),
                    metrics.get("map@10", 0),
                )

        all_results[ds_name] = ds_results
        # Save per-dataset intermediate results
        ds_path = output_dir / f"{ds_name}_results.json"
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(ds_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved: %s", ds_path)

    encoder.cleanup()

    # Save full results
    summary_path = output_dir / "layer_ablation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 80)
    print("LAYER ABLATION SUMMARY")
    print("=" * 80)
    for ds_name, ds_results in all_results.items():
        print(f"\n### {ds_name} ###")
        print(f"{'Method/Layer':<25s} {'nDCG@10':>10s} {'Recall@10':>10s} {'MRR@10':>10s} {'MAP@10':>10s}")
        print("-" * 65)
        for key, metrics in ds_results.items():
            print(
                f"{key:<25s} {metrics.get('ndcg@10',0):>10.4f} "
                f"{metrics.get('recall@10',0):>10.4f} "
                f"{metrics.get('mrr@10',0):>10.4f} "
                f"{metrics.get('map@10',0):>10.4f}"
            )
    print(f"\nFull results: {summary_path}")


if __name__ == "__main__":
    main()
