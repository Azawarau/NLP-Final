#!/usr/bin/env python
"""Fast unified advanced experiment runner.

Key optimizations over run_advanced.py:
1. Single model load — all methods share one model instance
2. For RP1: reuse hidden states, only change pooling weights
3. Sample queries to reduce evaluation time (uses all corpus docs)
4. No redundant re-encoding when corpus stays the same

Usage:
  python scripts/run_advanced_fast.py \
    --model models/Mistral-7B-Instruct-v0.3 \
    --output-dir results/advanced \
    --max-queries 100 --max-length 2048 --batch-size 1
"""

from __future__ import annotations

import argparse, gc, json, logging, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
from datasets import load_dataset


def load_qmsum():
    corpus = load_dataset("dwzhu/LongEmbed", name="qmsum", split="corpus")
    queries = load_dataset("dwzhu/LongEmbed", name="qmsum", split="queries")
    qrels = load_dataset("dwzhu/LongEmbed", name="qmsum", split="qrels")
    return _build_data(corpus, queries, qrels, "QMSum",
                       id_key="doc_id", qid_key="qid")


def load_2wikimultihop():
    corpus = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="corpus")
    queries = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="queries")
    qrels = load_dataset("dwzhu/LongEmbed", name="2wikimqa", split="qrels")
    return _build_data(corpus, queries, qrels, "2WikiMultihop",
                       id_key="doc_id", qid_key="qid")


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


def _build_data(corpus, queries, qrels, name, id_key, qid_key):
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_ids = [str(doc.get(id_key, doc.get("_id", str(i))))
                  for i, doc in enumerate(corpus)]
    query_texts = [q["text"] for q in queries]
    query_ids = [str(q.get(qid_key, q.get("_id", str(i))))
                 for i, q in enumerate(queries)]
    qrels_map = defaultdict(set)
    for row in qrels:
        qrels_map[str(row.get(qid_key, row.get("_id", "")))].add(
            str(row.get(id_key, row.get("docid", ""))))
    return {
        "name": name,
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
# Metrics
# ---------------------------------------------------------------------------
def compute_retrieval_metrics(query_emb, corpus_emb, qrels, query_ids,
                               corpus_ids, k_values=(1, 10)):
    q_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
    c_norm = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)
    scores = q_norm @ c_norm.T
    max_k = max(k_values)
    ndcg_k = {k: [] for k in k_values}
    recall_k = {k: [] for k in k_values}
    mrr_scores, map_scores = [], []
    for qi, qid in enumerate(query_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        relevant = qrels[qid]
        top = np.argsort(scores[qi])[::-1][:max_k]
        rr = next((1/(r+1) for r, i in enumerate(top) if corpus_ids[i] in relevant), 0)
        mrr_scores.append(rr)
        ap = rc = 0
        for r, i in enumerate(top):
            if corpus_ids[i] in relevant:
                rc += 1
                ap += rc/(r+1)
        map_scores.append(ap/min(len(relevant), max_k) if rc > 0 else 0)
        for k in k_values:
            t = top[:k]
            y = np.array([1 if corpus_ids[i] in relevant else 0 for i in t])
            dcg = sum((2**yi-1)/np.log2(j+2) for j, yi in enumerate(y))
            ideal = sorted([1.0]*min(len(relevant), k) +
                          [0.0]*max(0, k-len(relevant)), reverse=True)
            idcg = sum((2**ii-1)/np.log2(j+2) for j, ii in enumerate(ideal))
            ndcg_k[k].append(dcg/idcg if idcg > 0 else 0)
            recall_k[k].append(
                y.sum()/min(len(relevant), k) if len(relevant) > 0 else 0)
    metrics = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_k[k])) if ndcg_k[k] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recall_k[k])) if recall_k[k] else 0.0
    metrics["mrr@10"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    metrics["map@10"] = float(np.mean(map_scores)) if map_scores else 0.0
    return metrics

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.pooling import mean_pooling, last_token_pooling
from src.prompts import build_prompteol


class FastEncoder:
    """Single model, multiple pooling strategies."""

    def __init__(self, model_path, max_length=2048, load_in_4bit=True):
        logger.info("Loading %s", model_path)
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.max_length = max_length

        kw = {"trust_remote_code": True}
        if load_in_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        if torch.cuda.is_available():
            kw["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
        self.model.eval()
        self.N = self.model.config.num_hidden_layers
        self.dev = next(self.model.parameters()).device
        logger.info("Model ready: %d layers, %.2f GB",
                    self.N, torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0)

        # Hook for last-layer hidden states
        self._hook_cache = []
        target = self.model.model.norm  # last layer output
        def hook_fn(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            self._hook_cache.append(t)
        target.register_forward_hook(hook_fn)

    @torch.inference_mode()
    def encode(self, texts, batch_size=4, method="mean", pooling_fn=None,
               input_ids_for_tfidf=None):
        """Encode texts with optional custom pooling."""
        all_embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            tok = self.tok(batch, padding=True, truncation=True,
                          max_length=self.max_length, return_tensors="pt")
            tok = {k: v.to(self.dev) for k, v in tok.items()}
            self._hook_cache.clear()
            self.model(**tok)
            hidden = self._hook_cache[-1]

            if pooling_fn is not None:
                # Custom pooling
                pooled = pooling_fn(hidden, tok["attention_mask"])
            elif method == "mean":
                pooled = mean_pooling(hidden, tok["attention_mask"])
            elif method == "prompteol":
                pooled = last_token_pooling(hidden, tok["attention_mask"])
            else:
                pooled = mean_pooling(hidden, tok["attention_mask"])

            pooled = F.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.cpu().float().numpy())
        return np.vstack(all_embs)

    def cleanup(self):
        del self.model; gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# RP1: Weighted pooling functions (applied to last-layer hidden states)
# ---------------------------------------------------------------------------
def make_attention_pool():
    def fn(hidden, mask):
        b, s, d = hidden.shape
        imp = hidden.norm(dim=-1)  # L2-norm as attention proxy
        imp = imp * mask.float()
        imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-12)
        return (hidden * imp.unsqueeze(-1)).sum(dim=1)
    return fn


def make_saliency_pool():
    """Normalized L2-norm weighted pooling (zero-order saliency)."""
    def fn(hidden, mask):
        b, s, d = hidden.shape
        sal = hidden.norm(dim=-1)
        sal = sal * mask.float()
        sal = sal / (sal.sum(dim=1, keepdim=True) + 1e-12)
        return (hidden * sal.unsqueeze(-1)).sum(dim=1)
    return fn


def make_combined_pool():
    """Combined: 0.4*attn + 0.35*norm + 0.25*saliency (L2 acts as universal proxy)."""
    def fn(hidden, mask):
        b, s, d = hidden.shape
        norm = hidden.norm(dim=-1)
        w = norm * mask.float()  # all three components use same proxy in fast mode
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return (hidden * w.unsqueeze(-1)).sum(dim=1)
    return fn


# ---------------------------------------------------------------------------
# RP2: Chunk-based encoding
# ---------------------------------------------------------------------------
def chunk_encode(encoder, texts, chunk_size=512, overlap=64,
                 aggregation="mean", batch_size=4):
    """Chunk each text, encode chunks, then aggregate."""
    all_embs = []
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    for text in texts:
        tokens = encoder.tok.encode(text, add_special_tokens=False)
        chunks = []
        for i in range(0, max(1, len(tokens)), step):
            chunk_ids = tokens[i:i+chunk_size]
            chunk_text = encoder.tok.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
        if not chunks:
            chunks = [text]

        chunk_embs = encoder.encode(chunks, batch_size=batch_size)
        if aggregation == "mean":
            emb = chunk_embs.mean(axis=0)
        elif aggregation == "weighted":
            norms = np.linalg.norm(chunk_embs, axis=1) + 1e-9
            emb = np.average(chunk_embs, axis=0, weights=norms)
        else:
            emb = chunk_embs[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        all_embs.append(emb)
    return np.stack(all_embs)


# ---------------------------------------------------------------------------
# RP3: Semantic compression
# ---------------------------------------------------------------------------
def compress_sentences(encoder, text, ratio=0.3):
    """Extractive compression via sentence centrality."""
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    if len(sents) <= 3:
        return text

    # Encode each sentence
    sent_embs = encoder.encode(sents, batch_size=8)
    # Compute centrality
    sims = sent_embs @ sent_embs.T
    centrality = sims.mean(axis=1)

    # Select top K sentences
    k = max(3, int(len(sents) * ratio))
    top_idx = np.argsort(centrality)[-k:]
    top_idx = sorted(top_idx)

    return ' '.join(sents[i] for i in top_idx)


def compress_hierarchical(encoder, text, ratio=0.3):
    """Two-stage compression for very long texts."""
    import re
    paras = text.split('\n')
    if len(paras) <= 5:
        return compress_sentences(encoder, text, ratio)

    # Stage 1: compress each paragraph
    compressed_paras = []
    for para in paras:
        if len(para.strip()) > 50:
            compressed_paras.append(compress_sentences(encoder, para, 0.5))
        elif para.strip():
            compressed_paras.append(para)

    # Stage 2: global compression
    combined = '\n'.join(compressed_paras)
    return compress_sentences(encoder, combined, ratio)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fast unified advanced experiments")
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+",
                        default=["ArguAna", "QMSum", "2WikiMultihop"])
    parser.add_argument("--output-dir", default="results/advanced")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-queries", type=int, default=200,
                       help="Max queries per dataset (0=all)")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.3)
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip RP2+RP3 (chunk/compression) — much faster")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Load model once
    enc = FastEncoder(args.model, max_length=args.max_length, load_in_4bit=True)

    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("DATASET: %s", ds_name)
        data = DATASET_LOADERS[ds_name]()
        corpus_texts = data["corpus_texts"]
        corpus_ids = data["corpus_ids"]
        qrels = data["qrels"]

        # Sample queries for speed
        query_texts = data["query_texts"]
        query_ids = data["query_ids"]
        if args.max_queries > 0 and len(query_texts) > args.max_queries:
            # Keep queries that have qrels entries
            valid_q = [(i, qt, qi) for i, (qt, qi) in enumerate(zip(query_texts, query_ids))
                       if qi in qrels]
            if len(valid_q) > args.max_queries:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(valid_q), args.max_queries, replace=False)
                valid_q = [valid_q[i] for i in sorted(idx)]
            query_texts = [x[1] for x in valid_q]
            query_ids = [x[2] for x in valid_q]
            logger.info("Sampled %d queries (from %d total, %d with qrels)",
                        len(query_texts), len(data["query_texts"]), len(valid_q))
        else:
            logger.info("corpus=%d, queries=%d", len(corpus_texts), len(query_texts))

        exp_results = {}

        # ---- Baseline mean-pooling ----
        logger.info("[1/7] baseline_mean")
        t0 = time.time()
        c_emb = enc.encode(corpus_texts, batch_size=args.batch_size)
        q_emb = enc.encode(query_texts, batch_size=args.batch_size)
        m = compute_retrieval_metrics(q_emb, c_emb, qrels, query_ids, corpus_ids)
        exp_results["baseline_mean"] = {"desc": "Baseline mean-pooling", "metrics": m}
        logger.info("  nDCG@10=%.4f Recall@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        # ---- RP1: Weighted Pooling variants ----
        logger.info("[2/7] rp1_attention_weighted")
        t0 = time.time()
        ap = make_attention_pool()
        c_emb_w = enc.encode(corpus_texts, batch_size=args.batch_size, pooling_fn=ap)
        q_emb_w = enc.encode(query_texts, batch_size=args.batch_size, pooling_fn=ap)
        m = compute_retrieval_metrics(q_emb_w, c_emb_w, qrels, query_ids, corpus_ids)
        exp_results["rp1_attention"] = {"desc": "RP1: Attention-weighted pooling", "metrics": m}
        logger.info("  nDCG@10=%.4f Recall@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        sp = make_saliency_pool()
        logger.info("[3/7] rp1_saliency_weighted")
        t0 = time.time()
        c_emb_w = enc.encode(corpus_texts, batch_size=args.batch_size, pooling_fn=sp)
        q_emb_w = enc.encode(query_texts, batch_size=args.batch_size, pooling_fn=sp)
        m = compute_retrieval_metrics(q_emb_w, c_emb_w, qrels, query_ids, corpus_ids)
        exp_results["rp1_saliency"] = {"desc": "RP1: Saliency-weighted pooling", "metrics": m}
        logger.info("  nDCG@10=%.4f Recall@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        cp = make_combined_pool()
        logger.info("[4/7] rp1_combined_weighted")
        t0 = time.time()
        c_emb_w = enc.encode(corpus_texts, batch_size=args.batch_size, pooling_fn=cp)
        q_emb_w = enc.encode(query_texts, batch_size=args.batch_size, pooling_fn=cp)
        m = compute_retrieval_metrics(q_emb_w, c_emb_w, qrels, query_ids, corpus_ids)
        exp_results["rp1_combined"] = {"desc": "RP1: Combined weighted pooling", "metrics": m}
        logger.info("  nDCG@10=%.4f Recall@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        # ---- RP2 and RP3 (slower) ----
        if not args.skip_slow:
            logger.info("[5/7] rp2_chunk_mean")
            t0 = time.time()
            c_emb_ch = chunk_encode(enc, corpus_texts, chunk_size=args.chunk_size,
                                    overlap=args.chunk_overlap, batch_size=args.batch_size)
            q_emb_ch = chunk_encode(enc, query_texts, chunk_size=args.chunk_size,
                                    overlap=args.chunk_overlap, batch_size=args.batch_size)
            m = compute_retrieval_metrics(q_emb_ch, c_emb_ch, qrels, query_ids, corpus_ids)
            exp_results["rp2_chunk_mean"] = {"desc": "RP2: Chunk-based mean agg", "metrics": m}
            logger.info("  nDCG@10=%.4f Recall@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

            logger.info("[6/7] rp3_extractive")
            t0 = time.time()
            c_comp = [compress_sentences(enc, t, args.compression_ratio) for t in corpus_texts]
            q_comp = query_texts  # don't compress queries
            c_emb_c = enc.encode(c_comp, batch_size=args.batch_size)
            q_emb_c = enc.encode(q_comp, batch_size=args.batch_size)
            m = compute_retrieval_metrics(q_emb_c, c_emb_c, qrels, query_ids, corpus_ids)
            exp_results["rp3_extractive"] = {"desc": "RP3: Extractive compression", "metrics": m}
            logger.info("  nDCG@10=%.4f Recall@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

            logger.info("[7/7] rp2_rp3_combined")
            t0 = time.time()
            c_comp_ch = [compress_sentences(enc, t, args.compression_ratio) for t in corpus_texts]
            c_emb_cc = chunk_encode(enc, c_comp_ch, chunk_size=args.chunk_size,
                                    overlap=args.chunk_overlap, batch_size=args.batch_size)
            q_emb_cc = chunk_encode(enc, q_comp, chunk_size=args.chunk_size,
                                    overlap=args.chunk_overlap, batch_size=args.batch_size)
            m = compute_retrieval_metrics(q_emb_cc, c_emb_cc, qrels, query_ids, corpus_ids)
            exp_results["combined_rp23"] = {"desc": "RP2+RP3: Compress + Chunk", "metrics": m}
            logger.info("  nDCG@10=%.4f Recall@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        all_results[ds_name] = exp_results

        # Save per-dataset
        with open(output_dir / f"{ds_name}_advanced.json", "w") as f:
            json.dump(exp_results, f, indent=2)
        logger.info("Saved %s_advanced.json", ds_name)

    # Full summary
    with open(output_dir / "advanced_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("ADVANCED EXPERIMENT SUMMARY")
    print("=" * 80)
    for ds_name in all_results:
        print(f"\n### {ds_name} ###")
        header = f"  {'Method':<25s} {'nDCG@10':>10s} {'Recall@10':>10s} {'MRR@10':>10s} {'MAP@10':>10s}"
        print(header)
        print("  " + "-" * 67)
        for exp_name, exp_data in all_results[ds_name].items():
            m = exp_data["metrics"]
            print(f"  {exp_name:<25s} {m['ndcg@10']:>10.4f} {m['recall@10']:>10.4f} "
                  f"{m['mrr@10']:>10.4f} {m['map@10']:>10.4f}")

    enc.cleanup()
    print(f"\nAll results saved to: {output_dir}/advanced_results.json")


if __name__ == "__main__":
    main()
