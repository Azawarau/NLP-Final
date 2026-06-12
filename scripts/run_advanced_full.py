#!/usr/bin/env python
"""Advanced experiments: RP1 + RP2 + RP3 + combinations. Sampled for speed.

RP1: Weighted pooling (L2-norm, absmax, combined)
RP2: Chunk-based aggregation (mean and weighted)
RP3: Extractive / hierarchical compression
Combined: RP2+RP3 (compress → chunk → mean pool)
"""

from __future__ import annotations
import argparse, gc, json, logging, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.pooling import mean_pooling


# ======================== Data loading ========================
def load_data(ds_name, max_corpus=200, max_queries=50):
    if ds_name == "ArguAna":
        c = load_dataset("mteb/arguana", "corpus", split="corpus")
        q = load_dataset("mteb/arguana", "queries", split="queries")
        r = load_dataset("mteb/arguana", "default", split="test")
        ct = []
        for doc in c:
            title = doc.get('title', '') or ''
            txt = doc.get('text', '') or ''
            full = f"{title}\n{txt}".strip() if title else txt
            ct.append(full)
        ci = [str(doc["_id"]) for doc in c]
        qt = [doc["text"] for doc in q]
        qi = [str(doc["_id"]) for doc in q]
    else:
        key = {"QMSum": "qmsum", "2WikiMultihop": "2wikimqa"}[ds_name]
        c = load_dataset("dwzhu/LongEmbed", name=key, split="corpus")
        q = load_dataset("dwzhu/LongEmbed", name=key, split="queries")
        r = load_dataset("dwzhu/LongEmbed", name=key, split="qrels")
        ct = [doc["text"] for doc in c]
        ci = [str(doc.get("doc_id", doc.get("_id", str(i)))) for i, doc in enumerate(c)]
        qt = [doc["text"] for doc in q]
        qi = [str(doc.get("qid", doc.get("_id", str(i)))) for i, doc in enumerate(q)]

    qrels = defaultdict(set)
    for row in r:
        if ds_name == "ArguAna":
            if float(row["score"]) > 0:
                qrels[str(row["query-id"])].add(str(row["corpus-id"]))
        else:
            qrels[str(row.get("qid", row.get("_id", "")))].add(
                str(row.get("doc_id", row.get("docid", ""))))

    rng = np.random.RandomState(42)
    valid = [(i, qt[i], qi[i]) for i in range(len(qt)) if qi[i] in qrels]
    if len(valid) > max_queries:
        idx = sorted(rng.choice(len(valid), max_queries, replace=False))
        valid = [valid[i] for i in idx]
    qt_use = [v[1] for v in valid]
    qi_use = [v[2] for v in valid]

    keep_doc_ids = set()
    for qid in qi_use:
        keep_doc_ids.update(qrels.get(qid, set()))
    other_docs = [(i, ct[i], ci[i]) for i in range(len(ct)) if ci[i] not in keep_doc_ids]
    needed = max(0, max_corpus - len(keep_doc_ids))
    if needed > 0 and other_docs:
        extra = sorted(rng.choice(len(other_docs), min(needed, len(other_docs)),
                                  replace=False))
        for e in extra:
            keep_doc_ids.add(other_docs[e][2])

    keep_idx = [i for i, cid in enumerate(ci) if cid in keep_doc_ids]
    ct_use = [ct[i] for i in keep_idx]
    ci_use = [ci[i] for i in keep_idx]

    qrels_use = {}
    for qid in qi_use:
        rel = qrels.get(qid, set()) & keep_doc_ids
        if rel:
            qrels_use[qid] = rel

    logger.info("Data: %d corpus, %d queries, %d qrels pairs",
                len(ct_use), len(qt_use), sum(len(v) for v in qrels_use.values()))
    return ct_use, ci_use, qt_use, qi_use, qrels_use


# ======================== Metrics ========================
def calc_metrics(q_emb, c_emb, qrels, q_ids, c_ids):
    q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    c = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
    scores = q @ c.T
    ndcg10, recall10, mrr, map_ = [], [], [], []
    for qi, qid in enumerate(q_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        rel = qrels[qid]
        top = np.argsort(scores[qi])[::-1][:10]
        rr = next((1/(r+1) for r, i in enumerate(top) if c_ids[i] in rel), 0)
        mrr.append(rr)
        rc = 0; ap = 0
        for r, i in enumerate(top):
            if c_ids[i] in rel:
                rc += 1; ap += rc/(r+1)
        map_.append(ap/min(len(rel), 10) if rc > 0 else 0)
        y = np.array([1 if c_ids[i] in rel else 0 for i in top])
        dcg = sum((2**yi-1)/np.log2(j+2) for j, yi in enumerate(y))
        ideal = sorted([1.0]*min(len(rel), 10) + [0.0]*max(0, 10-len(rel)),
                       reverse=True)
        idcg = sum((2**ii-1)/np.log2(j+2) for j, ii in enumerate(ideal))
        ndcg10.append(dcg/idcg if idcg > 0 else 0)
        recall10.append(y.sum()/min(len(rel), 10) if len(rel) > 0 else 0)
    return {
        "ndcg@10": float(np.mean(ndcg10)) if ndcg10 else 0.0,
        "recall@10": float(np.mean(recall10)) if recall10 else 0.0,
        "mrr@10": float(np.mean(mrr)) if mrr else 0.0,
        "map@10": float(np.mean(map_)) if map_ else 0.0,
    }


# ======================== RP1 pooling ========================
def l2_wp(hidden, mask):
    w = hidden.norm(dim=-1) * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return (hidden * w.unsqueeze(-1)).sum(dim=1)

def combined_wp(hidden, mask):
    w = hidden.norm(dim=-1) * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return (hidden * w.unsqueeze(-1)).sum(dim=1)


# ======================== RP2: Chunk encoding ========================
def chunk_encode(encode_fn, texts, chunk_size=512, overlap=64,
                 aggregation="mean"):
    """Split each text into overlapping chunks, encode, then aggregate."""
    step = chunk_size - overlap if chunk_size > overlap else chunk_size
    all_embs = []
    for text in texts:
        # Tokenize to get accurate chunk boundaries
        chunks = _split_into_chunks(text, chunk_size, step)
        if not chunks:
            chunks = [text]
        # Encode all chunks at once if they fit, otherwise one per batch
        chunk_embs = encode_fn(chunks)
        if aggregation == "mean":
            emb = chunk_embs.mean(axis=0)
        else:  # weighted by L2-norm
            norms = np.linalg.norm(chunk_embs, axis=1) + 1e-9
            emb = np.average(chunk_embs, axis=0, weights=norms)
        # Re-normalize
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        all_embs.append(emb)
    return np.stack(all_embs)


def _split_into_chunks(text, chunk_size, step):
    """Split text into chunks of approximately chunk_size tokens."""
    # Approximate: ~4 chars per token for English text
    words = text.split()
    chunks = []
    i = 0
    # Approx token count: assume 1 token ≈ 0.75 words in English
    words_per_chunk = int(chunk_size * 0.75)
    words_step = int(step * 0.75)
    if words_step <= 0:
        words_step = max(1, words_per_chunk // 2)
    while i < len(words):
        chunk_words = words[i:i + words_per_chunk]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        i += words_step
    return chunks


# ======================== RP3: Semantic compression ========================
def compress_extractive(encode_fn, text, ratio=0.3, min_sents=3):
    """Extractive compression via sentence centrality."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    if len(sents) <= min_sents:
        return text

    # Encode each sentence
    sent_embs = encode_fn(sents)

    # Pairwise cosine similarity → centrality = mean similarity to others
    sim_matrix = sent_embs @ sent_embs.T
    centrality = sim_matrix.mean(axis=1)

    k = max(min_sents, int(len(sents) * ratio))
    top_idx = np.argsort(centrality)[-k:]
    top_idx = sorted(top_idx)

    return ' '.join(sents[i] for i in top_idx)


def compress_hierarchical(encode_fn, text, ratio=0.3):
    """Two-stage: per-paragraph → global compression."""
    # Split by newlines as rough paragraph boundaries
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paras) <= 3:
        return compress_extractive(encode_fn, text, ratio)

    # Stage 1: per-paragraph compression at 50%
    compressed = []
    for para in paras:
        word_count = len(para.split())
        if word_count > 30:
            compressed.append(compress_extractive(encode_fn, para, 0.5))
        else:
            compressed.append(para)

    # Stage 2: global compression
    combined = ' '.join(compressed)
    return compress_extractive(encode_fn, combined, ratio)


# ======================== Main ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+",
                        default=["ArguAna", "QMSum", "2WikiMultihop"])
    parser.add_argument("--output-dir", default="results/advanced")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-corpus", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--compression-ratio", type=float, default=0.3)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    # ---- Load model ----
    logger.info("Loading model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True}
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    dev = next(model.parameters()).device

    hook_cache = []
    target = model.model.norm
    def hook_fn(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        hook_cache.append(t)
    target.register_forward_hook(hook_fn)

    @torch.inference_mode()
    def encode(texts, pool_fn=None):
        """Encode a batch of texts into normalized embeddings."""
        embs = []
        for start in range(0, len(texts), args.batch_size):
            b = texts[start:start + args.batch_size]
            t = tok(b, padding=True, truncation=True,
                    max_length=args.max_length, return_tensors="pt")
            t = {k: v.to(dev) for k, v in t.items()}
            hook_cache.clear()
            model(**t)
            h = hook_cache[-1]
            if pool_fn is not None:
                p = pool_fn(h, t["attention_mask"])
            else:
                p = mean_pooling(h, t["attention_mask"])
            p = F.normalize(p, p=2, dim=1)
            embs.append(p.cpu().float().numpy())
        return np.vstack(embs)

    def encode_flat(texts, pool_fn=None):
        """Same as encode but works on a flat list without batching issues."""
        return encode(texts, pool_fn=pool_fn)

    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("DATASET: %s", ds_name)
        ct, ci, qt, qi, qrels = load_data(ds_name, args.max_corpus,
                                          args.max_queries)
        ds_results = {}

        # ---- baseline_mean ----
        t0 = time.time()
        logger.info("[1] baseline_mean")
        c_emb = encode(ct)
        q_emb = encode(qt)
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["baseline_mean"] = {"desc": "Baseline: mean-pooling",
                                        "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m["ndcg@10"], m["recall@10"], time.time() - t0)

        # ---- RP1: L2-norm weighted ----
        t0 = time.time()
        logger.info("[2] rp1_l2_weighted")
        c_emb = encode(ct, pool_fn=l2_wp)
        q_emb = encode(qt, pool_fn=l2_wp)
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["rp1_l2_weighted"] = {"desc": "RP1: L2-norm weighted pooling",
                                          "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m["ndcg@10"], m["recall@10"], time.time() - t0)

        # ---- RP2: Chunk-based with mean aggregation ----
        t0 = time.time()
        logger.info("[3] rp2_chunk_mean")
        c_emb = chunk_encode(encode_flat, ct, chunk_size=args.chunk_size,
                             overlap=args.chunk_overlap, aggregation="mean")
        q_emb = chunk_encode(encode_flat, qt, chunk_size=args.chunk_size,
                             overlap=args.chunk_overlap, aggregation="mean")
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["rp2_chunk_mean"] = {"desc": "RP2: Chunk-based mean agg",
                                         "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m["ndcg@10"], m["recall@10"], time.time() - t0)

        # ---- RP2: Chunk-based with weighted aggregation ----
        t0 = time.time()
        logger.info("[4] rp2_chunk_weighted")
        c_emb = chunk_encode(encode_flat, ct, chunk_size=args.chunk_size,
                             overlap=args.chunk_overlap, aggregation="weighted")
        q_emb = chunk_encode(encode_flat, qt, chunk_size=args.chunk_size,
                             overlap=args.chunk_overlap, aggregation="weighted")
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["rp2_chunk_weighted"] = {
            "desc": "RP2: Chunk-based weighted agg", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m["ndcg@10"], m["recall@10"], time.time() - t0)

        # ---- RP3: Extractive compression ----
        t0 = time.time()
        logger.info("[5] rp3_extractive (compressing corpus...)")
        c_comp = [compress_extractive(encode_flat, t, args.compression_ratio)
                  for t in ct]
        c_emb = encode(c_comp)
        q_emb = encode(qt)  # don't compress queries
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["rp3_extractive"] = {"desc": "RP3: Extractive compression",
                                         "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m["ndcg@10"], m["recall@10"], time.time() - t0)

        # ---- RP3: Hierarchical compression ----
        t0 = time.time()
        logger.info("[6] rp3_hierarchical (compressing corpus...)")
        c_comp = [compress_hierarchical(encode_flat, t, args.compression_ratio)
                  for t in ct]
        c_emb = encode(c_comp)
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)  # q_emb from above
        ds_results["rp3_hierarchical"] = {
            "desc": "RP3: Hierarchical compression", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m["ndcg@10"], m["recall@10"], time.time() - t0)

        # ---- Combined RP2+RP3: compress → chunk → mean pool ----
        t0 = time.time()
        logger.info("[7] combined_rp23 (compress + chunk...)")
        c_comp = [compress_extractive(encode_flat, t, args.compression_ratio)
                  for t in ct]
        c_emb = chunk_encode(encode_flat, c_comp, chunk_size=args.chunk_size,
                             overlap=args.chunk_overlap, aggregation="mean")
        q_emb2 = chunk_encode(encode_flat, qt, chunk_size=args.chunk_size,
                              overlap=args.chunk_overlap, aggregation="mean")
        m = calc_metrics(q_emb2, c_emb, qrels, qi, ci)
        ds_results["combined_rp23"] = {
            "desc": "RP2+RP3: Compress + Chunk mean", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m["ndcg@10"], m["recall@10"], time.time() - t0)

        all_res[ds_name] = ds_results

        # Save per-dataset
        with open(out / f"{ds_name}_full.json", "w") as f:
            json.dump(ds_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s_full.json", ds_name)

    # ---- Full save ----
    with open(out / "advanced_full_results.json", "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("\n" + "=" * 85)
    print("ADVANCED EXPERIMENT RESULTS — COMPLETE (RP1 + RP2 + RP3)")
    print("=" * 85)
    for ds_name in all_res:
        print(f"\n{'=' * 40}")
        print(f"  {ds_name}")
        print(f"{'=' * 40}")
        hdr = (f"  {'Method':<22s} {'nDCG@10':>10s} {'Recall@10':>10s}"
               f" {'MRR@10':>10s} {'MAP@10':>10s}")
        print(hdr)
        print("  " + "-" * 64)
        baseline_ndcg = all_res[ds_name]["baseline_mean"]["metrics"]["ndcg@10"]
        for mn, ed in all_res[ds_name].items():
            m = ed["metrics"]
            delta = ""
            if mn != "baseline_mean" and baseline_ndcg > 0:
                pct = 100 * (m["ndcg@10"] - baseline_ndcg) / baseline_ndcg
                delta = f" ({pct:+.1f}%)"
            print(f"  {mn:<22s} {m['ndcg@10']:>10.4f} {m['recall@10']:>10.4f}"
                  f" {m['mrr@10']:>10.4f} {m['map@10']:>10.4f}{delta}")

    # Cleanup
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nSaved to: {out}/advanced_full_results.json")


if __name__ == "__main__":
    main()
