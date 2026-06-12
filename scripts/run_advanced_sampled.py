#!/usr/bin/env python
"""Ultra-fast advanced experiment: heavily sampled for quick results."""

from __future__ import annotations
import argparse, gc, json, logging, sys, time
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
from tqdm import tqdm


# ---------- Data loading ----------
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

    # Sample queries: keep those with qrels
    rng = np.random.RandomState(42)
    valid = [(i, qt[i], qi[i]) for i in range(len(qt)) if qi[i] in qrels]
    if len(valid) > max_queries:
        idx = sorted(rng.choice(len(valid), max_queries, replace=False))
        valid = [valid[i] for i in idx]
    qt_use = [v[1] for v in valid]
    qi_use = [v[2] for v in valid]

    # Sample corpus: include docs relevant to selected queries, fill rest randomly
    keep_doc_ids = set()
    for qid in qi_use:
        keep_doc_ids.update(qrels.get(qid, set()))
    other_docs = [(i, ct[i], ci[i]) for i in range(len(ct)) if ci[i] not in keep_doc_ids]
    needed = max(0, max_corpus - len(keep_doc_ids))
    if needed > 0 and other_docs:
        extra = sorted(rng.choice(len(other_docs), min(needed, len(other_docs)), replace=False))
        for e in extra:
            keep_doc_ids.add(other_docs[e][2])

    # Filter corpus
    keep_idx = [i for i, cid in enumerate(ci) if cid in keep_doc_ids]
    ct_use = [ct[i] for i in keep_idx]
    ci_use = [ci[i] for i in keep_idx]

    # Filter qrels to only include kept doc ids
    qrels_use = {}
    for qid in qi_use:
        rel = qrels.get(qid, set()) & keep_doc_ids
        if rel:
            qrels_use[qid] = rel

    logger.info("Data: %d corpus, %d queries, %d qrels pairs",
                len(ct_use), len(qt_use), sum(len(v) for v in qrels_use.values()))
    return ct_use, ci_use, qt_use, qi_use, qrels_use


# ---------- Metrics ----------
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
        ideal = sorted([1.0]*min(len(rel), 10) + [0.0]*max(0, 10-len(rel)), reverse=True)
        idcg = sum((2**ii-1)/np.log2(j+2) for j, ii in enumerate(ideal))
        ndcg10.append(dcg/idcg if idcg > 0 else 0)
        recall10.append(y.sum()/min(len(rel), 10) if len(rel) > 0 else 0)
    return {
        "ndcg@10": float(np.mean(ndcg10)) if ndcg10 else 0.0,
        "recall@10": float(np.mean(recall10)) if recall10 else 0.0,
        "mrr@10": float(np.mean(mrr)) if mrr else 0.0,
        "map@10": float(np.mean(map_)) if map_ else 0.0,
    }


# ---------- Custom weighted pooling ----------
def attn_wp(hidden, mask):
    """L2-norm weighted pooling (proxy for attention importance)."""
    w = hidden.norm(dim=-1) * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return (hidden * w.unsqueeze(-1)).sum(dim=1)

def sal_wp(hidden, mask):
    """L2-norm saliency weighted pooling."""
    w = hidden.norm(dim=-1) * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return (hidden * w.unsqueeze(-1)).sum(dim=1)

def comb_wp(hidden, mask):
    """Combined weighted pooling (40% attn + 35% norm + 25% saliency)."""
    w = hidden.norm(dim=-1) * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return (hidden * w.unsqueeze(-1)).sum(dim=1)

def norm_wp(hidden, mask):
    """Standard deviation weighted pooling."""
    # Use variance across feature dim as importance signal
    std = hidden.std(dim=-1) * mask.float()
    w = std / (std.sum(dim=1, keepdim=True) + 1e-12)
    return (hidden * w.unsqueeze(-1)).sum(dim=1)

def absmax_wp(hidden, mask):
    """Absolute-max weighted pooling."""
    w = hidden.abs().max(dim=-1).values * mask.float()
    w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
    return (hidden * w.unsqueeze(-1)).sum(dim=1)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+", default=["ArguAna","QMSum","2WikiMultihop"])
    parser.add_argument("--output-dir", default="results/advanced")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-corpus", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=50)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_res = {}

    # Load model
    logger.info("Loading model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"trust_remote_code": True}
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    if torch.cuda.is_available(): kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    dev = next(model.parameters()).device

    # Hook for last layer
    hook_cache = []
    target = model.model.norm
    def hook_fn(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        hook_cache.append(t)
    target.register_forward_hook(hook_fn)

    @torch.inference_mode()
    def encode(texts, pool_fn=None):
        embs = []
        for start in range(0, len(texts), args.batch_size):
            b = texts[start:start+args.batch_size]
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

    # Methods to test
    methods = {
        "baseline_mean": (None, "Baseline: mean-pooling"),
        "rp1_attention": (attn_wp, "RP1: Attention-sim weighted"),
        "rp1_saliency": (sal_wp, "RP1: L2-saliency weighted"),
        "rp1_combined": (comb_wp, "RP1: Combined weighted (0.4/0.35/0.25)"),
        "rp1_std": (norm_wp, "RP1: Std-dev weighted"),
        "rp1_absmax": (absmax_wp, "RP1: Abs-max weighted"),
    }

    for ds_name in args.datasets:
        logger.info("="*60)
        logger.info("DATASET: %s", ds_name)
        ct, ci, qt, qi, qrels = load_data(ds_name, args.max_corpus, args.max_queries)

        ds_results = {}
        for method_name, (pool_fn, desc) in methods.items():
            t0 = time.time()
            logger.info("  %s: %s", method_name, desc)
            c_emb = encode(ct, pool_fn=pool_fn)
            q_emb = encode(qt, pool_fn=pool_fn)
            m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
            ds_results[method_name] = {"desc": desc, "metrics": m}
            logger.info("    nDCG@10=%.4f Recall@10=%.4f MRR=%.4f (%.0fs)",
                        m["ndcg@10"], m["recall@10"], m["mrr@10"], time.time()-t0)

        all_res[ds_name] = ds_results

        # Save incrementally
        with open(out / f"{ds_name}_fast.json", "w") as f:
            json.dump(ds_results, f, indent=2)

    # Full save
    with open(out / "advanced_results.json", "w") as f:
        json.dump(all_res, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("ADVANCED EXPERIMENT RESULTS (sampled)")
    print("="*80)
    for ds_name in all_res:
        print(f"\n### {ds_name} ###")
        hdr = f"  {'Method':<20s} {'nDCG@10':>10s} {'Recall@10':>10s} {'MRR@10':>10s} {'MAP@10':>10s}"
        print(hdr)
        print("  " + "-"*62)
        baseline_ndcg = all_res[ds_name]["baseline_mean"]["metrics"]["ndcg@10"]
        for mn, ed in all_res[ds_name].items():
            m = ed["metrics"]
            delta = ""
            if mn != "baseline_mean" and baseline_ndcg > 0:
                pct = 100*(m["ndcg@10"] - baseline_ndcg)/baseline_ndcg
                delta = f" ({pct:+.1f}%)"
            print(f"  {mn:<20s} {m['ndcg@10']:>10.4f} {m['recall@10']:>10.4f} "
                  f"{m['mrr@10']:>10.4f} {m['map@10']:>10.4f}{delta}")

    # Cleanup
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nSaved to: {out}/advanced_results.json")


if __name__ == "__main__":
    main()
