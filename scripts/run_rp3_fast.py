#!/usr/bin/env python
"""Fast RP3: batch ALL sentences across ALL texts, compress, evaluate."""

import argparse, gc, json, logging, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.pooling import mean_pooling


def load_data(ds_name, max_corpus=100, max_queries=50):
    if ds_name == "ArguAna":
        c = load_dataset("mteb/arguana", "corpus", split="corpus")
        q = load_dataset("mteb/arguana", "queries", split="queries")
        r = load_dataset("mteb/arguana", "default", split="test")
        ct, ci = [], []
        for doc in c:
            title = doc.get('title', '') or ''
            txt = doc.get('text', '') or ''
            ct.append(f"{title}\n{txt}".strip() if title else txt)
            ci.append(str(doc["_id"]))
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
    qt_use, qi_use = [v[1] for v in valid], [v[2] for v in valid]

    keep = set()
    for qid in qi_use:
        keep.update(qrels.get(qid, set()))
    other = [(i, ct[i], ci[i]) for i in range(len(ct)) if ci[i] not in keep]
    needed = max(0, max_corpus - len(keep))
    if needed > 0 and other:
        for e in sorted(rng.choice(len(other), min(needed, len(other)), replace=False)):
            keep.add(other[e][2])
    kidx = [i for i, cid in enumerate(ci) if cid in keep]
    ct_use = [ct[i] for i in kidx]
    ci_use = [ci[i] for i in kidx]
    qrels_use = {qid: qrels.get(qid, set()) & keep for qid in qi_use}
    qrels_use = {k: v for k, v in qrels_use.items() if v}
    logger.info("Data: %d corpus, %d queries", len(ct_use), len(qt_use))
    return ct_use, ci_use, qt_use, qi_use, qrels_use


def calc_metrics(q_emb, c_emb, qrels, q_ids, c_ids):
    q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    c = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
    scores = q @ c.T
    ndcg10, recall10 = [], []
    for qi, qid in enumerate(q_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        rel = qrels[qid]
        top = np.argsort(scores[qi])[::-1][:10]
        y = np.array([1 if c_ids[i] in rel else 0 for i in top])
        dcg = sum((2**yi-1)/np.log2(j+2) for j, yi in enumerate(y))
        ideal = sorted([1.0]*min(len(rel), 10) + [0.0]*max(0, 10-len(rel)), reverse=True)
        idcg = sum((2**ii-1)/np.log2(j+2) for j, ii in enumerate(ideal))
        ndcg10.append(dcg/idcg if idcg > 0 else 0)
        recall10.append(y.sum()/min(len(rel), 10) if len(rel) > 0 else 0)
    return {"ndcg@10": float(np.mean(ndcg10)) if ndcg10 else 0.0,
            "recall@10": float(np.mean(recall10)) if recall10 else 0.0}


def split_sentences(text, max_sents=200):
    """Split text into sentences, filtering empty/whitespace-only, cap at max_sents."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    clean = [s.strip() for s in sents if s.strip()]
    if len(clean) > max_sents:
        # Take evenly-spaced sentences to cover the whole text
        step = len(clean) / max_sents
        clean = [clean[int(i * step)] for i in range(max_sents)]
    return clean


def compress_extractive_batched(encode_fn, texts, ratio=0.3, min_sents=3):
    """Compress ALL texts by batching ALL sentences from ALL texts together."""
    # Phase 1: gather all sentences with text index
    all_sents = []
    text_to_sent_idxs = []  # list of lists: per-text sentence indices in all_sents
    for ti, text in enumerate(texts):
        sents = split_sentences(text)
        start = len(all_sents)
        all_sents.extend(sents)
        text_to_sent_idxs.append(list(range(start, len(all_sents))))

    n_texts = len(texts)
    n_sents = len(all_sents)
    logger.info("    %d texts → %d sentences", n_texts, n_sents)

    # Phase 2: encode ALL sentences at once (batched)
    sent_embs = encode_fn(all_sents)

    # Phase 3: per text, compute centrality, select top
    compressed = []
    for ti in range(n_texts):
        idxs = text_to_sent_idxs[ti]
        if len(idxs) <= min_sents:
            compressed.append(texts[ti])
            continue
        embs = sent_embs[idxs]
        sims = embs @ embs.T
        centrality = sims.mean(axis=1)
        k = max(min_sents, int(len(idxs) * ratio))
        top_local = np.argsort(centrality)[-k:]
        top_local = sorted(top_local)
        orig_sents = [all_sents[idxs[j]] for j in top_local]
        compressed.append(' '.join(orig_sents))

    return compressed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+", default=["QMSum", "2WikiMultihop"])
    parser.add_argument("--output-dir", default="results/advanced")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-corpus", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--compression-ratio", type=float, default=0.3)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    logger.info("Loading Mistral-7B (4bit)...")
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
    def hook_fn(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        hook_cache.append(t)
    model.model.norm.register_forward_hook(hook_fn)

    @torch.inference_mode()
    def encode(texts, bs=4, max_len=None):
        ml = max_len or args.max_length
        embs = []
        for start in range(0, len(texts), bs):
            b = texts[start:start+bs]
            t = tok(b, padding=True, truncation=True,
                    max_length=ml, return_tensors="pt")
            t = {k: v.to(dev) for k, v in t.items()}
            hook_cache.clear()
            model(**t)
            h = hook_cache[-1]
            p = mean_pooling(h, t["attention_mask"])
            p = F.normalize(p, p=2, dim=1)
            embs.append(p.cpu().float().numpy())
            if len(embs) % 50 == 0:
                logger.info("    encoded %d/%d", start + bs, len(texts))
        return np.vstack(embs)

    def encode_sentences(texts):
        """Fast encoding for short sentences: max_len=128, bs=64."""
        return encode(texts, bs=64, max_len=128)

    all_results = {}

    for ds_name in args.datasets:
        logger.info("="*60)
        logger.info("DATASET: %s", ds_name)
        ct, ci, qt, qi, qrels = load_data(ds_name, args.max_corpus, args.max_queries)
        ds_res = {}

        # [1] Baseline
        t0 = time.time()
        logger.info("[1] baseline_mean")
        c_emb = encode(ct)
        q_emb = encode(qt)
        m_bl = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_res["baseline_mean"] = {"ndcg@10": m_bl["ndcg@10"], "recall@10": m_bl["recall@10"]}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m_bl["ndcg@10"], m_bl["recall@10"], time.time()-t0)

        # [2] Extractive compression
        t0 = time.time()
        logger.info("[2] rp3_extractive (batched all-sentences)")
        c_comp = compress_extractive_batched(encode_sentences, ct, args.compression_ratio)
        c_emb = encode(c_comp)
        m_ex = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_res["rp3_extractive"] = {"ndcg@10": m_ex["ndcg@10"], "recall@10": m_ex["recall@10"]}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m_ex["ndcg@10"], m_ex["recall@10"], time.time()-t0)

        # [3] Hierarchical compression (2-stage)
        t0 = time.time()
        logger.info("[3] rp3_hierarchical (2-stage batched)")
        # Stage 1: split by newlines as paragraphs, compress each
        t1 = time.time()
        all_paras = []
        para_to_text = []
        for ti, text in enumerate(ct):
            paras = [p.strip() for p in text.split('\n') if p.strip()]
            if not paras:
                paras = [text]  # fallback
            for p in paras:
                all_paras.append(p)
                para_to_text.append(ti)
        n_paras = len(all_paras)
        logger.info("  Stage 1: %d texts → %d paragraphs", len(ct), n_paras)
        p_compressed = compress_extractive_batched(encode_sentences, all_paras, ratio=0.5, min_sents=2)

        # Reassemble per text
        text_paras = [[] for _ in range(len(ct))]
        for pi, ti in enumerate(para_to_text):
            if len(all_paras[pi].split()) > 30:
                text_paras[ti].append(p_compressed[pi])
            else:
                text_paras[ti].append(all_paras[pi])
        stage1_texts = ['\n'.join(tp) for tp in text_paras]
        logger.info("  Stage 1 done (%.0fs). Stage 2: global compression", time.time()-t1)

        # Stage 2: global compression across paragraphs
        t2 = time.time()
        c_hier = compress_extractive_batched(encode_sentences, stage1_texts, ratio=args.compression_ratio, min_sents=3)
        c_emb = encode(c_hier)
        m_hi = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_res["rp3_hierarchical"] = {"ndcg@10": m_hi["ndcg@10"], "recall@10": m_hi["recall@10"]}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs total)", m_hi["ndcg@10"], m_hi["recall@10"], time.time()-t0)

        all_results[ds_name] = ds_res
        with open(out / f"{ds_name}_rp3.json", "w") as f:
            json.dump(ds_res, f, indent=2)
        logger.info("Saved %s_rp3.json", ds_name)

    # Summary
    with open(out / "rp3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print("RP3 SEMANTIC COMPRESSION RESULTS")
    print("="*70)
    for ds_name in all_results:
        print(f"\n### {ds_name} ###")
        bl = all_results[ds_name]["baseline_mean"]["ndcg@10"]
        for method in ["baseline_mean", "rp3_extractive", "rp3_hierarchical"]:
            if method not in all_results[ds_name]:
                continue
            m = all_results[ds_name][method]
            d = ""
            if method != "baseline_mean" and bl > 0:
                pct = 100*(m["ndcg@10"]-bl)/bl
                d = f" ({pct:+.1f}%)"
            print(f"  {method:<20s} nDCG@10={m['ndcg@10']:.4f} R@10={m['recall@10']:.4f}{d}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nSaved to: {out}/rp3_results.json")


if __name__ == "__main__":
    main()
