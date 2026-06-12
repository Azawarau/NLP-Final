#!/usr/bin/env python
"""Properly evaluate and SAVE RP2 and RP3 results for QMSum and 2WikiMultihop.

RP2: chunk-based encoding with mean/weighted aggregation
RP3: extractive semantic compression
"""

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


# ======================== Data loading ========================
def load_data(ds_name, max_corpus=100, max_queries=50):
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
    logger.info("Data: %d corpus, %d queries, %d qrels",
                len(ct_use), len(qt_use), len(qrels_use))
    return ct_use, ci_use, qt_use, qi_use, qrels_use


# ======================== Metrics ========================
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


# ======================== Token-based chunking ========================
def token_chunk_texts(tok, texts, chunk_size=1024, overlap=128):
    step = chunk_size - overlap
    flat_chunks, chunk_lens = [], []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        chunks_for_text = []
        start = 0
        while start < len(ids):
            end = min(start + chunk_size, len(ids))
            chunk_ids = ids[start:end]
            chunk_text = tok.decode(chunk_ids, skip_special_tokens=True)
            chunks_for_text.append(chunk_text)
            if end >= len(ids):
                break
            start += step
        chunk_lens.append(len(chunks_for_text))
        flat_chunks.extend(chunks_for_text)
    return flat_chunks, chunk_lens


def aggregate_chunks(embeddings, chunk_lens, method="mean"):
    result = []
    pos = 0
    for n in chunk_lens:
        chunk_embs = embeddings[pos:pos+n]
        if method == "mean":
            emb = chunk_embs.mean(axis=0)
        else:
            norms = np.linalg.norm(chunk_embs, axis=1) + 1e-9
            emb = np.average(chunk_embs, axis=0, weights=norms)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        result.append(emb)
        pos += n
    return np.stack(result)


# ======================== RP3: Batched extractive compression ========================
def split_sentences(text, max_sents=200):
    sents = re.split(r'(?<=[.!?])\s+', text)
    clean = [s.strip() for s in sents if s.strip()]
    if len(clean) > max_sents:
        step = len(clean) / max_sents
        clean = [clean[int(i * step)] for i in range(max_sents)]
    return clean


def compress_extractive_batched(encode_fn, texts, ratio=0.3, min_sents=3):
    all_sents = []
    text_to_sent_idxs = []
    for ti, text in enumerate(texts):
        sents = split_sentences(text)
        start = len(all_sents)
        all_sents.extend(sents)
        text_to_sent_idxs.append(list(range(start, len(all_sents))))

    logger.info("    %d texts → %d sentences", len(texts), len(all_sents))
    sent_embs = encode_fn(all_sents)

    compressed = []
    for ti in range(len(texts)):
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


# ======================== Main ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+", default=["QMSum", "2WikiMultihop"])
    parser.add_argument("--output-dir", default="results/advanced")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-corpus", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=128)
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
    def encode(texts, bs=4, ml=None):
        max_len = ml or args.max_length
        embs = []
        for start in range(0, len(texts), bs):
            b = texts[start:start+bs]
            t = tok(b, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
            t = {k: v.to(dev) for k, v in t.items()}
            hook_cache.clear()
            model(**t)
            h = hook_cache[-1]
            p = mean_pooling(h, t["attention_mask"])
            p = F.normalize(p, p=2, dim=1)
            embs.append(p.cpu().float().numpy())
        return np.vstack(embs)

    def encode_sentences(texts):
        return encode(texts, bs=64, ml=128)

    for ds_name in args.datasets:
        logger.info("=" * 60)
        logger.info("DATASET: %s", ds_name)
        ct, ci, qt, qi, qrels = load_data(ds_name, args.max_corpus, args.max_queries)

        ds_result = {}

        # ----- baseline_mean -----
        t0 = time.time()
        logger.info("[1] baseline_mean")
        c_emb = encode(ct)
        q_emb = encode(qt)
        m_bl = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_result["baseline_mean"] = {"ndcg@10": m_bl["ndcg@10"],
                                       "recall@10": m_bl["recall@10"]}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m_bl["ndcg@10"], m_bl["recall@10"], time.time()-t0)

        # ----- RP2: chunk mean -----
        t0 = time.time()
        logger.info("[2] rp2_chunk_mean (token-based chunking...)")
        flat_c, c_lens = token_chunk_texts(tok, ct, args.chunk_size, args.chunk_overlap)
        flat_q, q_lens = token_chunk_texts(tok, qt, args.chunk_size, args.chunk_overlap)
        logger.info("  %d texts → %d chunks (corpus), %d → %d (queries)",
                    len(ct), len(flat_c), len(qt), len(flat_q))
        c_chunk_emb = encode(flat_c)
        q_chunk_emb = encode(flat_q)
        c_emb_rp2 = aggregate_chunks(c_chunk_emb, c_lens, "mean")
        q_emb_rp2 = aggregate_chunks(q_chunk_emb, q_lens, "mean")
        m_rp2m = calc_metrics(q_emb_rp2, c_emb_rp2, qrels, qi, ci)
        ds_result["rp2_chunk_mean"] = {"ndcg@10": m_rp2m["ndcg@10"],
                                        "recall@10": m_rp2m["recall@10"]}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m_rp2m["ndcg@10"], m_rp2m["recall@10"], time.time()-t0)

        # ----- RP2: chunk weighted -----
        c_emb_rp2w = aggregate_chunks(c_chunk_emb, c_lens, "weighted")
        q_emb_rp2w = aggregate_chunks(q_chunk_emb, q_lens, "weighted")
        m_rp2w = calc_metrics(q_emb_rp2w, c_emb_rp2w, qrels, qi, ci)
        ds_result["rp2_chunk_weighted"] = {"ndcg@10": m_rp2w["ndcg@10"],
                                            "recall@10": m_rp2w["recall@10"]}
        logger.info("[3] rp2_chunk_weighted: nDCG@10=%.4f R@10=%.4f",
                    m_rp2w["ndcg@10"], m_rp2w["recall@10"])

        # ----- RP3: extractive compression -----
        t0 = time.time()
        logger.info("[4] rp3_extractive (batched sentence encoding)")
        c_comp = compress_extractive_batched(encode_sentences, ct, args.compression_ratio)
        c_emb_rp3 = encode(c_comp)
        m_rp3 = calc_metrics(q_emb, c_emb_rp3, qrels, qi, ci)  # reuse baseline q_emb
        ds_result["rp3_extractive"] = {"ndcg@10": m_rp3["ndcg@10"],
                                        "recall@10": m_rp3["recall@10"]}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m_rp3["ndcg@10"], m_rp3["recall@10"], time.time()-t0)

        # ----- RP2+RP3 Combined: compress + chunk -----
        t0 = time.time()
        logger.info("[5] combined_rp23 (compress + chunk...)")
        flat_c2, c_lens2 = token_chunk_texts(tok, c_comp, args.chunk_size, args.chunk_overlap)
        flat_q2, q_lens2 = token_chunk_texts(tok, qt, args.chunk_size, args.chunk_overlap)
        logger.info("  compressed %d texts → %d chunks (corpus), %d → %d (queries)",
                    len(c_comp), len(flat_c2), len(qt), len(flat_q2))
        c_chunks_emb2 = encode(flat_c2)
        q_chunks_emb2 = encode(flat_q2)
        c_emb_rp23 = aggregate_chunks(c_chunks_emb2, c_lens2, "mean")
        q_emb_rp23 = aggregate_chunks(q_chunks_emb2, q_lens2, "mean")
        m_rp23 = calc_metrics(q_emb_rp23, c_emb_rp23, qrels, qi, ci)
        ds_result["combined_rp23"] = {"ndcg@10": m_rp23["ndcg@10"],
                                       "recall@10": m_rp23["recall@10"]}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)",
                    m_rp23["ndcg@10"], m_rp23["recall@10"], time.time()-t0)

        # ----- Save -----
        out_path = out / f"{ds_name}_full.json"
        with open(out_path, "w") as f:
            json.dump(ds_result, f, indent=2, ensure_ascii=False)
        logger.info("Saved: %s", out_path)

        # Print summary line
        for method in ["baseline_mean", "rp2_chunk_mean", "rp3_extractive", "combined_rp23"]:
            m = ds_result[method]
            pct = ""
            if method != "baseline_mean" and m_bl["ndcg@10"] > 0:
                pct = f" ({100*(m['ndcg@10']-m_bl['ndcg@10'])/m_bl['ndcg@10']:+.1f}%)"
            logger.info("  %s: nDCG@10=%.4f R@10=%.4f%s",
                        method, m["ndcg@10"], m["recall@10"], pct)

    del model; gc.collect(); torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
