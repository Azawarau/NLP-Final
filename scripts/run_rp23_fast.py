#!/usr/bin/env python
"""Fast RP2 + RP3 experiments. Optimized: encodes all chunks in one flat batch call."""

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
    other = [(i, ct[i], ci[i]) for i in range(len(ct)) if ci[i] not in keep_doc_ids]
    needed = max(0, max_corpus - len(keep_doc_ids))
    if needed > 0 and other:
        extra = sorted(rng.choice(len(other), min(needed, len(other)), replace=False))
        for e in extra:
            keep_doc_ids.add(other[e][2])

    keep_idx = [i for i, cid in enumerate(ci) if cid in keep_doc_ids]
    ct_use = [ct[i] for i in keep_idx]
    ci_use = [ci[i] for i in keep_idx]
    qrels_use = {qid: qrels.get(qid, set()) & keep_doc_ids for qid in qi_use}
    qrels_use = {k: v for k, v in qrels_use.items() if v}

    logger.info("Data: %d corpus, %d queries, %d qrels pairs",
                len(ct_use), len(qt_use), sum(len(v) for v in qrels_use.values()))
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
def token_chunk_texts(tok, texts, chunk_size=512, overlap=64):
    """Split texts into chunks using the tokenizer. Returns (flat_chunks, chunk_lens).
    chunk_lens[i] = number of chunks for text i.
    """
    step = chunk_size - overlap
    flat_chunks = []
    chunk_lens = []
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
    """Re-group flat embeddings per text and aggregate."""
    result = []
    pos = 0
    for n in chunk_lens:
        chunk_embs = embeddings[pos:pos+n]
        if method == "mean":
            emb = chunk_embs.mean(axis=0)
        else:  # weighted by L2-norm
            norms = np.linalg.norm(chunk_embs, axis=1) + 1e-9
            emb = np.average(chunk_embs, axis=0, weights=norms)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        result.append(emb)
        pos += n
    return np.stack(result)


# ======================== Compression ========================
def compress_extractive(encode_fn, text, ratio=0.3, min_sents=3):
    sents = re.split(r'(?<=[.!?])\s+', text)
    if len(sents) <= min_sents:
        return text
    sent_embs = encode_fn(sents)
    sim_matrix = sent_embs @ sent_embs.T
    centrality = sim_matrix.mean(axis=1)
    k = max(min_sents, int(len(sents) * ratio))
    top_idx = np.argsort(centrality)[-k:]
    top_idx = sorted(top_idx)
    return ' '.join(sents[i] for i in top_idx)


# ======================== Main ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+", default=["ArguAna","QMSum","2WikiMultihop"])
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
    def encode_batch(texts):
        """Encode a list of texts in batches, return numpy array."""
        embs = []
        for start in range(0, len(texts), args.batch_size):
            b = texts[start:start+args.batch_size]
            t = tok(b, padding=True, truncation=True,
                    max_length=args.max_length, return_tensors="pt")
            t = {k: v.to(dev) for k, v in t.items()}
            hook_cache.clear()
            model(**t)
            h = hook_cache[-1]
            p = mean_pooling(h, t["attention_mask"])
            p = F.normalize(p, p=2, dim=1)
            embs.append(p.cpu().float().numpy())
        return np.vstack(embs)

    for ds_name in args.datasets:
        logger.info("="*60)
        logger.info("DATASET: %s", ds_name)
        ct, ci, qt, qi, qrels = load_data(ds_name, args.max_corpus, args.max_queries)
        ds_results = {}

        # --- baseline ---
        t0 = time.time()
        logger.info("[1] baseline_mean")
        c_emb = encode_batch(ct)
        q_emb = encode_batch(qt)
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["baseline_mean"] = {"desc": "Baseline mean-pooling", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        # --- RP2: Chunk mean ---
        t0 = time.time()
        logger.info("[2] rp2_chunk_mean (token-based chunking...)")
        flat_c, c_lens = token_chunk_texts(tok, ct, args.chunk_size, args.chunk_overlap)
        flat_q, q_lens = token_chunk_texts(tok, qt, args.chunk_size, args.chunk_overlap)
        logger.info("  %d texts -> %d chunks (corpus), %d -> %d (queries)",
                    len(ct), len(flat_c), len(qt), len(flat_q))
        c_chunk_emb = encode_batch(flat_c)
        q_chunk_emb = encode_batch(flat_q)
        c_emb = aggregate_chunks(c_chunk_emb, c_lens, "mean")
        q_emb = aggregate_chunks(q_chunk_emb, q_lens, "mean")
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["rp2_chunk_mean"] = {"desc": "RP2: Chunk-based mean agg", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        # --- RP2: Chunk weighted ---
        t0 = time.time()
        logger.info("[3] rp2_chunk_weighted")
        c_emb = aggregate_chunks(c_chunk_emb, c_lens, "weighted")
        q_emb = aggregate_chunks(q_chunk_emb, q_lens, "weighted")
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["rp2_chunk_weighted"] = {"desc": "RP2: Chunk-based weighted agg", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        # --- RP3: Extractive compression ---
        t0 = time.time()
        logger.info("[4] rp3_extractive (compressing corpus...)")
        c_comp = []
        for i, t in enumerate(ct):
            c_comp.append(compress_extractive(encode_batch, t, args.compression_ratio))
            if (i+1) % 50 == 0:
                logger.info("  compressed %d/%d", i+1, len(ct))
        c_emb = encode_batch(c_comp)
        q_emb = encode_batch(qt)  # don't compress queries
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["rp3_extractive"] = {"desc": "RP3: Extractive compression", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        # --- RP3: Hierarchical ---
        t0 = time.time()
        logger.info("[5] rp3_hierarchical")
        c_comp = []
        for i, t in enumerate(ct):
            paras = [p.strip() for p in t.split('\n') if p.strip()]
            if len(paras) <= 3:
                c_comp.append(compress_extractive(encode_batch, t, args.compression_ratio))
            else:
                stage1 = []
                for para in paras:
                    if len(para.split()) > 30:
                        stage1.append(compress_extractive(encode_batch, para, 0.5))
                    else:
                        stage1.append(para)
                combined = ' '.join(stage1)
                c_comp.append(compress_extractive(encode_batch, combined, args.compression_ratio))
            if (i+1) % 50 == 0:
                logger.info("  compressed %d/%d", i+1, len(ct))
        c_emb = encode_batch(c_comp)
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)  # reuse q_emb from extractive
        ds_results["rp3_hierarchical"] = {"desc": "RP3: Hierarchical compression", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        # --- RP2+RP3 combined ---
        t0 = time.time()
        logger.info("[6] combined_rp23 (compress + chunk...)")
        flat_c2, c_lens2 = token_chunk_texts(tok, c_comp, args.chunk_size, args.chunk_overlap)
        flat_q2, q_lens2 = token_chunk_texts(tok, qt, args.chunk_size, args.chunk_overlap)
        c_chunks_emb = encode_batch(flat_c2)
        q_chunks_emb = encode_batch(flat_q2)
        c_emb = aggregate_chunks(c_chunks_emb, c_lens2, "mean")
        q_emb = aggregate_chunks(q_chunks_emb, q_lens2, "mean")
        m = calc_metrics(q_emb, c_emb, qrels, qi, ci)
        ds_results["combined_rp23"] = {"desc": "RP2+RP3: Compress + Chunk", "metrics": m}
        logger.info("  nDCG@10=%.4f R@10=%.4f (%.0fs)", m["ndcg@10"], m["recall@10"], time.time()-t0)

        all_res[ds_name] = ds_results
        with open(out / f"{ds_name}_full.json", "w") as f:
            json.dump(ds_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s_full.json", ds_name)

    # Save
    with open(out / "advanced_full_results.json", "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "="*85)
    print("ADVANCED EXPERIMENT RESULTS (RP1 + RP2 + RP3)")
    print("="*85)
    for ds_name in all_res:
        print(f"\n### {ds_name} ###")
        hdr = f"  {'Method':<22s} {'nDCG@10':>10s} {'Recall@10':>10s}"
        print(hdr)
        print("  "+"-"*44)
        bl = all_res[ds_name]["baseline_mean"]["metrics"]["ndcg@10"]
        for mn, ed in all_res[ds_name].items():
            m = ed["metrics"]
            d = ""
            if mn != "baseline_mean" and bl > 0:
                pct = 100*(m["ndcg@10"]-bl)/bl
                d = f" ({pct:+.1f}%)"
            print(f"  {mn:<22s} {m['ndcg@10']:>10.4f} {m['recall@10']:>10.4f}{d}")

    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"\nSaved to: {out}/advanced_full_results.json")

if __name__ == "__main__":
    main()
