#!/usr/bin/env python
"""一次性实验：2 方法 × 3 数据集 × 5 层。核心优化：每方法每数据集只编码 1 次，同时提取所有层。"""
from datasets import load_dataset  # before torch

import json, sys, time, gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pooling import mean_pooling, last_token_pooling
from src.prompts import build_prompteol

MODEL = "models/Mistral-7B-Instruct-v0.3"
LAYERS = [-1, 8, 16, 24, 32]  # -1=last, others=specific
METHODS = ["prompteol", "mean"]
MAX_LEN = 4096; BS = 4; OUT = Path("results/all")


def load_longembed(name):
    c = load_dataset("dwzhu/LongEmbed", name=name, split="corpus")
    q = load_dataset("dwzhu/LongEmbed", name=name, split="queries")
    r = load_dataset("dwzhu/LongEmbed", name=name, split="qrels")
    qrels = defaultdict(set)
    for row in r: qrels[str(row.get("qid",""))].add(str(row.get("doc_id","")))
    return ([d["text"] for d in c], [str(d.get("doc_id","")) for d in c],
            [d["text"] for d in q], [str(d.get("qid","")) for d in q], dict(qrels))

def load_arguana():
    c = load_dataset("mteb/arguana", "corpus", split="corpus")
    q = load_dataset("mteb/arguana", "queries", split="queries")
    r = load_dataset("mteb/arguana", split="test")
    qrels = defaultdict(set)
    for row in r:
        if float(row["score"]) > 0: qrels[str(row["query-id"])].add(str(row["corpus-id"]))
    return ([f"{d.get('title','')}\n{d.get('text','')}".strip() for d in c],
            [str(d["_id"]) for d in c], [d["text"] for d in q],
            [str(d["_id"]) for d in q], dict(qrels))


def calc_metrics(q_emb, c_emb, qrels, q_ids, c_ids, ks=(1, 10)):
    q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    c = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
    scores = q @ c.T
    mk = max(ks)
    ndcg, recall, mrr, map_ = {k: [] for k in ks}, {k: [] for k in ks}, [], []
    for qi, qid in enumerate(q_ids):
        if qid not in qrels or not qrels[qid]: continue
        rel = qrels[qid]; top = np.argsort(scores[qi])[::-1][:mk]
        rr = next((1/(r+1) for r, i in enumerate(top) if c_ids[i] in rel), 0)
        mrr.append(rr); ap = rc = 0
        for r, i in enumerate(top):
            if c_ids[i] in rel: rc += 1; ap += rc/(r+1)
        map_.append(ap/min(len(rel), mk) if rc > 0 else 0)
        for k in ks:
            t = top[:k]; y = np.array([1 if c_ids[i] in rel else 0 for i in t])
            d = sum((2**yi-1)/np.log2(j+2) for j, yi in enumerate(y))
            ideal = sorted([1.0]*min(len(rel), k)+[0.0]*(k-min(len(rel), k)), reverse=True)
            idc = sum((2**ii-1)/np.log2(j+2) for j, ii in enumerate(ideal))
            ndcg[k].append(d/idc if idc > 0 else 0)
            recall[k].append(y.sum()/min(len(rel), k) if len(rel)>0 else 0)
    return {f"ndcg@{k}": float(np.mean(ndcg[k])) if ndcg[k] else 0 for k in ks} | \
           {f"recall@{k}": float(np.mean(recall[k])) if recall[k] else 0 for k in ks} | \
           {"mrr@10": float(np.mean(mrr)) if mrr else 0, "map@10": float(np.mean(map_)) if map_ else 0}


class Enc:
    def __init__(self):
        print("Loading Mistral-7B (4bit)...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        kw = {"trust_remote_code": True, "output_hidden_states": True}
        if torch.cuda.is_available():
            kw["device_map"] = "auto"
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
        self.model.eval(); self.N = self.model.config.num_hidden_layers
        self.dev = next(self.model.parameters()).device
        print(f"  OK: {self.N} layers, {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    def _idx(self, layer):
        return self.N if layer in (-1, self.N) else (self.N + layer + 1 if layer < 0 else layer)

    @torch.inference_mode()
    def encode_all_layers(self, texts, method):
        """Encode once, return {layer_idx: np.ndarray(embeddings)} for all LAYERS."""
        layer_idxs = {l: self._idx(l) for l in LAYERS}
        results = {l: [] for l in LAYERS}
        for i in tqdm(range(0, len(texts), BS), desc=f"{method}", leave=False):
            batch = texts[i:i+BS]
            if method == "prompteol": batch = [build_prompteol(t) for t in batch]
            tok = self.tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            tok = {k: v.to(self.dev) for k, v in tok.items()}
            out = self.model(**tok)
            for layer, li in layer_idxs.items():
                h = out.hidden_states[li]
                p = last_token_pooling(h, tok["attention_mask"]) if method == "prompteol" else mean_pooling(h, tok["attention_mask"])
                results[layer].append(F.normalize(p, p=2, dim=1).cpu().float().numpy())
        return {l: np.vstack(v) for l, v in results.items()}

    def cleanup(self):
        del self.model; gc.collect(); torch.cuda.empty_cache()


def main():
    datasets = [
        ("ArguAna", load_arguana),
        ("QMSum", lambda: load_longembed("qmsum")),
        ("2WikiMultihop", lambda: load_longembed("2wikimqa")),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    all_res = {}
    enc = Enc()

    for ds_name, loader in datasets:
        print(f"\n{'='*50}\n  {ds_name}\n{'='*50}", flush=True)
        c_texts, c_ids, q_texts, q_ids, qrels = loader()
        print(f"  corpus={len(c_texts)} queries={len(q_texts)} rel_q={len(qrels)}", flush=True)
        ds_res = {}

        for method in METHODS:
            t0 = time.time()
            c_embs = enc.encode_all_layers(c_texts, method)  # {layer: emb}
            q_embs = enc.encode_all_layers(q_texts, method)
            print(f"  [{method}] encoded in {time.time()-t0:.0f}s", flush=True)

            for layer in LAYERS:
                m = calc_metrics(q_embs[layer], c_embs[layer], qrels, q_ids, c_ids)
                key = f"{method}_L{layer}"
                ds_res[key] = m
                print(f"    {key}: nDCG@10={m['ndcg@10']:.4f} R@10={m['recall@10']:.4f} MRR={m['mrr@10']:.4f} MAP={m['map@10']:.4f}", flush=True)

        all_res[ds_name] = ds_res
        (OUT / f"{ds_name}.json").write_text(json.dumps(ds_res, indent=2))
        print(f"  saved -> results/all/{ds_name}.json", flush=True)

    enc.cleanup()
    (OUT / "all.json").write_text(json.dumps(all_res, indent=2))

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for ds, res in all_res.items():
        print(f"\n### {ds} ###")
        keys = sorted(res.keys())
        print(f"{'':>25s}", end="")
        for k in keys: print(f" {k:>16s}", end="")
        print()
        for m in ["ndcg@10", "recall@10", "mrr@10", "map@10"]:
            print(f"{m:>25s}", end="")
            for k in keys: print(f" {res[k][m]:>16.4f}", end="")
            print()

    print(f"\nSaved: results/all/all.json")

if __name__ == "__main__":
    main()
