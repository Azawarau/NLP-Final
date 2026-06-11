#!/usr/bin/env python
"""ArguAna layer ablation — loads model ONCE, extracts 4 layers in one pass."""
from __future__ import annotations

# *** CRITICAL: datasets must be imported before torch (DLL conflict) ***
from datasets import load_dataset  # noqa: E402

import json, time, logging, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "results" / "layer_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(ROOT / "models" / "Mistral-7B-Instruct-v0.3")
MAX_LENGTH = 512
BATCH_SIZE = 8
LAYERS = [8, 16, 24, 32]


def load_arguana():
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
    k_values = [1, 10]
    q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
    c_norm = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-9)
    scores = q_norm @ c_norm.T
    max_k = max(k_values)
    ndcg_k = {k: [] for k in k_values}
    recall_k = {k: [] for k in k_values}
    precision_k = {k: [] for k in k_values}
    map_k = {k: [] for k in k_values}
    mrr_scores = []
    total = 0
    for qi, qid in enumerate(q_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        total += 1
        relevant = qrels[qid]
        top_indices = np.argsort(scores[qi])[::-1][:max_k]
        rr = 0.0
        for rank, idx in enumerate(top_indices):
            if c_ids[idx] in relevant:
                rr = 1.0 / (rank + 1)
                break
        mrr_scores.append(rr)
        for k in k_values:
            top_k = top_indices[:k]
            y_true = np.zeros(k)
            for rank, idx in enumerate(top_k):
                if c_ids[idx] in relevant:
                    y_true[rank] = 1.0
            dcg = sum((2**y_true[i] - 1) / np.log2(i + 2) for i in range(k))
            ideal = sorted([1.0]*min(len(relevant), k) + [0.0]*max(0, k-min(len(relevant), k)), reverse=True)
            idcg = sum((2**ideal[i] - 1) / np.log2(i + 2) for i in range(k))
            ndcg_k[k].append(dcg/idcg if idcg > 0 else 0.0)
            rel_found = sum(1 for idx in top_k if c_ids[idx] in relevant)
            recall_k[k].append(rel_found / min(len(relevant), k) if len(relevant) > 0 else 0)
            precision_k[k].append(rel_found / k)
            # MAP: average precision @ k
            ap_num = 0.0
            ap_hits = 0
            for rank, idx in enumerate(top_k):
                if c_ids[idx] in relevant:
                    ap_hits += 1
                    ap_num += ap_hits / (rank + 1)
            ap_denom = min(len(relevant), k)
            map_k[k].append(ap_num / ap_denom if ap_denom > 0 else 0.0)
    metrics = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = float(np.mean(ndcg_k[k])) if ndcg_k[k] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recall_k[k])) if recall_k[k] else 0.0
        metrics[f"precision@{k}"] = float(np.mean(precision_k[k])) if precision_k[k] else 0.0
        metrics[f"map@{k}"] = float(np.mean(map_k[k])) if map_k[k] else 0.0
    metrics["mrr@10"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    return metrics


def encode_all(texts, tokenizer, model, method, layers, max_length, batch_size):
    """Encode texts, extracting hidden states for all target layers in one pass."""
    from src.prompts import build_prompteol

    all_hidden = {layer: [] for layer in layers}
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        if method == "prompteol":
            batch = [build_prompteol(t) for t in batch]

        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt")
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            outputs = model(**enc, output_hidden_states=True)

        for layer in layers:
            hidden = outputs.hidden_states[layer]  # 0=embed, 1..N=decoder layers
            if method == "prompteol":
                # Last non-padding token
                mask = enc["attention_mask"]
                seq_lens = mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=device), seq_lens]
            else:
                # Mean pooling
                mask_expanded = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

            pooled = F.normalize(pooled, p=2, dim=1)
            all_hidden[layer].append(pooled.cpu().float().numpy())

        if (start // batch_size + 1) % 50 == 0:
            end = min(start + batch_size, total)
            logger.info("  encode[%s] %d/%d", method, end, total)

    return {layer: np.vstack(all_hidden[layer]) for layer in layers}


def main():
    # Load model once with output_hidden_states support
    from src.llm_encoder import LLMEmbeddingEncoder

    logger.info("Loading model...")
    t0 = time.time()
    # Use the encoder just for loading model + tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.eval()
    logger.info("Model loaded in %.1fs. VRAM: %.2f GB", time.time()-t0,
                torch.cuda.memory_allocated()/1e9)

    # Load data once
    data = load_arguana()
    logger.info("ArguAna: corpus=%d queries=%d qrels=%d",
                len(data["c_texts"]), len(data["q_texts"]), len(data["qrels"]))

    all_results = {}

    for method in ["prompteol", "mean"]:
        logger.info("=== Method: %s ===", method)

        # Encode corpus — get all 4 layers in one pass
        t1 = time.time()
        logger.info("Encoding corpus...")
        c_embs = encode_all(data["c_texts"], tokenizer, model, method, LAYERS,
                            MAX_LENGTH, BATCH_SIZE)
        logger.info("Corpus encoded in %.1fs", time.time()-t1)

        # Encode queries
        t2 = time.time()
        logger.info("Encoding queries...")
        q_embs = encode_all(data["q_texts"], tokenizer, model, method, LAYERS,
                            MAX_LENGTH, BATCH_SIZE)
        logger.info("Queries encoded in %.1fs", time.time()-t2)

        # Compute metrics for each layer
        for layer in LAYERS:
            logger.info("--- layer %d ---", layer)
            metrics = compute_metrics(q_embs[layer], c_embs[layer],
                          data["qrels"], data["q_ids"], data["c_ids"])
            run_id = f"{method}_layer{layer}"
            all_results[run_id] = {"ArguAna": metrics}

            logger.info("  nDCG@10=%.4f  Recall@10=%.4f  MRR@10=%.4f  MAP@10=%.4f",
                        metrics["ndcg@10"], metrics["recall@10"], metrics["mrr@10"],
                        metrics.get("map@10", 0))

            # Save per-run result
            result = {"model": MODEL_PATH, "method": method, "layer": layer,
                      "max_length": MAX_LENGTH, "datasets": {"ArguAna": metrics}}
            out_path = OUTPUT_DIR / f"{method}_layer{layer}_ArguAna.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

    # Save summary
    with open(OUTPUT_DIR / "arguana_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ARGUANA LAYER ABLATION RESULTS (nDCG@10)")
    print("=" * 60)
    print(f"{'layer':>8}  {'PromptEOL nDCG/Rec/MRR/MAP':>32}  {'mean-pooling nDCG/Rec/MRR/MAP':>32}")
    print("-" * 76)
    for layer in LAYERS:
        pe = all_results.get(f"prompteol_layer{layer}", {}).get("ArguAna", {})
        mp = all_results.get(f"mean_layer{layer}", {}).get("ArguAna", {})
        print(f"{layer:>8}  {pe.get('ndcg@10',0):.4f}/{pe.get('recall@10',0):.4f}/{pe.get('mrr@10',0):.4f}/{pe.get('map@10',0):.4f}"
              f"  {mp.get('ndcg@10',0):.4f}/{mp.get('recall@10',0):.4f}/{mp.get('mrr@10',0):.4f}/{mp.get('map@10',0):.4f}")


if __name__ == "__main__":
    main()
