#!/usr/bin/env python
"""Position Contribution Analysis — 分析 token 位置对最终表示的影响。

分析任务 3.2 的补充实验：对比 PromptEOL 和 mean-pooling 为何效果差异巨大。

分析内容：
  1. Token 位置贡献度分析 — 每个位置的 token 对最终嵌入的贡献
  2. 首/中/尾 token 的语义重要性分布
  3. PromptEOL 的信息瓶颈量化
  4. 不同层的位置贡献模式

Usage:
  python scripts/position_contribution_analysis.py \
    --model models/Mistral-7B-Instruct-v0.3 \
    --datasets QMSum 2WikiMultihop ArguAna \
    --output-dir results/position_analysis \
    --max-samples 50 --max-length 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pooling import last_token_pooling, mean_pooling  # noqa: E402
from src.prompts import build_prompteol  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Analysis 1: Per-Position Contribution to Mean-Pooled Embedding
# ============================================================================


def per_position_contribution(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    final_embedding: torch.Tensor,
    n_segments: int = 10,
) -> dict:
    """Measure how much each position segment contributes to the final embedding.

    Divides the sequence into n_segments and measures the cosine similarity
    between the mean-pooled embedding of each segment and the full embedding.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]
        final_embedding: [batch, hidden_dim] — the full mean-pooled embedding
        n_segments: number of position segments to divide into

    Returns:
        segment_contributions: per-segment similarity scores
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    segment_sims = [[] for _ in range(n_segments)]

    for b in range(batch):
        seq_len_b = int(attention_mask[b].sum().item())
        if seq_len_b < n_segments:
            continue

        h = hidden_states[b, :seq_len_b]  # [seq_len_b, hidden_dim]
        final = final_embedding[b]  # [hidden_dim]

        segment_size = seq_len_b / n_segments
        for seg in range(n_segments):
            start = int(seg * segment_size)
            end = int((seg + 1) * segment_size)
            segment_hidden = h[start:end].mean(dim=0)  # [hidden_dim]
            segment_hidden = F.normalize(segment_hidden, p=2, dim=0)
            final_norm = F.normalize(final, p=2, dim=0)
            sim = float((segment_hidden * final_norm).sum())
            if not np.isnan(sim):
                segment_sims[seg].append(sim)

    contributions = []
    for seg, sims in enumerate(segment_sims):
        contributions.append({
            "segment": seg,
            "position_percentile": f"{seg * 100 // n_segments}-{(seg + 1) * 100 // n_segments}%",
            "mean_cosine_sim": float(np.mean(sims)) if sims else None,
            "std_cosine_sim": float(np.std(sims)) if sims else None,
            "n_samples": len(sims),
        })

    # Identify which segments contribute most
    valid_contribs = [(c["mean_cosine_sim"], c["segment"], c["position_percentile"])
                      for c in contributions if c["mean_cosine_sim"] is not None]
    valid_contribs.sort(reverse=True)

    return {
        "segment_contributions": contributions,
        "top_segments": [
            {"segment": s, "percentile": p, "similarity": round(sim, 4)}
            for sim, s, p in valid_contribs[:3]
        ],
        "contribution_uniformity": _compute_uniformity(
            [c["mean_cosine_sim"] for c in contributions if c["mean_cosine_sim"] is not None]
        ),
    }


def _compute_uniformity(values: list[float]) -> dict:
    """Measure how uniform the contribution distribution is."""
    if not values or len(values) < 2:
        return {"entropy": None, "std": None}

    arr = np.array(values)
    arr = arr / (arr.sum() + 1e-12)  # normalize to sum=1
    entropy = -np.sum(arr * np.log(arr + 1e-12))
    max_entropy = np.log(len(arr))
    return {
        "normalized_entropy": round(float(entropy / max_entropy), 4),
        "std": round(float(np.std(values)), 6),
        "cv": round(float(np.std(values) / (np.mean(values) + 1e-12)), 4),
    }


# ============================================================================
# Analysis 2: PromptEOL Information Bottleneck Quantification
# ============================================================================


def information_bottleneck_analysis(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict:
    """Quantify the information loss in PromptEOL's single-token compression.

    Compares:
      - Variance of last-token representation vs mean-pooled representation
      - Cosine similarity between last token and all other tokens
      - Effective "coverage" of the last token representation
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    last_token_sims = []
    mean_sims = []

    for b in range(batch):
        seq_len_b = int(attention_mask[b].sum().item())
        if seq_len_b < 5:
            continue

        h = hidden_states[b, :seq_len_b]  # [seq_len_b, hidden_dim]
        last = h[-1]  # last token
        mean_vec = h.mean(dim=0)  # mean-pooled

        # Similarity of last token to all other tokens
        last_norm = F.normalize(last, p=2, dim=0)
        h_norm = F.normalize(h, p=2, dim=1)
        sims_to_last = (h_norm * last_norm.unsqueeze(0)).sum(dim=1)
        last_token_sims.append(float(sims_to_last.mean()))

        # Similarity of mean vector to all tokens
        mean_norm = F.normalize(mean_vec, p=2, dim=0)
        sims_to_mean = (h_norm * mean_norm.unsqueeze(0)).sum(dim=1)
        mean_sims.append(float(sims_to_mean.mean()))

    return {
        "last_token_avg_similarity_to_all_tokens": float(np.mean(last_token_sims)),
        "mean_pooled_avg_similarity_to_all_tokens": float(np.mean(mean_sims)),
        "coverage_gap": round(
            float(np.mean(mean_sims) - np.mean(last_token_sims)), 4
        ),
        "relative_improvement_pct": round(
            100 * (np.mean(mean_sims) - np.mean(last_token_sims)) / max(np.mean(last_token_sims), 1e-9), 1
        ),
    }


# ============================================================================
# Analysis 3: Layer-wise Position Sensitivity
# ============================================================================


def layer_position_sensitivity(
    all_layer_hidden: dict[int, np.ndarray],
    all_layer_masks: dict[int, np.ndarray],
    n_segments: int = 10,
) -> dict:
    """Analyze how position contribution patterns change across layers.

    Key hypothesis: Lower layers have more local patterns (high RoPE frequency),
    higher layers have more global patterns (low RoPE frequency dominant).
    """
    results = {}

    for layer, hidden_np in sorted(all_layer_hidden.items()):
        hidden = torch.from_numpy(hidden_np[:5]).float()  # first 5 samples
        mask = torch.from_numpy(all_layer_masks[layer][:5])

        # Compute mean-pooled embedding
        mean_emb = mean_pooling(hidden, mask)

        contrib = per_position_contribution(hidden, mask, mean_emb, n_segments)

        # Extract key metrics
        seg_sims = [
            c["mean_cosine_sim"] for c in contrib["segment_contributions"]
            if c["mean_cosine_sim"] is not None
        ]
        if seg_sims:
            first_half_sim = np.mean(seg_sims[: len(seg_sims) // 2])
            second_half_sim = np.mean(seg_sims[len(seg_sims) // 2:])
            results[f"layer_{layer}"] = {
                "first_half_contrib": round(float(first_half_sim), 4),
                "second_half_contrib": round(float(second_half_sim), 4),
                "recency_bias": round(float(second_half_sim - first_half_sim), 4),
                "uniformity": contrib["contribution_uniformity"],
                "top_segment": contrib["top_segments"][0] if contrib["top_segments"] else None,
            }

    return results


# ============================================================================
# Analysis 4: Token Count vs Representation Quality
# ============================================================================


def token_count_impact_analysis(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_embeddings: torch.Tensor | None = None,
) -> dict:
    """Analyze how the number of tokens affects PromptEOL vs mean-pooling quality.

    For texts of varying lengths, measure:
      - How much the last-token representation captures of the full text
      - Representation stability with increasing token count
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    length_bins = [(0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
    bin_results = []

    for lo, hi in length_bins:
        bin_last_sims = []
        bin_mean_sims = []

        for b in range(batch):
            seq_len_b = int(attention_mask[b].sum().item())
            if seq_len_b < lo or seq_len_b > hi:
                continue
            if seq_len_b < 10:
                continue

            h = hidden_states[b, :seq_len_b]
            last = h[-1]
            mean_vec = h.mean(dim=0)

            # Similarity between last-token and mean-pooled
            last_norm = F.normalize(last, p=2, dim=0)
            mean_norm = F.normalize(mean_vec, p=2, dim=0)
            sim = float((last_norm * mean_norm).sum())
            bin_last_sims.append(sim)

        if bin_last_sims:
            bin_results.append({
                "length_range": f"{lo}-{hi}",
                "mean_length": (lo + hi) / 2,
                "last_vs_mean_cosine_sim": round(float(np.mean(bin_last_sims)), 4),
                "std": round(float(np.std(bin_last_sims)), 4),
                "n_samples": len(bin_last_sims),
            })

    return {
        "length_bins": bin_results,
        "trend": "decreasing" if (
            bin_results and
            len(bin_results) >= 2 and
            bin_results[0].get("last_vs_mean_cosine_sim", 0) >
            bin_results[-1].get("last_vs_mean_cosine_sim", 0)
        ) else "stable_or_increasing",
    }


# ============================================================================
# Analysis 5: Embedding Dispersion Analysis
# ============================================================================


def embedding_dispersion_analysis(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    qrels: dict[str, set[str]],
    query_ids: list[str],
    corpus_ids: list[str],
) -> dict:
    """Analyze the dispersion and discriminability of embeddings.

    Compares PromptEOL and mean-pooling embeddings in terms of:
      - Average pairwise cosine similarity (lower = more discriminative)
      - Nearest-neighbor margin (relevant doc similarity - max irrelevant similarity)
    """
    # Normalize
    c_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-9)
    q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-9)

    # Average pairwise cosine similarity of corpus
    if c_norm.shape[0] <= 2000:
        c_sim_matrix = c_norm @ c_norm.T
        np.fill_diagonal(c_sim_matrix, 0)
        avg_pairwise_sim = float(c_sim_matrix.mean())
    else:
        # Sample for efficiency
        sample_idx = np.random.choice(c_norm.shape[0], min(2000, c_norm.shape[0]), replace=False)
        c_sample = c_norm[sample_idx]
        c_sim_sample = c_sample @ c_sample.T
        np.fill_diagonal(c_sim_sample, 0)
        avg_pairwise_sim = float(c_sim_sample.mean())

    # Nearest-neighbor margin
    scores = q_norm @ c_norm.T
    margins = []
    for qi, qid in enumerate(query_ids):
        if qid not in qrels or not qrels[qid]:
            continue
        relevant = qrels[qid]
        q_scores = scores[qi]
        # Top relevant score
        rel_indices = [i for i, cid in enumerate(corpus_ids) if cid in relevant]
        if not rel_indices:
            continue
        max_rel_score = float(q_scores[rel_indices].max())

        # Top irrelevant score
        sorted_idx = np.argsort(q_scores)[::-1]
        max_irrel_score = None
        for idx in sorted_idx[:100]:
            if corpus_ids[idx] not in relevant:
                max_irrel_score = float(q_scores[idx])
                break

        if max_irrel_score is not None:
            margins.append(max_rel_score - max_irrel_score)

    return {
        "avg_pairwise_corpus_cosine_sim": round(avg_pairwise_sim, 6),
        "mean_nn_margin": round(float(np.mean(margins)), 6) if margins else None,
        "std_nn_margin": round(float(np.std(margins)), 6) if margins else None,
        "negative_margin_pct": round(
            100 * sum(1 for m in margins if m < 0) / len(margins), 1
        ) if margins else None,
    }


# ============================================================================
# Main Runner
# ============================================================================


def run_position_analysis(
    model_path: str,
    datasets: list[str],
    output_dir: Path,
    max_samples: int = 50,
    max_length: int = 2048,
    layers: list[int] | None = None,
    batch_size: int = 8,
    load_in_4bit: bool = True,
) -> dict:
    """Run complete position contribution analysis."""
    if layers is None:
        layers = [8, 16, 24, 32]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    # ---- Load model ----
    logger.info("Loading model: %s", model_path)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs: dict = {
        "trust_remote_code": True,
        "output_hidden_states": True,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    logger.info("Model loaded. VRAM: %.2f GB", torch.cuda.memory_allocated() / 1e9)

    # Load data
    from datasets import load_dataset

    for ds_name in datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds_name)

        if ds_name.lower() == "arguana":
            corpus_ds = load_dataset("mteb/arguana", "corpus", split="corpus")
            sample_texts = []
            for doc in corpus_ds:
                title = doc.get("title", "") or ""
                text = doc.get("text", "") or ""
                full = f"{title}\n{text}".strip() if title else text
                sample_texts.append(full)
        else:
            name_map = {"QMSum": "qmsum", "2WikiMultihop": "2wikimqa"}
            ds_key = name_map.get(ds_name, ds_name.lower())
            corpus = load_dataset("dwzhu/LongEmbed", name=ds_key, split="corpus")
            sample_texts = [doc["text"] for doc in corpus]

        if len(sample_texts) > max_samples:
            step = len(sample_texts) // max_samples
            sample_texts = sample_texts[::step][:max_samples]

        ds_results: dict = {
            "per_position_contrib": {},
            "info_bottleneck": {},
            "layer_sensitivity": {},
            "token_count_impact": {},
        }

        for method in ["mean", "prompteol"]:
            logger.info("--- Method: %s ---", method)

            # Encode and collect hidden states
            all_hidden = {l: [] for l in layers}
            all_masks = {l: [] for l in layers}

            for start in tqdm(range(0, len(sample_texts), batch_size),
                              desc=f"encode[{method}]"):
                batch = sample_texts[start:start + batch_size]
                if method == "prompteol":
                    batch_inputs = [build_prompteol(t) for t in batch]
                else:
                    batch_inputs = list(batch)

                encoded = tokenizer(
                    batch_inputs, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                device = next(model.parameters()).device
                encoded = {k: v.to(device) for k, v in encoded.items()}

                with torch.inference_mode():
                    outputs = model(**encoded, output_hidden_states=True)

                for layer in layers:
                    hidden = outputs.hidden_states[layer]
                    all_hidden[layer].append(hidden.cpu().float().numpy())
                    all_masks[layer].append(
                        encoded["attention_mask"].cpu().numpy()
                    )

            # Concatenate
            combined_hidden = {}
            combined_masks = {}
            for layer in layers:
                combined_hidden[layer] = np.concatenate(all_hidden[layer], axis=0)
                combined_masks[layer] = np.concatenate(all_masks[layer], axis=0)

            # -- Analysis 1: Per-Position Contribution (last layer) --
            last_hidden = torch.from_numpy(combined_hidden[layers[-1]]).float()
            last_mask = torch.from_numpy(combined_masks[layers[-1]])

            if method == "mean":
                final_emb = mean_pooling(last_hidden, last_mask)
            else:
                final_emb = last_token_pooling(last_hidden, last_mask)

            pos_contrib = per_position_contribution(
                last_hidden, last_mask, final_emb, n_segments=10
            )
            ds_results["per_position_contrib"][method] = pos_contrib
            logger.info(
                "  Position top segment: %s (sim=%.4f), uniformity=%.4f",
                pos_contrib["top_segments"][0]["percentile"]
                if pos_contrib["top_segments"] else "N/A",
                pos_contrib["top_segments"][0]["similarity"]
                if pos_contrib["top_segments"] else 0,
                pos_contrib["contribution_uniformity"].get("normalized_entropy", 0),
            )

            # -- Analysis 2: Information Bottleneck --
            bottleneck = information_bottleneck_analysis(last_hidden, last_mask)
            ds_results["info_bottleneck"][method] = bottleneck
            logger.info(
                "  Info bottleneck: last_token_sim=%.3f, mean_sim=%.3f, gap=%.3f",
                bottleneck["last_token_avg_similarity_to_all_tokens"],
                bottleneck["mean_pooled_avg_similarity_to_all_tokens"],
                bottleneck["coverage_gap"],
            )

            # -- Analysis 3: Layer-wise Position Sensitivity --
            layer_sens = layer_position_sensitivity(
                {l: combined_hidden[l] for l in layers},
                {l: combined_masks[l] for l in layers},
                n_segments=10,
            )
            ds_results["layer_sensitivity"][method] = layer_sens
            for key, val in sorted(layer_sens.items()):
                logger.info(
                    "  %s: first_half=%.4f, second_half=%.4f, recency_bias=%.4f",
                    key, val["first_half_contrib"], val["second_half_contrib"],
                    val["recency_bias"],
                )

            # -- Analysis 4: Token Count Impact --
            tc_impact = token_count_impact_analysis(last_hidden, last_mask)
            ds_results["token_count_impact"][method] = tc_impact
            for lb in tc_impact.get("length_bins", []):
                logger.info(
                    "  length %s: last_vs_mean_sim=%.4f (n=%d)",
                    lb["length_range"], lb["last_vs_mean_cosine_sim"],
                    lb["n_samples"],
                )

            # Free GPU memory
            del all_hidden, all_masks, combined_hidden, combined_masks

        all_results[ds_name] = ds_results

        # Save per-dataset results
        ds_output = output_dir / f"{ds_name}_position_analysis.json"
        with open(ds_output, "w", encoding="utf-8") as f:
            json.dump(ds_results, f, indent=2, ensure_ascii=False)
        logger.info("Saved: %s", ds_output)

    # Cleanup
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save full results
    full_output = output_dir / "position_analysis_full.json"
    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    _print_position_summary(all_results)

    return all_results


def _print_position_summary(results: dict) -> None:
    """Print summary of position analysis."""
    print("\n" + "=" * 80)
    print("POSITION CONTRIBUTION ANALYSIS — SUMMARY")
    print("=" * 80)

    for ds_name, ds_results in results.items():
        print(f"\n### {ds_name} ###")

        # Position contribution
        pos = ds_results.get("per_position_contrib", {})
        for method, p in pos.items():
            if p.get("top_segments"):
                top = p["top_segments"][0]
                print(f"  [{method}] Top position: {top['percentile']} "
                      f"(sim={top['similarity']:.4f}), "
                      f"uniformity={p['contribution_uniformity'].get('normalized_entropy', '?')}")

        # Info bottleneck
        bn = ds_results.get("info_bottleneck", {})
        for method, b in bn.items():
            print(f"  [{method}] Last-token coverage: {b.get('last_token_avg_similarity_to_all_tokens', '?')}, "
                  f"Mean-pool coverage: {b.get('mean_pooled_avg_similarity_to_all_tokens', '?')}")

        # Layer sensitivity
        ls = ds_results.get("layer_sensitivity", {})
        for method, layers_sens in ls.items():
            print(f"  [{method}] Layer-wise recency bias:")
            for layer_key, val in sorted(layers_sens.items()):
                print(f"    {layer_key}: recency_bias={val.get('recency_bias', '?')}")


def main():
    parser = argparse.ArgumentParser(
        description="Position Contribution Analysis"
    )
    parser.add_argument("--model", default="models/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--datasets", nargs="+",
                        default=["QMSum", "2WikiMultihop", "ArguAna"])
    parser.add_argument("--output-dir", default="results/position_analysis")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 32])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    run_position_analysis(
        model_path=args.model,
        datasets=args.datasets,
        output_dir=output_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        layers=args.layers,
        batch_size=args.batch_size,
        load_in_4bit=not args.no_4bit,
    )

    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
