#!/usr/bin/env python
"""RoPE Frequency Analysis for Long-Text Representation.

分析任务 3.2 的核心代码：探索 RoPE 位置编码中频率对长文本表示效果的影响。

分析内容：
  1. RoPE 理论频率谱分析 — 计算 Mistral-7B 的各维度对频率分布
  2. 隐状态频谱分析 — 对 hidden states 沿序列维度做 FFT，分析频谱能量分布
  3. 频段滤波实验 — 低通/高通/带通滤波后测量嵌入变化
  4. 位置-距离相似度衰减 — 测量 token 间相似度如何随距离衰减
  5. RoPE base theta 敏感性分析 — 理论分析不同 theta 对频率覆盖的影响

Usage:
  python scripts/rope_frequency_analysis.py \
    --model models/Mistral-7B-Instruct-v0.3 \
    --datasets QMSum 2WikiMultihop ArguAna \
    --output-dir results/rope_analysis \
    --max-samples 100 --max-length 2048
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
# Part 1: RoPE Theoretical Frequency Analysis
# ============================================================================


def compute_rope_frequencies(
    head_dim: int = 128,
    base: float = 1_000_000.0,
    max_seq_len: int = 32768,
) -> dict:
    """Compute RoPE frequency spectrum for Mistral-7B.

    RoPE applies rotation at frequencies θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1.
    The wavelength (period) for frequency i is 2π/θ_i.

    Returns dict with:
      - frequencies: list of (dim_pair_idx, theta_i, wavelength, band)
      - bands: {low, mid, high} with count of dimension pairs in each
      - base, head_dim, max_seq_len
    """
    d = head_dim
    dim_pairs = d // 2  # 64 pairs for Mistral-7B

    freqs = []
    for i in range(dim_pairs):
        theta = base ** (-2 * i / d)
        wavelength = 2 * math.pi / theta

        # Classify into frequency bands based on wavelength
        # Low frequency (long wavelength): captures long-range dependencies
        # High frequency (short wavelength): captures local position info
        if wavelength > max_seq_len:
            band = "low"  # wavelength exceeds any practical sequence
        elif wavelength > 1024:
            band = "mid"  # medium-range
        else:
            band = "high"  # local patterns

        freqs.append({
            "dim_pair": i,
            "theta": float(theta),
            "wavelength": float(wavelength),
            "band": band,
            "log10_wavelength": float(math.log10(max(wavelength, 1e-10))),
        })

    bands = defaultdict(int)
    for f in freqs:
        bands[f["band"]] += 1

    logger.info(
        "RoPE frequency spectrum: %d dim pairs, base=%.0f, "
        "low=%d, mid=%d, high=%d pairs",
        dim_pairs, base, bands["low"], bands["mid"], bands["high"],
    )

    return {
        "head_dim": head_dim,
        "base": base,
        "max_seq_len": max_seq_len,
        "dim_pairs": dim_pairs,
        "frequencies": freqs,
        "bands": dict(bands),
    }


def analyze_frequency_coverage(rope_spec: dict) -> dict:
    """Analyze what sequence lengths each frequency band can distinguish.

    Key insight: RoPE can distinguish positions only if the wavelength
    is less than the sequence length. For very long texts, low-frequency
    components are essential.
    """
    freqs = rope_spec["frequencies"]
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    coverage = {}
    for seq_len in seq_lengths:
        # Count dimension pairs that can distinguish positions at this length
        distinguishable = sum(
            1 for f in freqs if f["wavelength"] < seq_len * 2
        )
        # Pairs where wavelength > seq_len essentially encode the same
        # rotation for all tokens — they lose position info
        saturated = sum(
            1 for f in freqs if f["wavelength"] >= seq_len * 2
        )
        coverage[f"seq_{seq_len}"] = {
            "distinguishable_pairs": distinguishable,
            "saturated_pairs": saturated,
            "effective_resolution_pct": round(
                100 * distinguishable / len(freqs), 1
            ),
        }

    logger.info("Frequency coverage at seq=2048: %.1f%% pairs distinguishable",
                coverage["seq_2048"]["effective_resolution_pct"])
    logger.info("Frequency coverage at seq=8192: %.1f%% pairs distinguishable",
                coverage["seq_8192"]["effective_resolution_pct"])

    return coverage


# ============================================================================
# Part 2: Hidden State Spectral Analysis
# ============================================================================


def spectral_analysis(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray | None = None,
    n_fft: int | None = None,
) -> dict:
    """Perform FFT-based spectral analysis of hidden states along sequence dim.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len] — used to mask padding
        n_fft: FFT size (default: next power of 2 from seq_len)

    Returns:
        dict with spectral energy distribution across frequency bands
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    if n_fft is None:
        n_fft = 2 ** int(math.ceil(math.log2(seq_len)))

    # Apply FFT along sequence dimension for each hidden dim
    # Shape: [batch, n_fft, hidden_dim]
    if attention_mask is not None:
        # Zero out padding positions
        mask = attention_mask[:, :, None]  # [batch, seq_len, 1]
        hidden_states = hidden_states * mask

    # Pad to n_fft
    if seq_len < n_fft:
        pad = np.zeros((batch, n_fft - seq_len, hidden_dim), dtype=hidden_states.dtype)
        hidden_states = np.concatenate([hidden_states, pad], axis=1)

    # FFT along sequence dimension
    fft_result = np.fft.rfft(hidden_states, axis=1)  # [batch, n_fft//2+1, hidden_dim]
    magnitude = np.abs(fft_result)  # amplitude spectrum

    # Frequency bins
    n_freq_bins = fft_result.shape[1]
    nyquist = n_fft // 2
    freq_bins = np.fft.rfftfreq(n_fft)

    # Define frequency bands
    # Low: 0 to nyquist/8 (slow variation — long-range semantics)
    # Mid: nyquist/8 to nyquist/2
    # High: nyquist/2 to nyquist (fast variation — local syntax)
    low_cutoff = nyquist // 8
    mid_cutoff = nyquist // 2

    low_energy = []
    mid_energy = []
    high_energy = []

    for b in range(batch):
        mag = magnitude[b]  # [n_freq_bins, hidden_dim]
        total = np.sum(mag ** 2)

        # Aggregate energy in each band
        low_idx = min(low_cutoff, n_freq_bins)
        mid_idx = min(mid_cutoff, n_freq_bins)

        low_e = np.sum(mag[:low_idx] ** 2) / max(total, 1e-12)
        mid_e = np.sum(mag[low_idx:mid_idx] ** 2) / max(total, 1e-12)
        high_e = np.sum(mag[mid_idx:] ** 2) / max(total, 1e-12)

        low_energy.append(float(low_e))
        mid_energy.append(float(mid_e))
        high_energy.append(float(high_e))

    return {
        "low_freq_energy_ratio": float(np.mean(low_energy)),
        "mid_freq_energy_ratio": float(np.mean(mid_energy)),
        "high_freq_energy_ratio": float(np.mean(high_energy)),
        "low_freq_std": float(np.std(low_energy)),
        "mid_freq_std": float(np.std(mid_energy)),
        "high_freq_std": float(np.std(high_energy)),
        "per_sample": [
            {"low": l, "mid": m, "high": h}
            for l, m, h in zip(low_energy, mid_energy, high_energy)
        ],
        "n_fft": n_fft,
        "n_freq_bins": n_freq_bins,
        "freq_bin_boundaries": {
            "low_max": int(freq_bins[min(low_cutoff, n_freq_bins - 1)] * n_fft)
            if low_cutoff < n_freq_bins else int(freq_bins[-1] * n_fft),
            "mid_max": int(freq_bins[min(mid_cutoff, n_freq_bins - 1)] * n_fft)
            if mid_cutoff < n_freq_bins else int(freq_bins[-1] * n_fft),
        },
    }


# ============================================================================
# Part 3: Frequency Band Filtering Experiment
# ============================================================================


def apply_frequency_filter(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    filter_type: str,  # "lowpass", "highpass", "bandpass", "none"
    cutoff_low_ratio: float = 0.125,
    cutoff_high_ratio: float = 0.5,
) -> np.ndarray:
    """Apply frequency-domain filtering to hidden states.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]
        filter_type: type of filter to apply
        cutoff_low_ratio: lowpass cutoff as fraction of Nyquist
        cutoff_high_ratio: highpass cutoff as fraction of Nyquist

    Returns:
        filtered hidden states with same shape as input
    """
    batch, seq_len, hidden_dim = hidden_states.shape
    n_fft = 2 ** int(math.ceil(math.log2(seq_len)))

    # Mask padding
    mask = attention_mask[:, :, None].astype(hidden_states.dtype)
    masked = hidden_states * mask

    # Pad and FFT
    if seq_len < n_fft:
        pad = np.zeros((batch, n_fft - seq_len, hidden_dim), dtype=hidden_states.dtype)
        masked = np.concatenate([masked, pad], axis=1)

    fft_result = np.fft.rfft(masked, axis=1)
    n_freq_bins = fft_result.shape[1]

    # Create frequency mask
    freq_mask = np.ones((n_freq_bins, 1), dtype=hidden_states.dtype)
    low_bin = int(n_freq_bins * cutoff_low_ratio)
    high_bin = int(n_freq_bins * cutoff_high_ratio)

    if filter_type == "lowpass":
        freq_mask[low_bin:] = 0.0  # Keep only low frequencies
    elif filter_type == "highpass":
        freq_mask[:high_bin] = 0.0  # Keep only high frequencies
    elif filter_type == "bandpass":
        freq_mask[:low_bin] = 0.0
        freq_mask[high_bin:] = 0.0  # Keep only mid frequencies
    # "none": keep all

    # Apply filter
    filtered_fft = fft_result * freq_mask[np.newaxis, :, :]

    # IFFT
    filtered_seq = np.fft.irfft(filtered_fft, n=n_fft, axis=1)

    # Truncate back to original sequence length
    return filtered_seq[:, :seq_len, :]


def frequency_filtering_experiment(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    pooling_method: str = "mean",
    layers: list[int] | None = None,
) -> dict:
    """Measure impact of frequency-band filtering on embeddings.

    For each filter type, compute:
      - Cosine similarity between original and filtered embeddings
      - Change in embedding norm
      - Per-band contribution score
    """
    filter_types = ["lowpass", "bandpass", "highpass", "none"]
    results = {}

    for ft in filter_types:
        filtered_hidden = apply_frequency_filter(
            hidden_states, attention_mask, ft
        )
        filtered_tensor = torch.from_numpy(filtered_hidden).float()
        attn_tensor = torch.from_numpy(attention_mask)

        if pooling_method == "mean":
            original_pooled = mean_pooling(
                torch.from_numpy(hidden_states).float(), attn_tensor
            )
            filtered_pooled = mean_pooling(filtered_tensor, attn_tensor)
        else:
            original_pooled = last_token_pooling(
                torch.from_numpy(hidden_states).float(), attn_tensor
            )
            filtered_pooled = last_token_pooling(filtered_tensor, attn_tensor)

        # Normalize
        original_pooled = F.normalize(original_pooled, p=2, dim=1)
        filtered_pooled = F.normalize(filtered_pooled, p=2, dim=1)

        # Cosine similarity between original and filtered
        cosine_sim = (original_pooled * filtered_pooled).sum(dim=1)
        results[ft] = {
            "mean_cosine_sim": float(cosine_sim.mean()),
            "std_cosine_sim": float(cosine_sim.std()),
            "cosine_sim_list": cosine_sim.tolist(),
        }

    # Compute band contribution: how much does removing a band change the embedding?
    # Higher change (lower cosine sim) → more important band
    for band_name, filter_name in [
        ("high_contribution", "lowpass"),   # lowpass keeps low → high band is removed
        ("low_contribution", "highpass"),   # highpass keeps high → low band is removed
        ("mid_contribution", "bandpass"),   # bandpass keeps mid → low+high removed
    ]:
        sim = results[filter_name]["mean_cosine_sim"]
        results[band_name] = {
            "embedding_change": round(1.0 - sim, 6),
            "importance_score": round(1.0 - sim, 6),  # higher = more important
        }

    return results


# ============================================================================
# Part 4: Position-Distance Similarity Analysis
# ============================================================================


def position_distance_analysis(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    max_distance: int = 512,
    num_bins: int = 20,
) -> dict:
    """Measure how token similarity decays with position distance.

    This directly relates to RoPE: RoPE's frequency-dependent encoding means
    that the positional similarity between tokens decays with distance,
    and the decay rate depends on the frequency composition.

    Returns:
        dict with distance bins and average cosine similarity per bin
    """
    batch, seq_len, hidden_dim = hidden_states.shape
    tensor_hidden = torch.from_numpy(hidden_states).float()

    # Distance bins (log-spaced for better resolution at short distances)
    bins = np.unique(np.logspace(
        0, math.log10(max_distance + 1), num_bins
    ).astype(int))
    bin_centers = []
    similarities = []

    for i in range(len(bins) - 1):
        d_start = bins[i]
        d_end = bins[i + 1]
        bin_center = (d_start + d_end) / 2
        bin_centers.append(float(bin_center))

        sims = []
        for b in range(batch):
            seq_len_b = int(attention_mask[b].sum())
            if seq_len_b <= d_end:
                continue
            # Sample pairs at this distance
            n_pairs = min(50, seq_len_b - d_end)
            for offset in range(0, n_pairs, max(1, n_pairs // 10)):
                j = offset + d_start + (d_end - d_start) // 2
                i = offset
                if j < seq_len_b and i < seq_len_b:
                    a = tensor_hidden[b, i]  # [hidden_dim]
                    b_vec = tensor_hidden[b, j]
                    cos_sim = float(
                        F.cosine_similarity(a.unsqueeze(0), b_vec.unsqueeze(0))
                    )
                    if not np.isnan(cos_sim):
                        sims.append(cos_sim)

        similarities.append({
            "distance_bin_start": int(d_start),
            "distance_bin_end": int(d_end),
            "distance_center": float(bin_center),
            "mean_cosine_similarity": float(np.mean(sims)) if sims else None,
            "std_cosine_similarity": float(np.std(sims)) if sims else None,
            "n_pairs": len(sims),
        })

    return {
        "distance_bins": similarities,
        "summary": {
            "short_range_sim": float(np.mean([
                s["mean_cosine_similarity"]
                for s in similarities[:5]
                if s["mean_cosine_similarity"] is not None
            ])),
            "long_range_sim": float(np.mean([
                s["mean_cosine_similarity"]
                for s in similarities[-5:]
                if s["mean_cosine_similarity"] is not None
            ])),
        },
    }


# ============================================================================
# Part 5: RoPE Theta Sensitivity (Theoretical)
# ============================================================================


def rope_theta_sensitivity_analysis() -> dict:
    """Theoretical analysis of how RoPE base theta affects frequency coverage.

    Compare different theta values:
      - theta=10000 (original RoPE)
      - theta=100000
      - theta=1000000 (Mistral-7B default)
      - theta=10000000 (extended)
    """
    thetas = [10000, 100000, 500000, 1_000_000, 10_000_000]
    head_dim = 128
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    results = {}
    for theta in thetas:
        spec = compute_rope_frequencies(
            head_dim=head_dim, base=theta, max_seq_len=32768
        )
        coverage = analyze_frequency_coverage(spec)

        # For each seq length, what fraction of dim pairs can distinguish positions
        pct = {}
        for seq_len in seq_lengths:
            key = f"seq_{seq_len}"
            pct[f"seq_{seq_len}"] = coverage[key]["effective_resolution_pct"]

        results[f"theta_{theta}"] = {
            "theta": theta,
            "band_distribution": spec["bands"],
            "coverage_pct": pct,
        }

        logger.info(
            "theta=%d: low=%d mid=%d high=%d pairs, "
            "coverage@2k=%.1f%%, coverage@8k=%.1f%%",
            theta,
            spec["bands"]["low"],
            spec["bands"]["mid"],
            spec["bands"]["high"],
            pct["seq_2048"],
            pct["seq_8192"],
        )

    return results


# ============================================================================
# Main Analysis Runner
# ============================================================================


def run_rope_analysis(
    model_path: str,
    datasets: list[str],
    output_dir: Path,
    max_samples: int = 50,
    max_length: int = 2048,
    layers: list[int] | None = None,
    batch_size: int = 8,
    load_in_4bit: bool = True,
) -> dict:
    """Run complete RoPE frequency analysis pipeline."""
    if layers is None:
        layers = [8, 16, 24, 32]

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict = {
        "rope_theory": {},
        "spectral": {},
        "frequency_filtering": {},
        "position_distance": {},
        "theta_sensitivity": {},
    }

    # ---- Part 1: Theoretical RoPE Analysis (no model needed) ----
    logger.info("=" * 60)
    logger.info("PART 1: RoPE Theoretical Frequency Analysis")
    logger.info("=" * 60)

    rope_spec = compute_rope_frequencies(
        head_dim=128, base=1_000_000.0, max_seq_len=32768
    )
    coverage = analyze_frequency_coverage(rope_spec)
    all_results["rope_theory"] = {
        "spec": rope_spec,
        "coverage": coverage,
    }

    # ---- Part 5: Theta Sensitivity (no model needed) ----
    logger.info("=" * 60)
    logger.info("PART 5: RoPE Theta Sensitivity Analysis")
    logger.info("=" * 60)

    theta_results = rope_theta_sensitivity_analysis()
    all_results["theta_sensitivity"] = theta_results

    # ---- Load model for empirical analyses ----
    logger.info("=" * 60)
    logger.info("Loading model for empirical analyses")
    logger.info("=" * 60)

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
    num_layers = model.config.num_hidden_layers
    logger.info("Model loaded. %d layers, VRAM: %.2f GB",
                num_layers, torch.cuda.memory_allocated() / 1e9)

    # Load sample texts from datasets
    from datasets import load_dataset

    for ds_name in datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds_name)
        logger.info("=" * 60)

        # Load data
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

        # Limit samples
        if len(sample_texts) > max_samples:
            # Stratified sampling: take evenly from the dataset
            step = len(sample_texts) // max_samples
            sample_texts = sample_texts[::step][:max_samples]
        logger.info("Sampled %d texts for analysis", len(sample_texts))

        ds_spectral = {f"layer_{l}": None for l in layers}
        ds_filtering = {f"layer_{l}": None for l in layers}
        ds_distance = {f"layer_{l}": None for l in layers}

        # Process each method (use mean-pooling for spectral analysis)
        for method in ["mean", "prompteol"]:
            logger.info("--- Method: %s ---", method)

            # ---- Part 2: Spectral Analysis ----
            logger.info("Part 2: Hidden State Spectral Analysis")
            all_hidden = {l: [] for l in layers}
            all_masks = {l: [] for l in layers}

            for start in tqdm(range(0, len(sample_texts), batch_size),
                              desc=f"encode_spectral[{method}]"):
                batch = sample_texts[start:start + batch_size]
                if method == "prompteol":
                    batch_inputs = [build_prompteol(t) for t in batch]
                else:
                    batch_inputs = list(batch)

                encoded = tokenizer(
                    batch_inputs,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
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

            # Concatenate and run spectral analysis
            for layer in layers:
                combined_hidden = np.concatenate(all_hidden[layer], axis=0)
                combined_mask = np.concatenate(all_masks[layer], axis=0)

                spec_result = spectral_analysis(combined_hidden, combined_mask)
                key = f"{method}_layer{layer}"
                ds_spectral[key] = spec_result

                logger.info(
                    "  %s: low=%.3f mid=%.3f high=%.3f",
                    key,
                    spec_result["low_freq_energy_ratio"],
                    spec_result["mid_freq_energy_ratio"],
                    spec_result["high_freq_energy_ratio"],
                )

            # ---- Part 3: Frequency Filtering ----
            logger.info("Part 3: Frequency Band Filtering Experiment")
            # Use a small subset for filtering experiment
            filter_hidden = np.concatenate(all_hidden[layers[-1]], axis=0)[:10]
            filter_mask = np.concatenate(all_masks[layers[-1]], axis=0)[:10]

            filter_result = frequency_filtering_experiment(
                filter_hidden, filter_mask, method, layers
            )
            ds_filtering[f"{method}_layer{layers[-1]}"] = filter_result

            for ft, ft_result in filter_result.items():
                if "mean_cosine_sim" in ft_result:
                    logger.info(
                        "  Filter=%s: cosine_sim=%.4f",
                        ft, ft_result["mean_cosine_sim"],
                    )

            # ---- Part 4: Position-Distance Analysis ----
            logger.info("Part 4: Position-Distance Similarity")
            dist_result = position_distance_analysis(
                combined_hidden[:5], combined_mask[:5], max_distance=512
            )
            key = f"{method}_layer{layers[-1]}"
            ds_distance[key] = dist_result

            if dist_result["distance_bins"]:
                short = dist_result["summary"]["short_range_sim"]
                long_r = dist_result["summary"]["long_range_sim"]
                logger.info(
                    "  %s: short_range_sim=%.4f, long_range_sim=%.4f, "
                    "decay_ratio=%.3f",
                    key, short, long_r,
                    (short - long_r) / max(short, 1e-9) if short > 0 else 0,
                )

            # Free intermediate data
            del all_hidden, all_masks
            all_hidden = {l: [] for l in layers}
            all_masks = {l: [] for l in layers}

        all_results["spectral"][ds_name] = ds_spectral
        all_results["frequency_filtering"][ds_name] = ds_filtering
        all_results["position_distance"][ds_name] = ds_distance

        # Save intermediate results
        ds_output = output_dir / f"{ds_name}_rope_analysis.json"
        ds_data = {
            "spectral": ds_spectral,
            "frequency_filtering": ds_filtering,
            "position_distance": ds_distance,
        }
        with open(ds_output, "w", encoding="utf-8") as f:
            json.dump(ds_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved: %s", ds_output)

    # Cleanup
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save full results
    full_output = output_dir / "rope_analysis_full.json"
    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Full results saved: %s", full_output)

    # ---- Print Summary ----
    _print_summary(all_results)

    return all_results


def _print_summary(results: dict) -> None:
    """Print a human-readable summary of all RoPE analysis results."""
    print("\n" + "=" * 80)
    print("ROPE FREQUENCY ANALYSIS — COMPLETE SUMMARY")
    print("=" * 80)

    # RoPE Theory
    theory = results.get("rope_theory", {})
    spec = theory.get("spec", {})
    if spec:
        print("\n## 1. RoPE Frequency Spectrum")
        print(f"   Head dim: {spec.get('head_dim')}, Base theta: {spec.get('base')}")
        bands = spec.get("bands", {})
        print(f"   Band distribution: low={bands.get('low')}, mid={bands.get('mid')}, "
              f"high={bands.get('high')} (out of {spec.get('dim_pairs')} dim pairs)")

    coverage = theory.get("coverage", {})
    if coverage:
        print("\n   Effective position resolution:")
        for key in sorted(coverage.keys()):
            c = coverage[key]
            print(f"     {key}: {c['effective_resolution_pct']}% pairs distinguishable "
                  f"({c['distinguishable_pairs']}/{c['distinguishable_pairs'] + c['saturated_pairs']})")

    # Theta sensitivity
    theta = results.get("theta_sensitivity", {})
    if theta:
        print("\n## 5. RoPE Theta Sensitivity")
        for key, val in sorted(theta.items()):
            cov = val.get("coverage_pct", {})
            print(f"   {key}: 2k={cov.get('seq_2048', '?')}%, "
                  f"8k={cov.get('seq_8192', '?')}%, "
                  f"bands={val.get('band_distribution', {})}")

    # Spectral analysis
    spectral = results.get("spectral", {})
    if spectral:
        print("\n## 2. Hidden State Spectral Energy Distribution")
        for ds_name, ds_results in spectral.items():
            print(f"\n   ### {ds_name} ###")
            for key in sorted(ds_results.keys()):
                spec_r = ds_results[key]
                if spec_r:
                    print(f"     {key}: low={spec_r['low_freq_energy_ratio']:.3f} "
                          f"mid={spec_r['mid_freq_energy_ratio']:.3f} "
                          f"high={spec_r['high_freq_energy_ratio']:.3f}")

    # Frequency filtering
    filtering = results.get("frequency_filtering", {})
    if filtering:
        print("\n## 3. Frequency Band Filtering Impact")
        for ds_name, ds_results in filtering.items():
            print(f"\n   ### {ds_name} ###")
            for key, f_results in ds_results.items():
                if isinstance(f_results, dict):
                    print(f"     {key}:")
                    for ft in ["lowpass", "bandpass", "highpass"]:
                        if ft in f_results:
                            sim = f_results[ft].get("mean_cosine_sim", 0)
                            print(f"       {ft}: similarity_to_original={sim:.4f} "
                                  f"(change={1-sim:.4f})")

    # Position distance
    dist = results.get("position_distance", {})
    if dist:
        print("\n## 4. Position-Distance Similarity Decay")
        for ds_name, ds_results in dist.items():
            print(f"\n   ### {ds_name} ###")
            for key, d_results in ds_results.items():
                if isinstance(d_results, dict) and "summary" in d_results:
                    s = d_results["summary"]
                    print(f"     {key}: short_range={s['short_range_sim']:.4f}, "
                          f"long_range={s['long_range_sim']:.4f}")


# ============================================================================
# Visualization helpers
# ============================================================================


def plot_rope_frequencies(
    rope_spec: dict, output_path: Path | str
) -> None:
    """Plot RoPE frequency spectrum."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    freqs = rope_spec["frequencies"]
    dim_pairs = [f["dim_pair"] for f in freqs]
    wavelengths = [f["wavelength"] for f in freqs]
    bands = [f["band"] for f in freqs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Wavelength vs dimension pair
    colors = {"low": "#2196F3", "mid": "#4CAF50", "high": "#FF9800"}
    bar_colors = [colors[b] for b in bands]

    ax1.bar(dim_pairs, wavelengths, color=bar_colors, width=1.0, alpha=0.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("Dimension Pair Index")
    ax1.set_ylabel("Wavelength (tokens, log scale)")
    ax1.set_title(f"RoPE Frequency Spectrum (θ={rope_spec['base']:.0f})")
    ax1.axhline(y=8192, color="red", linestyle="--", alpha=0.5, label="8k seq")
    ax1.axhline(y=2048, color="orange", linestyle="--", alpha=0.5, label="2k seq")
    ax1.legend()

    # Plot 2: Band distribution pie chart
    band_counts = rope_spec["bands"]
    labels = [f"Low Freq\n(long-range)\n{band_counts['low']} pairs",
              f"Mid Freq\n{band_counts['mid']} pairs",
              f"High Freq\n(local)\n{band_counts['high']} pairs"]
    sizes = [band_counts["low"], band_counts["mid"], band_counts["high"]]
    pie_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    ax2.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%",
            startangle=90)
    ax2.set_title("Frequency Band Distribution")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot: %s", output_path)


def plot_spectral_energy(
    spectral_results: dict, output_path: Path | str
) -> None:
    """Plot spectral energy distribution across layers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    datasets = list(spectral_results.keys())
    n_ds = len(datasets)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        ds_data = spectral_results[ds_name]
        # Extract per-layer energy ratios for mean pooling
        layers = []
        low_vals, mid_vals, high_vals = [], [], []
        for key, val in sorted(ds_data.items()):
            if val and "mean" in key and "low_freq_energy_ratio" in val:
                layer_num = int(key.split("layer")[1])
                layers.append(layer_num)
                low_vals.append(val["low_freq_energy_ratio"])
                mid_vals.append(val["mid_freq_energy_ratio"])
                high_vals.append(val["high_freq_energy_ratio"])

        if layers:
            x = np.arange(len(layers))
            width = 0.25
            ax.bar(x - width, low_vals, width, label="Low Freq", color="#2196F3")
            ax.bar(x, mid_vals, width, label="Mid Freq", color="#4CAF50")
            ax.bar(x + width, high_vals, width, label="High Freq", color="#FF9800")
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{l}" for l in layers])
            ax.set_ylabel("Energy Ratio")
            ax.set_title(f"{ds_name} — Spectral Energy by Layer")
            ax.legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot: %s", output_path)


def plot_position_similarity(
    distance_results: dict, output_path: Path | str
) -> None:
    """Plot token similarity vs position distance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    datasets = list(distance_results.keys())
    n_ds = len(datasets)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        ds_data = distance_results[ds_name]
        for key, d_results in ds_data.items():
            if isinstance(d_results, dict) and "distance_bins" in d_results:
                bins = d_results["distance_bins"]
                distances = [b["distance_center"] for b in bins]
                sims = [b["mean_cosine_similarity"] for b in bins]
                # Filter None values
                valid = [(d, s) for d, s in zip(distances, sims) if s is not None]
                if valid:
                    d, s = zip(*valid)
                    method = "PromptEOL" if "prompteol" in key else "mean-pooling"
                    ax.plot(d, s, "o-", label=key, markersize=4, linewidth=1.5)

        ax.set_xlabel("Token Distance")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_title(f"{ds_name} — Position Similarity Decay")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot: %s", output_path)


def generate_all_plots(results: dict, output_dir: Path) -> None:
    """Generate all analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    theory = results.get("rope_theory", {})
    spec = theory.get("spec", {})
    if spec:
        plot_rope_frequencies(spec, output_dir / "rope_spectrum.png")

    spectral = results.get("spectral", {})
    if spectral:
        plot_spectral_energy(spectral, output_dir / "spectral_energy.png")

    dist = results.get("position_distance", {})
    if dist:
        plot_position_similarity(dist, output_dir / "position_similarity.png")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RoPE Frequency Analysis for Long-Text Representation"
    )
    parser.add_argument(
        "--model", default="models/Mistral-7B-Instruct-v0.3",
        help="Path to Mistral-7B model"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["QMSum", "2WikiMultihop", "ArguAna"],
    )
    parser.add_argument(
        "--output-dir", default="results/rope_analysis",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50,
        help="Max texts per dataset for empirical analysis"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[8, 16, 24, 32],
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
    )
    parser.add_argument(
        "--theory-only", action="store_true",
        help="Run only theoretical analysis (no model needed)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.theory_only:
        logger.info("Running theory-only analysis (no GPU needed)")
        results = {
            "rope_theory": {},
            "theta_sensitivity": {},
        }
        rope_spec = compute_rope_frequencies(128, 1_000_000, 32768)
        coverage = analyze_frequency_coverage(rope_spec)
        results["rope_theory"] = {"spec": rope_spec, "coverage": coverage}
        results["theta_sensitivity"] = rope_theta_sensitivity_analysis()

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "rope_theory_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        _print_summary(results)

        try:
            generate_all_plots(results, output_dir)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)
    else:
        results = run_rope_analysis(
            model_path=args.model,
            datasets=args.datasets,
            output_dir=output_dir,
            max_samples=args.max_samples,
            max_length=args.max_length,
            layers=args.layers,
            batch_size=args.batch_size,
            load_in_4bit=not args.no_4bit,
        )

        try:
            generate_all_plots(results, output_dir)
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    print("\nDone! Results saved to:", output_dir)


if __name__ == "__main__":
    main()
