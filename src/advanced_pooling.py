"""Advanced pooling strategies for long-text representation.

Research Point 1: Keyword-Enhanced Weighted Pooling
- attention_weighted: Use attention scores from the last Transformer layer as token weights
- tfidf_weighted: Precompute TF-IDF weights and apply during pooling
- saliency_weighted: Gradient-based token importance scoring
- combined: Ensemble of the above methods
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

WeightMethod = Literal["attention", "tfidf", "saliency", "combined"]


# ---------------------------------------------------------------------------
# Attention-Score Weighted Pooling
# ---------------------------------------------------------------------------

def attention_weighted_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: torch.Tensor | None = None,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Pool hidden states using attention scores as importance weights.

    Core idea: In each Transformer layer, attention scores reflect which tokens
    the model considers important. By aggregating attention weights across heads
    and layers, we obtain a token-importance signal that is more semantically
    meaningful than uniform weights.

    Args:
        hidden_states: [batch, seq_len, hidden_dim] — hidden states from a layer
        attention_mask: [batch, seq_len]
        attention_weights: [batch, n_heads, seq_len, seq_len] — optional pre-extracted
                          attention weights. If None, uses a heuristic fallback.

    Returns:
        pooled: [batch, hidden_dim]
    """
    batch, seq_len, hidden_dim = hidden_states.shape
    mask = attention_mask.float()  # [batch, seq_len]

    if attention_weights is not None:
        # Aggregate attention: mean across heads, then sum of attention TO each token
        # attention_weights: [batch, n_heads, seq_len, seq_len]
        # For each token j, compute mean attention from all other tokens to j
        attn_to_token = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        # Average over source positions (how much attention does token j receive?)
        token_importance = attn_to_token.mean(dim=1)  # [batch, seq_len]
    else:
        # Heuristic: use L2 norm of hidden state as proxy for importance
        # This is a well-established saliency proxy when attention is unavailable
        token_importance = hidden_states.norm(dim=-1)  # [batch, seq_len]

    # Normalize importance scores per sample (softmax over valid tokens)
    token_importance = token_importance * mask
    token_importance = token_importance / (token_importance.sum(dim=1, keepdim=True) + 1e-12)

    # Weighted mean pooling
    weight_expanded = token_importance.unsqueeze(-1)  # [batch, seq_len, 1]
    pooled = (hidden_states * weight_expanded).sum(dim=1)

    return pooled


# ---------------------------------------------------------------------------
# TF-IDF Weighted Pooling
# ---------------------------------------------------------------------------

class TFIDFWeightCalculator:
    """Compute TF-IDF weights for token-level importance in pooling.

    Unlike standard TF-IDF (which operates at the document level), we compute
    token-level IDF by treating each document as a collection of tokens and
    measuring how "distinctive" each token is to its document.

    Weights are precomputed once per corpus and applied during pooling.
    """

    def __init__(self, tokenizer, corpus_texts: list[str] | None = None):
        self.tokenizer = tokenizer
        self.idf: dict[int, float] = {}
        if corpus_texts:
            self.fit(corpus_texts)

    def fit(self, corpus_texts: list[str], max_samples: int = 5000):
        """Compute token IDF from a sample of the corpus."""
        import math
        from collections import Counter

        texts = corpus_texts[:max_samples] if len(corpus_texts) > max_samples else corpus_texts
        n_docs = len(texts)
        doc_freq: Counter = Counter()

        logger.info("Computing TF-IDF weights from %d documents...", n_docs)
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                doc_freq[t] += 1

        self.idf = {
            token_id: math.log((n_docs + 1) / (df + 1)) + 1.0
            for token_id, df in doc_freq.items()
        }
        self.default_idf = 1.0  # for unseen tokens
        logger.info("TF-IDF: %d unique tokens indexed, avg IDF=%.3f",
                    len(self.idf), np.mean(list(self.idf.values())))

    def get_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get TF-IDF weights for a batch of tokenized inputs.

        Args:
            input_ids: [batch, seq_len] — token IDs
            attention_mask: [batch, seq_len]

        Returns:
            weights: [batch, seq_len] — normalized per-sample weights
        """
        batch, seq_len = input_ids.shape
        weights = torch.zeros(batch, seq_len, device=input_ids.device)

        for b in range(batch):
            for s in range(seq_len):
                if attention_mask[b, s] > 0:
                    tid = int(input_ids[b, s].item())
                    weights[b, s] = self.idf.get(tid, self.default_idf)

        # Normalize per sample
        weights = weights * attention_mask.float()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
        return weights


def tfidf_weighted_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    token_weights: torch.Tensor,
) -> torch.Tensor:
    """Pool hidden states with precomputed TF-IDF token weights.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]
        token_weights: [batch, seq_len] — normalized TF-IDF weights

    Returns:
        pooled: [batch, hidden_dim]
    """
    weight_expanded = token_weights.unsqueeze(-1).float()
    pooled = (hidden_states * weight_expanded).sum(dim=1)
    return pooled


# ---------------------------------------------------------------------------
# Saliency (Gradient-based) Weighted Pooling
# ---------------------------------------------------------------------------

def saliency_weighted_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    model: torch.nn.Module | None = None,
    input_embeds: torch.Tensor | None = None,
    norm_fn=None,
) -> torch.Tensor:
    """Pool using gradient-based token saliency scores.

    Saliency measures: ||∂(mean_pooled_norm)/∂(token_embedding)||₂
    Tokens with larger gradients contribute more to the final representation,
    hence they are more "important" for the semantic meaning.

    When model/input_embeds are not provided, falls back to L2-norm saliency.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]
        model: optional — the LLM (for gradient computation)
        input_embeds: optional — input embeddings for gradient computation
        norm_fn: optional — normalization function applied before gradient

    Returns:
        pooled: [batch, hidden_dim]
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    if model is not None and input_embeds is not None:
        # Gradient-based saliency
        input_embeds.requires_grad_(True)
        model.zero_grad()

        # Forward pass through the specific layer (simplified: use hidden_states directly)
        # Compute L2 norm of mean-pooled representation as scalar objective
        mask_expanded = attention_mask.unsqueeze(-1).float()
        mean_pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        objective = mean_pooled.norm(dim=-1).sum()  # scalar

        objective.backward(retain_graph=False)
        saliency = input_embeds.grad.norm(dim=-1)  # [batch, seq_len]
        input_embeds.requires_grad_(False)
        model.zero_grad()
    else:
        # Fallback: normalized L2 norm of hidden states
        # This is the zero-order approximation of saliency
        saliency = hidden_states.norm(dim=-1)

    # Apply mask and normalize
    saliency = saliency * attention_mask.float()
    saliency = saliency / (saliency.sum(dim=1, keepdim=True) + 1e-12)

    weight_expanded = saliency.unsqueeze(-1)
    pooled = (hidden_states * weight_expanded).sum(dim=1)

    return pooled


# ---------------------------------------------------------------------------
# Combined Weighted Pooling
# ---------------------------------------------------------------------------

def combined_weighted_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    attention_weights: torch.Tensor | None = None,
    tfidf_weights: torch.Tensor | None = None,
    alpha: float = 0.4,   # weight for attention-based
    beta: float = 0.35,    # weight for TF-IDF
    gamma: float = 0.25,   # weight for norm-based saliency
) -> torch.Tensor:
    """Combined weighted pooling: ensemble of attention, TF-IDF, and norm saliency.

    final_weight = alpha * attention_weight + beta * tfidf_weight + gamma * saliency_weight

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        attention_mask: [batch, seq_len]
        attention_weights: optional pre-extracted attention
        tfidf_weights: optional precomputed TF-IDF weights
        alpha, beta, gamma: mixing coefficients (should sum to ~1.0)

    Returns:
        pooled: [batch, hidden_dim]
    """
    batch, seq_len, hidden_dim = hidden_states.shape
    mask = attention_mask.float()

    # Component 1: Attention-based importance
    if attention_weights is not None:
        attn_to_token = attention_weights.mean(dim=1).mean(dim=1)  # [batch, seq_len]
    else:
        attn_to_token = hidden_states.norm(dim=-1)

    attn_weight = attn_to_token * mask
    attn_weight = attn_weight / (attn_weight.sum(dim=1, keepdim=True) + 1e-12)

    # Component 2: TF-IDF importance
    if tfidf_weights is not None:
        tfidf_weight = tfidf_weights.float()
    else:
        tfidf_weight = torch.ones_like(mask)
        tfidf_weight = tfidf_weight / (tfidf_weight.sum(dim=1, keepdim=True) + 1e-12)

    # Component 3: Norm-based saliency
    saliency = hidden_states.norm(dim=-1) * mask
    saliency = saliency / (saliency.sum(dim=1, keepdim=True) + 1e-12)

    # Ensemble
    combined_weight = (
        alpha * attn_weight +
        beta * tfidf_weight +
        gamma * saliency
    )

    # Re-normalize after combination
    combined_weight = combined_weight * mask
    combined_weight = combined_weight / (combined_weight.sum(dim=1, keepdim=True) + 1e-12)

    weight_expanded = combined_weight.unsqueeze(-1)
    pooled = (hidden_states * weight_expanded).sum(dim=1)

    return pooled


# ---------------------------------------------------------------------------
# Convenience: get all weighted pooling variants
# ---------------------------------------------------------------------------

def get_weighted_pooling(
    method: WeightMethod,
    tfidf_calculator: TFIDFWeightCalculator | None = None,
    alpha: float = 0.4,
    beta: float = 0.35,
    gamma: float = 0.25,
):
    """Factory returning a pooling function for the given method.

    Returns a callable with signature:
        (hidden_states: Tensor, attention_mask: Tensor,
         input_ids: Tensor | None = None, attention_weights: Tensor | None = None)
        -> Tensor
    """

    def attention_pool(hidden_states, attention_mask, input_ids=None, attention_weights=None):
        return attention_weighted_pooling(hidden_states, attention_mask, attention_weights)

    def tfidf_pool(hidden_states, attention_mask, input_ids=None, attention_weights=None):
        if tfidf_calculator is not None and input_ids is not None:
            weights = tfidf_calculator.get_weights(input_ids, attention_mask)
        else:
            # Fallback: uniform weights (equivalent to mean pooling)
            mask = attention_mask.float()
            weights = mask / (mask.sum(dim=1, keepdim=True) + 1e-12)
        return tfidf_weighted_pooling(hidden_states, attention_mask, weights)

    def saliency_pool(hidden_states, attention_mask, input_ids=None, attention_weights=None):
        return saliency_weighted_pooling(hidden_states, attention_mask)

    def combined_pool(hidden_states, attention_mask, input_ids=None, attention_weights=None):
        tfidf_w = None
        if tfidf_calculator is not None and input_ids is not None:
            tfidf_w = tfidf_calculator.get_weights(input_ids, attention_mask)
        return combined_weighted_pooling(
            hidden_states, attention_mask, attention_weights, tfidf_w,
            alpha=alpha, beta=beta, gamma=gamma,
        )

    mapping = {
        "attention": attention_pool,
        "tfidf": tfidf_pool,
        "saliency": saliency_pool,
        "combined": combined_pool,
    }
    return mapping[method]
