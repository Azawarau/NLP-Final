"""Pooling strategies over hidden states."""

from __future__ import annotations

import torch


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token hidden states, ignoring padding."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take the hidden state of the last non-padding token (PromptEOL)."""
    last_indices = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    batch_idx = torch.arange(batch_size, device=hidden_states.device)
    return hidden_states[batch_idx, last_indices]
