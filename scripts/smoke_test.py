#!/usr/bin/env python
"""快速自检：不跑完整 MTEB，仅验证编码与池化逻辑。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pooling import last_token_pooling, mean_pooling  # noqa: E402
from src.prompts import build_prompteol  # noqa: E402


def test_pooling() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]], [[3.0, 0.0], [4.0, 0.0], [0.0, 0.0]]]
    )
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    mean = mean_pooling(hidden, mask)
    last = last_token_pooling(hidden, mask)
    assert torch.allclose(mean[0], torch.tensor([4.0, 0.0]))  # (1+2+9)/3
    assert torch.allclose(mean[1], torch.tensor([3.5, 0.0]))
    assert torch.allclose(last[0], torch.tensor([9.0, 0.0]))
    assert torch.allclose(last[1], torch.tensor([4.0, 0.0]))
    print("pooling: OK")


def test_prompt() -> None:
    p = build_prompteol("Hello world")
    assert "Hello world" in p
    assert "one word" in p
    print("prompt: OK")


def test_encoder_optional(model: str, fallback: bool) -> None:
    try:
        from src.llm_encoder import load_encoder
    except ImportError as e:
        print(f"encoder skip (import): {e}")
        return

    name = model
    if fallback:
        name = "Qwen/Qwen2-1.5B-Instruct"
    print(f"Loading {name} ...")
    enc = load_encoder(
        model_name=name,
        use_fallback=fallback,
        method="mean",
        layer=-1,
        max_length=512,
        batch_size=1,
    )
    vec = enc.encode(["This is a short test sentence."])
    assert vec.shape[0] == 1
    print(f"encoder: OK, dim={vec.shape[1]}")


if __name__ == "__main__":
    test_pooling()
    test_prompt()
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--with-model", action="store_true")
    ap.add_argument("--fallback", action="store_true")
    a = ap.parse_args()
    if a.with_model:
        test_encoder_optional(a.model, a.fallback)
    else:
        print("Skip model load (use --with-model to test HF model)")
