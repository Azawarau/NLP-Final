#!/usr/bin/env python
"""离线验证：不访问 HuggingFace，用随机初始化小模型测试编码管线。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_encoder import LLMEmbeddingEncoder  # noqa: E402
from src.mteb_wrapper import wrap_for_mteb  # noqa: E402
from src.pooling import last_token_pooling, mean_pooling  # noqa: E402
from src.prompts import build_prompteol  # noqa: E402


class _OfflineTokenizer:
    """最小 tokenizer，不访问网络。"""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        del truncation
        batch_ids = []
        for t in texts:
            ids = [(ord(c) % 900) + 1 for c in t[:max_length]]
            if not ids:
                ids = [1]
            batch_ids.append(ids)
        max_len = max(len(x) for x in batch_ids) if padding else max(len(x) for x in batch_ids)
        input_ids, attn = [], []
        for ids in batch_ids:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [0] * pad_len)
            attn.append([1] * len(ids) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def _build_offline_encoder(method: str, layer: int) -> LLMEmbeddingEncoder:
    cfg = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=4,
        n_head=4,
    )
    model = GPT2LMHeadModel(cfg)
    tok = _OfflineTokenizer()

    enc = LLMEmbeddingEncoder.__new__(LLMEmbeddingEncoder)
    enc.model_name_or_path = "offline-gpt2-tiny"
    enc.method = method
    enc.layer = layer
    enc.max_length = 128
    enc.batch_size = 2
    enc.normalize = True
    enc.use_chat_template = False
    enc.device = "cpu"
    enc.tokenizer = tok
    enc.model = model
    enc.model.eval()
    enc.num_hidden_layers = cfg.n_layer
    enc._resolved_layer = enc._resolve_layer_index(layer)
    return enc


def main() -> None:
    texts = ["Paris is the capital of France.", "Deep learning uses neural networks."]
    errors: list[str] = []

    # Prompt template
    p = build_prompteol(texts[0])
    if "one word" not in p:
        errors.append("PromptEOL 模板异常")

    # Pooling on synthetic tensors
    h = torch.randn(2, 5, 64)
    m = torch.ones(2, 5, dtype=torch.long)
    m[1, 3:] = 0
    mean_pooling(h, m)
    last_token_pooling(h, m)

    for method in ("mean", "prompteol"):
        for layer in (-1, 2):
            enc = _build_offline_encoder(method, layer)
            out = enc.encode(texts)
            if out.shape != (2, 64):
                errors.append(f"shape 错误: {method} layer={layer} -> {out.shape}")
            wrap = wrap_for_mteb(enc)
            if not hasattr(wrap, "mteb_model_meta"):
                errors.append("MTEB wrapper 缺少 mteb_model_meta")

    # MTEB 任务注册
    try:
        import mteb

        names = {t.metadata.name for t in mteb.get_tasks(
            tasks=["LEMBQMSumRetrieval", "LEMBWikimQARetrieval", "ArguAna"]
        )}
        expected = {"LEMBQMSumRetrieval", "LEMBWikimQARetrieval", "ArguAna"}
        if not expected.issubset(names):
            errors.append(f"MTEB 任务缺失: {expected - names}")
    except Exception as e:
        errors.append(f"MTEB 任务检查失败: {e}")

    print("========== 离线管线验证 ==========")
    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)

    print("  [PASS] PromptEOL / mean-pooling 编码")
    print("  [PASS] 多层抽取 (layer=-1, 2)")
    print("  [PASS] MTEB 2.x wrapper")
    print("  [PASS] 三数据集任务已注册")
    print("\n结论: 基础阶段 **代码与管线** 已完成。")
    print("待办: 联网下载 Mistral-7B 后运行 scripts/run_basic.py 产出正式数值。")
    sys.exit(0)


if __name__ == "__main__":
    main()
