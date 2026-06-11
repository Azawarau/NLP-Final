"""MTEB-compatible encoder using Mistral (or fallback) with PromptEOL / mean-pooling."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pooling import last_token_pooling, mean_pooling
from src.prompts import build_prompteol

logger = logging.getLogger(__name__)

Method = Literal["prompteol", "mean"]
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
FALLBACK_MODEL = "Qwen/Qwen2-1.5B-Instruct"


class LLMEmbeddingEncoder:
    """Extract long-text embeddings from a causal LM without fine-tuning."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        method: Method = "mean",
        layer: int = -1,
        max_length: int = 8192,
        batch_size: int = 1,
        device: str | None = None,
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        normalize: bool = True,
        use_chat_template: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.method = method
        self.layer = layer
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self.use_chat_template = use_chat_template
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch_dtype)

        logger.info("Loading model %s on %s", model_name_or_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        load_kwargs: dict = {
            "trust_remote_code": trust_remote_code,
        }

        # Quantization config for limited VRAM (e.g. RTX 4060 8GB)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization (NF4, double-quant)")
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")

        if dtype != "auto":
            load_kwargs["torch_dtype"] = dtype
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        # *** CRITICAL: do NOT use output_hidden_states=True (OOM on 8GB VRAM) ***
        # We use a forward hook to capture only the target layer's output.
        load_kwargs["output_hidden_states"] = False

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        if device != "cuda" or "device_map" not in load_kwargs:
            self.model.to(device)
        self.model.eval()

        self.num_hidden_layers = self.model.config.num_hidden_layers
        self._resolved_layer = self._resolve_layer_index(layer)
        self._hook_output: torch.Tensor | None = None
        self._register_hook()

    def _resolve_layer_index(self, layer: int) -> int:
        """Map user layer index to hidden_states tuple index (0 = embed output)."""
        n = self.num_hidden_layers
        if layer == -1 or layer == n:
            return n
        if layer < 0:
            # -2 -> second-to-last transformer block, etc.
            return n + layer + 1
        if 1 <= layer <= n:
            return layer
        raise ValueError(f"layer must be in [1, {n}] or -1, got {layer}")

    def _format_inputs(self, texts: list[str]) -> list[str]:
        if self.method == "prompteol":
            return [build_prompteol(t) for t in texts]
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            formatted = []
            for t in texts:
                messages = [{"role": "user", "content": t}]
                formatted.append(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                )
            return formatted
        return texts

    def _register_hook(self) -> None:
        """Register forward hook to capture ONLY target layer hidden states.

        Saves ~1 GB VRAM vs output_hidden_states=True (avoids storing all 32 layers).
        """
        base_model = self.model.model
        target_idx = self._resolved_layer

        if target_idx == self.num_hidden_layers:
            target_module = base_model.norm
        elif 1 <= target_idx < self.num_hidden_layers:
            target_module = base_model.layers[target_idx]
        else:
            raise ValueError(f"Unsupported layer index: {target_idx}")

        self._hook_cache: list[torch.Tensor] = []

        def _hook_fn(module, input, output):
            t = output[0] if isinstance(output, tuple) else output
            self._hook_cache.append(t)

        target_module.register_forward_hook(_hook_fn)
        logger.debug("Hook on layer %d (%s)", target_idx, type(target_module).__name__)

    @torch.inference_mode()
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        inputs = self._format_inputs(texts)
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        self._hook_cache.clear()
        self.model(**encoded)  # output_hidden_states=False; hook captures target layer
        hidden = self._hook_cache[-1]

        if self.method == "prompteol":
            pooled = last_token_pooling(hidden, encoded["attention_mask"])
        else:
            pooled = mean_pooling(hidden, encoded["attention_mask"])

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.cpu().float().numpy()

    def encode(
        self,
        sentences: list[str],
        batch_size: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences; compatible with MTEB custom model interface."""
        del kwargs
        bs = batch_size or self.batch_size
        total = len(sentences)
        all_embeddings: list[np.ndarray] = []
        report_every = max(1, min(50, total // 10))  # report at most 10 times
        for start in tqdm(range(0, total, bs), desc=f"encode[{self.method}]"):
            batch = sentences[start : start + bs]
            all_embeddings.append(self._encode_batch(batch))
            end = min(start + bs, total)
            if end % (report_every * bs) < bs or end == total:
                logger.info("  encode[%s] %d/%d done", self.method, end, total)
        return np.vstack(all_embeddings)

    def encode_queries(self, queries: list[str], **kwargs) -> np.ndarray:
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: list[str] | list[dict[str, str]],
        **kwargs,
    ) -> np.ndarray:
        if corpus and isinstance(corpus[0], dict):
            texts = []
            for doc in corpus:
                title = doc.get("title", "") or ""
                text = doc.get("text", "") or ""
                texts.append(f"{title}\n{text}".strip() if title else text)
            return self.encode(texts, **kwargs)
        return self.encode(list(corpus), **kwargs)


def load_encoder(
    model_name: str | None = None,
    use_fallback: bool = False,
    **kwargs,
) -> LLMEmbeddingEncoder:
    name = FALLBACK_MODEL if use_fallback else (model_name or DEFAULT_MODEL)
    return LLMEmbeddingEncoder(model_name_or_path=name, **kwargs)


def estimate_model_vram(model_name_or_path: str) -> float | None:
    """Estimate VRAM needed for a model in GB (fp16), or None if unknown."""
    try:
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception:
        return None
    params = getattr(cfg, "num_params", None) or getattr(cfg, "num_parameters", None)
    if params is None:
        # rough estimate from hidden size and layers
        hidden = getattr(cfg, "hidden_size", 4096)
        layers = getattr(cfg, "num_hidden_layers", 32)
        vocab = getattr(cfg, "vocab_size", 32000)
        params = 2 * layers * hidden * hidden + vocab * hidden  # rough
    return 2 * params / 1e9  # fp16 = 2 bytes per param
