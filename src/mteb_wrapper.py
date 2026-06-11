"""MTEB 2.x AbsEncoder wrapper for LLMEmbeddingEncoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta

from src.llm_encoder import LLMEmbeddingEncoder

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class LLMEmbeddingMTEBWrapper(AbsEncoder):
    """Bridge LLMEmbeddingEncoder to mteb.evaluate (mteb >= 2)."""

    def __init__(self, encoder: LLMEmbeddingEncoder) -> None:
        self.encoder = encoder
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": encoder.model_name_or_path,
                "revision": "local",
                "embed_dim": encoder.model.config.hidden_size,
                "languages": ["eng-Latn"],
                "open_weights": True,
                "framework": ["PyTorch"],
            }
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        del task_metadata, hf_split, hf_subset, prompt_type
        texts = [text for batch in inputs for text in batch["text"]]
        batch_size = int(kwargs.get("batch_size", self.encoder.batch_size))
        return self.encoder.encode(texts, batch_size=batch_size)


def wrap_for_mteb(encoder: LLMEmbeddingEncoder) -> LLMEmbeddingMTEBWrapper:
    return LLMEmbeddingMTEBWrapper(encoder)
