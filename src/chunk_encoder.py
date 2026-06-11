"""Chunk-based aggregation encoder for long-text representation.

Research Point 2: Divide-and-Conquer Chunking
- Split long texts into overlapping semantic chunks
- Encode each chunk independently through the LLM
- Aggregate chunk embeddings via mean, weighted, or attention-based pooling

Rationale:
RoPE's long-distance position decay means tokens at the beginning of a long document
have attenuated influence on tokens at the end. By splitting into chunks, each chunk
is encoded within a shorter context window where RoPE frequency resolution is higher
(see analysis report §2.1.3: 48.4% effective resolution at 2K vs 57.8% at 8K).

The chunk encoder is a WRAPPER around an existing LLMEmbeddingEncoder — it reuses
the encoder for per-chunk forward passes and adds aggregation logic.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

AggregationMethod = Literal["mean", "first", "weighted"]


class ChunkEncoder:
    """Encode long texts by chunking, encoding independently, then aggregating.

    This wraps an existing LLMEmbeddingEncoder. For each text:
    1. Tokenize and split into chunks of `chunk_size` tokens
    2. Encode each chunk independently through the LLM
    3. Aggregate chunk embeddings into a single text embedding
    """

    def __init__(
        self,
        base_encoder,  # LLMEmbeddingEncoder instance
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        aggregation: AggregationMethod = "mean",
        max_chunks: int = 32,
    ):
        """
        Args:
            base_encoder: An existing LLMEmbeddingEncoder (already loaded)
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between consecutive chunks
            aggregation: How to combine chunk embeddings
            max_chunks: Maximum number of chunks per text (truncation safety)
        """
        self.encoder = base_encoder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.aggregation = aggregation
        self.max_chunks = max_chunks
        self.tokenizer = base_encoder.tokenizer

        # Validate
        assert chunk_overlap < chunk_size, "overlap must be < chunk_size"
        logger.info(
            "ChunkEncoder: chunk_size=%d, overlap=%d, aggregation=%s, max_chunks=%d",
            chunk_size, chunk_overlap, aggregation, max_chunks,
        )

    def _chunk_text(self, text: str) -> list[list[int]]:
        """Split a single text into overlapping token chunks.

        Returns list of token ID lists (each chunk ready for the tokenizer).
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.chunk_size:
            return [tokens]

        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        for start in range(0, len(tokens), stride):
            chunk_tokens = tokens[start:start + self.chunk_size]
            if len(chunk_tokens) < max(16, self.chunk_size // 4):
                # Skip very short trailing chunks
                continue
            chunks.append(chunk_tokens)
            if len(chunks) >= self.max_chunks:
                break

        if not chunks:
            chunks = [tokens[:self.chunk_size]]

        return chunks

    def _token_ids_to_text(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text (for re-encoding by the base encoder)."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _aggregate_chunk_embeddings(
        self,
        chunk_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate multiple chunk embeddings into one.

        Args:
            chunk_embeddings: [n_chunks, hidden_dim]

        Returns:
            aggregated: [hidden_dim]
        """
        if chunk_embeddings.shape[0] == 1:
            return chunk_embeddings[0]

        if self.aggregation == "mean":
            return chunk_embeddings.mean(dim=0)
        elif self.aggregation == "first":
            return chunk_embeddings[0]
        elif self.aggregation == "weighted":
            # Weight chunks by their L2 norm (more "informative" chunks get higher weight)
            norms = chunk_embeddings.norm(dim=-1)  # [n_chunks]
            weights = F.softmax(norms, dim=0)
            return (chunk_embeddings * weights.unsqueeze(-1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    @torch.inference_mode()
    def encode(
        self,
        sentences: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts via chunking + aggregation.

        Args:
            sentences: list of input texts
            batch_size: batch size for the base encoder (per-chunk encoding)
            show_progress: show tqdm progress bar

        Returns:
            embeddings: [n_sentences, hidden_dim]
        """
        all_embeddings: list[np.ndarray] = []

        iterator = tqdm(sentences, desc="chunk_encode") if show_progress else sentences

        for text in iterator:
            # Step 1: Split into chunks
            token_chunks = self._chunk_text(text)
            chunk_texts = [self._token_ids_to_text(tc) for tc in token_chunks]

            # Step 2: Encode chunks (batch them together for efficiency)
            if len(chunk_texts) == 1:
                chunk_emb = self.encoder.encode(
                    chunk_texts, batch_size=batch_size or self.encoder.batch_size
                )
            else:
                chunk_emb = self.encoder.encode(
                    chunk_texts, batch_size=batch_size or self.encoder.batch_size
                )

            chunk_tensor = torch.from_numpy(chunk_emb).float()

            # Step 3: Aggregate
            aggregated = self._aggregate_chunk_embeddings(chunk_tensor)

            # Step 4: Normalize (consistent with base encoder)
            if self.encoder.normalize:
                aggregated = F.normalize(aggregated, p=2, dim=0)

            all_embeddings.append(aggregated.cpu().float().numpy())

        return np.stack(all_embeddings, axis=0)

    def encode_queries(self, queries: list[str], **kwargs) -> np.ndarray:
        """MTEB-compatible query encoding."""
        return self.encode(queries, **kwargs)

    def encode_corpus(
        self,
        corpus: list[str] | list[dict[str, str]],
        **kwargs,
    ) -> np.ndarray:
        """MTEB-compatible corpus encoding."""
        if corpus and isinstance(corpus[0], dict):
            texts = []
            for doc in corpus:
                title = doc.get("title", "") or ""
                text = doc.get("text", "") or ""
                texts.append(f"{title}\n{text}".strip() if title else text)
            return self.encode(texts, **kwargs)
        return self.encode(list(corpus), **kwargs)


# ---------------------------------------------------------------------------
# Combined: Chunk + Weighted Pooling (RP1 + RP2)
# ---------------------------------------------------------------------------

class ChunkWeightedEncoder(ChunkEncoder):
    """Combines RP1 (weighted pooling) and RP2 (chunking).

    Each chunk is pooled using weighted pooling (attention/TF-IDF/saliency),
    then chunk embeddings are aggregated via mean/weighted aggregation.
    """

    def __init__(
        self,
        base_encoder,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        aggregation: AggregationMethod = "mean",
        max_chunks: int = 32,
        # RP1 params
        use_attention_weight: bool = True,
        use_tfidf_weight: bool = True,
        use_saliency_weight: bool = True,
    ):
        super().__init__(base_encoder, chunk_size, chunk_overlap, aggregation, max_chunks)
        self.use_attention_weight = use_attention_weight
        self.use_tfidf_weight = use_tfidf_weight
        self.use_saliency_weight = use_saliency_weight

        # Initialize TF-IDF calculator if needed
        self.tfidf_calc = None
        if use_tfidf_weight:
            from src.advanced_pooling import TFIDFWeightCalculator
            self.tfidf_calc = None  # lazy init when corpus is available

    def fit_tfidf(self, corpus_texts: list[str]):
        """Precompute TF-IDF weights from corpus texts."""
        from src.advanced_pooling import TFIDFWeightCalculator

        self.tfidf_calc = TFIDFWeightCalculator(self.tokenizer)
        self.tfidf_calc.fit(corpus_texts)
