"""
Chunking service implementing sliding window text segmentation with word boundaries and overlap.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TextChunk:
    """Represents a segmented portion of a document."""
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingService:
    """Sliding-window chunker with word-boundary preservation."""

    @classmethod
    def chunk_text(
        cls,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks of approximately chunk_size characters,
        ensuring boundaries do not break in the middle of words.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        cleaned = text.strip()
        if not cleaned:
            return []

        # If text is already under chunk size, return as a single chunk
        if len(cleaned) <= chunk_size:
            tokens = cls._estimate_tokens(cleaned)
            return [TextChunk(chunk_index=0, content=cleaned, token_count=tokens, metadata={"start_char": 0, "end_char": len(cleaned)})]

        chunks: List[TextChunk] = []
        start = 0
        text_length = len(cleaned)
        chunk_index = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)

            if end >= text_length:
                # Last chunk
                chunk_str = cleaned[start:].strip()
                if chunk_str:
                    tokens = cls._estimate_tokens(chunk_str)
                    chunks.append(
                        TextChunk(
                            chunk_index=chunk_index,
                            content=chunk_str,
                            token_count=tokens,
                            metadata={"start_char": start, "end_char": text_length},
                        )
                    )
                break

            # 1. Snap end boundary to whitespace to avoid splitting words at chunk end
            split_at = end
            while split_at > start and not cleaned[split_at].isspace():
                split_at -= 1

            if split_at <= start:
                split_at = end

            chunk_str = cleaned[start:split_at].strip()
            if chunk_str:
                tokens = cls._estimate_tokens(chunk_str)
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_index,
                        content=chunk_str,
                        token_count=tokens,
                        metadata={"start_char": start, "end_char": split_at},
                    )
                )
                chunk_index += 1

            # 2. Advance start with overlap and snap to word boundary
            target_start = split_at - chunk_overlap
            if target_start <= start:
                target_start = start + 1

            # Advance target_start to the beginning of the next whole word
            while target_start < split_at and not cleaned[target_start].isspace():
                target_start += 1
            while target_start < split_at and cleaned[target_start].isspace():
                target_start += 1

            start = target_start

        return chunks

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Heuristic token estimator: 1 token ~ 4 characters or ~0.75 words.
        Accurate within 10-15% of standard BPE tokenizers.
        """
        words = len(text.split())
        return max(1, int(words * 1.3))
