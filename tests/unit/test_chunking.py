"""
Unit tests for ChunkingService: sliding window chunking, word boundaries, and overlap.
"""
import pytest
from src.services.chunking_service import ChunkingService, TextChunk


def test_chunking_short_text():
    """Verify short text under chunk size produces a single chunk."""
    text = "Short sentence for semantic indexing."
    chunks = ChunkingService.chunk_text(text, chunk_size=200, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == text
    assert chunks[0].token_count > 0


def test_chunking_sliding_window_overlap():
    """Verify long text is broken into overlapping chunks preserving word boundaries."""
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)

    chunk_size = 100
    chunk_overlap = 30
    chunks = ChunkingService.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1
    # Check that chunks overlap
    for i in range(len(chunks) - 1):
        # Chunk i and Chunk i+1 should share some words due to overlap
        words_chunk_i = set(chunks[i].content.split())
        words_chunk_next = set(chunks[i + 1].content.split())
        common_words = words_chunk_i.intersection(words_chunk_next)
        assert len(common_words) > 0

    # Ensure chunk indices are contiguous starting at 0
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_invalid_chunk_overlap_raises():
    """Verify overlap equal or greater than chunk size raises ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        ChunkingService.chunk_text("Some text", chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        ChunkingService.chunk_text("Some text", chunk_size=100, chunk_overlap=-5)
