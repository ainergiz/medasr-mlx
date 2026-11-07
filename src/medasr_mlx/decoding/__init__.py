"""CTC decoders: greedy best-path and prefix beam search (LM + context biasing)."""

from .greedy import greedy_decode

__all__ = ["greedy_decode"]
