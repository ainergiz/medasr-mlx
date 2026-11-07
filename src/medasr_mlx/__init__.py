"""MedASR-MLX: Google's MedASR medical speech recognizer on Apple MLX."""

from .config import ModelConfig
from .result import Segment, TranscriptionResult, Word
from .tokenizer import CTCTokenizer
from .transcribe import MedASR
from .weights import DEFAULT_MODEL, load_model

__version__ = "2.0.0.dev0"


def load(path_or_repo: str = DEFAULT_MODEL, **kwargs) -> MedASR:
    """Load a MedASR checkpoint from a local directory or the Hugging Face Hub."""
    return MedASR.from_pretrained(path_or_repo, **kwargs)


__all__ = [
    "DEFAULT_MODEL",
    "CTCTokenizer",
    "MedASR",
    "ModelConfig",
    "Segment",
    "TranscriptionResult",
    "Word",
    "__version__",
    "load",
    "load_model",
]
