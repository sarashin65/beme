"""BeME deterministic kana conversion and Phase 2 context reranking."""

from .converter import Candidate, ConversionResult, Converter, Token
from .dictionary import Dictionary, DictionaryEntry, MiniDictionary, load_tsv
from .lattice import (
    Lattice,
    LatticeEdge,
    LatticeNode,
    LatticePath,
    build_lattice,
    k_best_viterbi,
    pos_transition_cost,
)
from .oracle import constrained_best_path
from .training_data import TrainingExample, build_manifest, load_jsonl, load_splits
from .training import (
    EpochResult,
    SplitEvaluation,
    TrainingResult,
    evaluate_split,
    evaluate_test,
    train_scorer,
)
from .reranker import (
    ConversionContext,
    CrossEncoderReranker,
    CrossEncoderRerankerUnavailable,
    LLMLogitsReranker,
    LLMRerankerUnavailable,
    MockContextReranker,
    Reranker,
)

__version__ = "0.1.0"

__all__ = [
    "Candidate",
    "ConversionResult",
    "ConversionContext",
    "CrossEncoderReranker",
    "CrossEncoderRerankerUnavailable",
    "Converter",
    "Dictionary",
    "DictionaryEntry",
    "Lattice",
    "LatticeEdge",
    "LatticeNode",
    "LatticePath",
    "MiniDictionary",
    "LLMLogitsReranker",
    "LLMRerankerUnavailable",
    "MockContextReranker",
    "Reranker",
    "Token",
    "__version__",
    "build_lattice",
    "constrained_best_path",
    "TrainingExample",
    "EpochResult",
    "SplitEvaluation",
    "TrainingResult",
    "build_manifest",
    "evaluate_split",
    "evaluate_test",
    "load_jsonl",
    "load_splits",
    "train_scorer",
    "k_best_viterbi",
    "load_tsv",
    "pos_transition_cost",
]
