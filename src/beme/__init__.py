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
    "k_best_viterbi",
    "load_tsv",
    "pos_transition_cost",
]
