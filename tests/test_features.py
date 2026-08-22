"""Path feature extraction and serialization primitives."""

from __future__ import annotations

from pathlib import Path
import json

from beme.dictionary import Dictionary
from beme.features import FeatureExtractor, feature_delta, legacy_v1_extractor, path_features
from beme.lattice import build_lattice, k_best_viterbi
from beme.scorer import LinearScorer
from beme.weights import DEFAULT_WEIGHTS_PATH, Weights


def test_feature_delta_is_sparse_symmetric_and_sorted() -> None:
    gold = {"z": 3.0, "a": 2.0, "zero": 1.0}
    pred = {"z": 1.0, "a": 2.0, "zero": 1.0, "b": 4.0}
    assert list(feature_delta(gold, pred)) == ["b", "z"]
    assert feature_delta(gold, pred) == {"b": -4.0, "z": 2.0}
    assert feature_delta(gold, pred) == {
        key: -value for key, value in feature_delta(pred, gold).items()
    }


def test_path_features_reconstruct_legacy_cost() -> None:
    dictionary = Dictionary.from_tsv(Path("data/dictionary.tsv"))
    extractor = legacy_v1_extractor()
    scorer = LinearScorer(extractor, Weights.load(DEFAULT_WEIGHTS_PATH))
    for reading in ("きしゃ", "きしゃのきしゃ", "きしゃ123かな!"):
        lattice = build_lattice(reading, dictionary, allow_literals=True)
        for path in k_best_viterbi(lattice, beam_width=5, scorer=scorer):
            features = path_features(path, extractor)
            assert sum(scorer.weights[name] * value for name, value in features.items()) == path.cost


def test_feature_json_round_trip(tmp_path: Path) -> None:
    document = {"features": {"lex.base_cost": 0.8, "trans.bos": 1.0}, "rank": 0}
    path = tmp_path / "features.jsonl"
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == document
