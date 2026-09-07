"""Feature specification registry and ablation behavior."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pytest

from beme.corpus_freq import CorpusFrequencyTable
from beme.features import (
    DECOMPOSED_V1_SPEC,
    FeatureExtractor,
    decomposed_v1_extractor,
    extractor_for,
    legacy_v1_extractor,
    registered_spec_ids,
    resolve_spec,
)
from beme.lattice import LatticeEdge, Token
from beme.weights import Weights


ROOT = Path(__file__).resolve().parents[1]


def _corpus_table() -> CorpusFrequencyTable:
    # Tiny standalone table so the registry test does not depend on the
    # untracked data/corpus_freq_v1.bin (see issue #19 task4 review M1).
    return CorpusFrequencyTable.from_counts(
        pairs={("きしゃ", "汽車"): 1},
        surfaces={"汽車": 1},
        total=1,
    )


def test_registry_contains_the_two_bases_and_eight_ablations() -> None:
    assert registered_spec_ids() == (
        "legacy_v1", "decomposed_v1", "decomposed_v1-lex_only",
        "decomposed_v1-trans_only", "decomposed_v1-no_ipadic_cost",
        "decomposed_v1-no_source", "decomposed_v1-no_katakana",
        "decomposed_v1-no_script", "decomposed_v1-no_script-surface_len",
        "decomposed_v1-no_shape", "decomposed_v1+matrix_v1",
        "decomposed_v1+matrix_v1-pos_pairs", "decomposed_v1+freq_v1",
        "decomposed_v1+freq_v1-no_ipadic_cost", "decomposed_v1+matrix_v1+freq_v1",
        "decomposed_v1-no_script-surface_len+corpus_v1", "decomposed_v1+corpus_v1",
        "decomposed_v1-no_script-surface_len+sparse_v1",
        "decomposed_v1-no_script-surface_len+corpus_v1+sparse_v1",
    )


def test_base_extractors_are_unchanged() -> None:
    edge = LatticeEdge(0, 2, Token("にほん", "日本", "proper_noun", 0.65, source="overlay"))
    for extractor in (legacy_v1_extractor(), decomposed_v1_extractor()):
        assert extractor.lexical_features(edge) == FeatureExtractor(extractor.spec).lexical_features(edge)
    assert decomposed_v1_extractor().transition_features(None, edge) == {"trans.bos": 1.0}


@pytest.mark.parametrize("spec_id", registered_spec_ids()[2:])
def test_ablation_outputs_are_declared(spec_id: str) -> None:
    edge = LatticeEdge(0, 2, Token("きしゃ", "汽車", "noun", 2.0, raw_cost=3500, source="ipadic"))
    if "+corpus_v1" in spec_id:
        extractor = FeatureExtractor(
            resolve_spec(spec_id), {}, corpus_frequency=_corpus_table()
        )
    else:
        extractor = extractor_for(spec_id, {})
    lexical = extractor.lexical_features(edge)
    transition = extractor.transition_features(None, edge)
    assert set(lexical) <= set(extractor.spec.lexical_features)
    assert set(transition) <= set(extractor.spec.transition_features)


def test_lexical_and_transition_only_ablate_the_opposite_family() -> None:
    edge = LatticeEdge(0, 1, Token("あ", "あ", "noun", 1.0))
    assert extractor_for("decomposed_v1-trans_only").lexical_features(edge) == {}
    assert extractor_for("decomposed_v1-lex_only").transition_features(None, edge) == {}


def test_script_ablations_remove_only_the_declared_shape_features() -> None:
    edge = LatticeEdge(
        0, 2, Token("にほん", "日本", "proper_noun", 0.65, source="overlay")
    )
    script_features = {
        "lex.script.kanji", "lex.script.hiragana",
        "lex.script.katakana", "lex.script.other",
    }

    no_script = extractor_for("decomposed_v1-no_script").lexical_features(edge)
    assert script_features.isdisjoint(no_script)
    assert no_script["lex.surface_len"] == 2.0
    assert no_script["lex.proper_noun"] == 1.0

    no_script_surface_len = extractor_for(
        "decomposed_v1-no_script-surface_len"
    ).lexical_features(edge)
    assert script_features.isdisjoint(no_script_surface_len)
    assert "lex.surface_len" not in no_script_surface_len
    assert no_script_surface_len["lex.proper_noun"] == 1.0


def test_unknown_spec_is_rejected_everywhere() -> None:
    with pytest.raises(ValueError, match="unsupported feature specification"):
        resolve_spec("does_not_exist")
    with pytest.raises(ValueError, match="unsupported feature specification"):
        extractor_for("does_not_exist")


def test_ablation_weight_files_are_strictly_loadable_after_build(tmp_path: Path) -> None:
    subprocess.run([sys.executable, "scripts/build_ablation_weights.py"], cwd=ROOT, check=True)
    for spec_id in registered_spec_ids()[2:]:
        path = ROOT / "data" / "weights" / "ablations" / f"{spec_id}.json"
        loaded = Weights.load(path, feature_spec=spec_id)
        assert set(loaded) <= set(resolve_spec(spec_id).lexical_features) | set(resolve_spec(spec_id).transition_features)


def test_removed_feature_is_rejected_by_strict_weights(tmp_path: Path) -> None:
    document = json.loads((ROOT / "data" / "weights" / "decomposed_v1.json").read_text())
    document["feature_spec"] = "decomposed_v1-no_source"
    path = tmp_path / "invalid-ablation.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown feature"):
        Weights.load(path, feature_spec=None)


def test_evaluate_does_not_fallback_for_unknown_spec(tmp_path: Path) -> None:
    document = json.loads((ROOT / "data" / "weights" / "decomposed_v1.json").read_text())
    document["feature_spec"] = "unknown_v1"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--suite", "handmade", "--reranker", "mock",
         "--dictionary", "data/dictionary_v2.tsv", "--weights", str(path)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "unsupported feature specification" in result.stderr
