"""Synthetic end-to-end gates for the Phase 2 contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from beme.converter import Converter
from beme.dictionary import Dictionary
from beme.features import FeatureExtractor, LEGACY_V1_SPEC, path_features
from beme.lattice import (
    Lattice,
    LatticeEdge,
    LatticeNode,
    LatticePath,
    Token,
    _path_key,
    backward_costs_by_state,
    k_best_viterbi,
)
from beme.oracle import constrained_best_path
from beme.scorer import LinearScorer
from beme.training import train_scorer
from beme.training_data import TrainingExample
from beme.weights import Weights


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_scorer.py"
EVALUATE = ROOT / "scripts" / "evaluate.py"


def graph() -> Lattice:
    return Lattice(
        "か",
        (
            LatticeNode(
                0,
                (
                    LatticeEdge(0, 1, Token("か", "A", "noun", 1.0)),
                    LatticeEdge(0, 1, Token("か", "B", "noun", 3.0)),
                ),
            ),
            LatticeNode(1, ()),
        ),
    )


def make_weights(path: Path) -> None:
    path.write_text(
        json.dumps({
            "format": "beme.weights/1",
            "feature_spec": "legacy_v1",
            "default_weight": 0.0,
            "weights": {"lex.base_cost": 1.0, "trans.bos": 0.0},
        }, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def make_cli_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    dictionary = tmp_path / "dictionary.tsv"
    dictionary.write_text(
        "reading\tsurface\tpos\tcost\nか\tA\tnoun\t1\nか\tB\tnoun\t3\n",
        encoding="utf-8",
    )
    initial = tmp_path / "initial.json"
    make_weights(initial)
    splits: dict[str, Path] = {}
    rows = {
        "train": ['{"id":"tr","article_id":"atr","reading":"か","target_surface":"B"}'],
        "dev": [
            '{"id":"dv1","article_id":"adv1","reading":"か","target_surface":"missing"}',
            '{"id":"dv2","article_id":"adv2","reading":"か","target_surface":"B"}',
            '{"id":"dv3","article_id":"adv3","reading":"か","target_surface":"A"}',
        ],
        "test": ['{"id":"te","article_id":"ate","reading":"か","target_surface":"A"}'],
    }
    for name, values in rows.items():
        splits[name] = tmp_path / f"{name}.jsonl"
        splits[name].write_text("\n".join(values) + "\n", encoding="utf-8")
    return dictionary, initial, splits


def run_training_cli(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    dictionary, initial, splits = make_cli_inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    subprocess.run(
        [sys.executable, str(SCRIPT), "--train", str(splits["train"]), "--dev", str(splits["dev"]),
         "--test", str(splits["test"]), "--dictionary", str(dictionary), "--initial-weights", str(initial),
         "--epochs", "1", "--learning-rate", "1", "--seed", "4", "--output-weights",
         str(output / "weights.json"), "--output-report", str(output / "report.json")],
        cwd=ROOT, check=True,
    )
    return output / "weights.json", splits


def test_cli_learned_weights_cross_process_and_evaluate_smoke(tmp_path: Path) -> None:
    learned, splits = run_training_cli(tmp_path)
    loaded = subprocess.run(
        [sys.executable, "-c", "from beme.weights import Weights; Weights.load(__import__('sys').argv[1], feature_spec=None)", str(learned)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert loaded.returncode == 0

    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id":"smoke","reading":"か","judgments":{"A":1,"B":3}}\n', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(EVALUATE), "--suite", "handmade", "--reranker", "mock", "--dictionary",
         str(tmp_path / "dictionary.tsv"), "--weights", str(learned), "--cases", str(cases), "--json"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)["query_count"] == 1
    assert splits["test"].exists()


def test_baseline_and_learned_one_best_are_different(tmp_path: Path) -> None:
    learned, _ = run_training_cli(tmp_path)
    dictionary = Dictionary.from_tsv(tmp_path / "dictionary.tsv")
    baseline = Converter(dictionary).convert("か").candidates[0].surface
    trained = Converter(
        dictionary,
        scorer=LinearScorer(FeatureExtractor(LEGACY_V1_SPEC), Weights.load(learned, feature_spec=None)),
    ).convert("か").candidates[0].surface
    assert baseline == "A"
    assert trained == "B"


def test_negative_weights_backward_and_kbest_match_exhaustive() -> None:
    scorer = LinearScorer(
        FeatureExtractor(LEGACY_V1_SPEC),
        Weights({"lex.base_cost": -1.0, "trans.bos": 0.0}, feature_spec="legacy_v1"),
    )
    lattice = Lattice(
        "か",
        (LatticeNode(0, (LatticeEdge(0, 1, Token("か", "A", "noun", 1.0)),
                        LatticeEdge(0, 1, Token("か", "B", "noun", 3.0)))), LatticeNode(1, ())),
    )
    remaining = backward_costs_by_state(lattice, scorer)
    paths = k_best_viterbi(lattice, beam_width=2, scorer=scorer)
    exhaustive = [
        LatticePath(scorer.edge_cost(item) + scorer.transition_cost(None, item), item.token, LatticePath(0.0), scorer.state_key(item))
        for item in lattice.nodes[0].edges
    ]
    assert remaining[0][None] == min(path.cost for path in exhaustive)
    assert [_path_key(path) for path in paths] == sorted(_path_key(path) for path in exhaustive)


def test_oracle_prediction_features_and_mixed_report_metrics(tmp_path: Path) -> None:
    model = LinearScorer(
        FeatureExtractor(LEGACY_V1_SPEC),
        Weights({"lex.base_cost": 1.0, "trans.bos": 0.0}, feature_spec="legacy_v1"),
    )
    lattice = graph()
    gold = constrained_best_path(lattice, "B", model)
    prediction = k_best_viterbi(lattice, 1, model)[0]
    assert gold is not None and prediction.surface == "A"
    assert sum(model.weights[name] * value for name, value in path_features(gold, model.extractor, model.weights).items()) == gold.cost

    learned, _ = run_training_cli(tmp_path)
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    dev = report["baseline"]["dev"]
    assert (dev["coverage_errors"], dev["covered_examples"], dev["model_errors"], dev["exact_matches"]) == (1, 2, 1, 1)
    assert report["selected_model"]["test"]["examples"] == 1
    assert Weights.load(learned, feature_spec=None).metadata["averaged"] is True
