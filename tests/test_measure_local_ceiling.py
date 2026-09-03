"""End-to-end tests for the local-ceiling tie/coverage measurement.

These build a tiny hand-written dictionary and JSONL in ``tmp_path`` and run
``scripts/measure_local_ceiling.py`` as a subprocess, so they never touch the
gitignored ``data/d1/*.jsonl`` that is absent from a clean CI checkout.
"""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _write_dictionary(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "dictionary.tsv"
    path.write_text("reading\tsurface\tpos\tcost\n" + rows, encoding="utf-8")
    return path


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


def _run(data_path: Path, dictionary_path: Path, output_path: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable, "scripts/measure_local_ceiling.py",
            "--data", str(data_path),
            "--dictionary", str(dictionary_path),
            "--weights", str(ROOT / "data" / "weights" / "ablations" / "decomposed_v1-no_script-surface_len.json"),
            "--output", str(output_path),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_required_keys_are_present(tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path, "か	か	noun	1\n")
    cases = _write_jsonl(tmp_path, [{"id": "c1", "reading": "か", "target_surface": "か", "candidates": ["か"], "viterbi_costs": [0.0]}])
    output = tmp_path / "out.json"

    document = _run(cases, dictionary, output)

    for key in ("cli", "git_commit", "dictionary_sha256", "weights_sha256", "inputs", "ties", "coverage", "feature_distributions"):
        assert key in document
    split = document["ties"]["test-like"] if "test-like" in document["ties"] else next(iter(document["ties"].values()))
    granularity_names = ("lex14", "lex9", "full25", "cost")
    for granularity in granularity_names:
        assert granularity in split["lattice"]
        assert granularity in split["jsonl"]
        for rate in ("any_tied_pair_rate", "top2_tied_rate", "gold_vs_nongold_tied_rate"):
            assert rate in split["lattice"][granularity]
    coverage = next(iter(document["coverage"].values()))
    for key in ("reading_in_dictionary_rate", "reading_lattice_connectable_rate", "gold_pair_in_dictionary_rate", "gold_reachable_rate", "gold_token_count_histogram", "unique_readings"):
        assert key in coverage


def test_tied_configuration_yields_any_tied_pair_rate_one(tmp_path: Path) -> None:
    # Two single-token entries that share reading, pos, length and cost produce
    # identical lexical-14 / lexical-9 / full25 vectors *and* identical cost, so
    # every rate is 1.0 for the only case.
    dictionary = _write_dictionary(tmp_path, "あ	あ	noun	1\nあ	い	noun	1\n")
    cases = _write_jsonl(tmp_path, [{"id": "c1", "reading": "あ", "target_surface": "あ", "candidates": ["あ", "い"], "viterbi_costs": [0.0, 0.0]}])
    output = tmp_path / "out.json"

    document = _run(cases, dictionary, output)
    split = next(iter(document["ties"].values()))

    for granularity in ("lex14", "lex9", "full25", "cost"):
        assert split["lattice"][granularity]["any_tied_pair_rate"] == 1.0
        assert split["lattice"][granularity]["top2_tied_rate"] == 1.0
        assert split["lattice"][granularity]["gold_vs_nongold_tied_rate"] == 1.0
        assert split["jsonl"][granularity]["any_tied_pair_rate"] == 1.0
    # gold "あ" is rank 1 and tied with the non-gold "い" at every granularity.
    assert split["lattice"]["gold_rank_histogram"]["1"] == 1
    assert split["lattice"]["misranked_not_tied"] == 0


def test_non_tied_configuration_yields_zero(tmp_path: Path) -> None:
    dictionary = _write_dictionary(tmp_path, "か	か	noun	1\n")
    cases = _write_jsonl(tmp_path, [{"id": "c1", "reading": "か", "target_surface": "か", "candidates": ["か"], "viterbi_costs": [0.0]}])
    output = tmp_path / "out.json"

    document = _run(cases, dictionary, output)
    split = next(iter(document["ties"].values()))

    for granularity in ("lex14", "lex9", "full25", "cost"):
        assert split["lattice"][granularity]["any_tied_pair_rate"] == 0.0
        assert split["lattice"][granularity]["top2_tied_rate"] == 0.0
        assert split["lattice"][granularity]["gold_vs_nongold_tied_rate"] == 0.0


def test_gold_unreachable_is_excluded_from_the_rate_denominator(tmp_path: Path) -> None:
    # The reading has exactly one entry whose surface is "い"; the gold target
    # "あ" cannot be built, so it is unreachable and dropped from the denominator.
    dictionary = _write_dictionary(tmp_path, "あ	い	noun	1\n")
    cases = _write_jsonl(tmp_path, [{"id": "c1", "reading": "あ", "target_surface": "あ", "candidates": []}])
    output = tmp_path / "out.json"

    document = _run(cases, dictionary, output)
    split = next(iter(document["ties"].values()))

    assert split["gold_unreachable"] == 1
    assert split["cases"] == 1
    # No reachable gold -> denominator is zero -> every rate is defined as 0.0.
    for granularity in ("lex14", "lex9", "full25", "cost"):
        assert split["lattice"][granularity]["gold_vs_nongold_tied_rate"] == 0.0
    assert split["lattice"]["gold_rank_histogram"]["not_in_kbest"] == 1
    coverage = next(iter(document["coverage"].values()))
    assert coverage["gold_reachable_rate"] == 0.0
    assert coverage["gold_unreachable_count"] == 1


def test_does_not_depend_on_d1(tmp_path: Path) -> None:
    # A clean CI checkout has no data/d1/*.jsonl; the measurement must run and
    # report only the tmp_path input it was handed.
    dictionary = _write_dictionary(tmp_path, "か	か	noun	1\n")
    cases = _write_jsonl(tmp_path, [{"id": "c1", "reading": "か", "target_surface": "か", "candidates": ["か"], "viterbi_costs": [0.0]}])
    output = tmp_path / "out.json"

    document = _run(cases, dictionary, output)

    assert len(document["inputs"]) == 1
    input_path = Path(document["inputs"][0]["path"])
    assert "d1" not in input_path.parts


def test_gold_pair_counts_are_not_double_counted(tmp_path: Path) -> None:
    # The gold path is shared by the lattice and JSONL candidate sets; counting
    # it once per set would emit 2x every count in coverage_pairs.jsonl, which
    # is the file task3 measures corpus coverage against.
    dictionary = _write_dictionary(tmp_path, "か\tか\tnoun\t1\nか\t蚊\tnoun\t2\n")
    cases = _write_jsonl(tmp_path, [
        {"id": "c1", "reading": "か", "target_surface": "か", "candidates": ["か", "蚊"], "viterbi_costs": [0.0, 0.0]},
        {"id": "c2", "reading": "か", "target_surface": "か", "candidates": ["か"], "viterbi_costs": [0.0]},
        {"id": "c3", "reading": "か", "target_surface": "蚊", "candidates": ["蚊"], "viterbi_costs": [0.0]},
    ])
    output = tmp_path / "out.json"

    _run(cases, dictionary, output)

    pairs = {
        (row["reading"], row["surface"]): row["count"]
        for row in (json.loads(line) for line in (tmp_path / "coverage_pairs.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert pairs == {("か", "か"): 2, ("か", "蚊"): 1}
    assert sum(pairs.values()) == 3


def test_k_must_equal_beam_width(tmp_path: Path) -> None:
    # Lattice.k_best exposes a single knob; a --k that differs from --beam-width
    # would be recorded in the manifest without ever reaching the search.
    dictionary = _write_dictionary(tmp_path, "か\tか\tnoun\t1\n")
    cases = _write_jsonl(tmp_path, [{"id": "c1", "reading": "か", "target_surface": "か", "candidates": ["か"], "viterbi_costs": [0.0]}])

    result = subprocess.run(
        [
            sys.executable, "scripts/measure_local_ceiling.py",
            "--data", str(cases),
            "--dictionary", str(dictionary),
            "--weights", str(ROOT / "data" / "weights" / "ablations" / "decomposed_v1-no_script-surface_len.json"),
            "--output", str(tmp_path / "out.json"),
            "--k", "32", "--beam-width", "64",
        ],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--k" in result.stderr
