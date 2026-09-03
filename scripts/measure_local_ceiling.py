#!/usr/bin/env python3
"""Measure how often the local (path) model cannot rank d1 candidates.

Two questions, both needed to size the F2 corpus-frequency features (issue #19):

* S1 -- ties: within the candidate set the local model actually scores, how
  often are two candidates indistinguishable?  Measured at four granularities
  (lexical-14, lexical-9, full lexical+transition, path cost) across two
  candidate sets (the v2 lattice k-best, and the JSONL ``candidates`` column),
  three rates each (any tied pair, top-2 tied, gold tied to a non-gold).
* S2 -- coverage: how often is the reading / the (reading, surface) pair in the
  dictionary, and how often can the gold surface even be built on the lattice.

The primary measurement is the v2 lattice k-best (the set the trained model
ranks); the JSONL column is a secondary, v1-dictionary-derived proxy.  ``src/beme/``
is never modified -- only read through its public API.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.dictionary import Dictionary  # noqa: E402
from beme.features import (  # noqa: E402
    DECOMPOSED_V1_SPEC,
    FeatureExtractor,
    extractor_for,
    path_features,
)
from beme.lattice import Lattice, build_lattice  # noqa: E402
from beme.oracle import constrained_best_path  # noqa: E402
from beme.scorer import LinearScorer  # noqa: E402
from beme.training_data import TrainingExample, load_jsonl  # noqa: E402
from beme.weights import Weights  # noqa: E402

# Cost ties are declared with an absolute tolerance rather than bit equality:
# a path cost is a sum of ~25 weighted features, so two genuinely different
# feature vectors can still round to costs that ``math.isclose`` treats as
# equal while ``==`` does not.  rel_tol=0 keeps this a pure absolute band around
# the quantisation floor of the IPADIC costs, which is exactly the "same score"
# question the linear model faces when it breaks a tie.
_COST_TIE_ABS_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class Granularity:
    name: str
    # ``fn`` returns the feature *dict* for a path (cost is handled separately as
    # the path's own ``.cost``); ``is_lexicographic`` marks the per-feature
    # distribution we accumulate for the F2 temperature constant.  ``is_cost``
    # marks the path-cost granularity, whose key is the path's own ``.cost``
    # rather than a feature vector.
    fn: Any
    is_lexicographic: bool
    is_cost: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name if path.is_absolute() else str(path)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _nonneg_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True, help="d1 JSONL split (repeatable)")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "data" / "dictionary_v2.tsv")
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "weights" / "ablations" / "decomposed_v1-no_script-surface_len.json", help="weights for the path-cost granularity")
    parser.add_argument("--specs", type=str, default="decomposed_v1,decomposed_v1-no_script-surface_len", help="comma-separated feature specs that gate the lexical/feature granularities")
    parser.add_argument("--k", type=_positive_int, default=64, help="k of the lattice k-best; Lattice.k_best exposes a single knob, so this must equal --beam-width")
    parser.add_argument("--beam-width", type=_positive_int, default=64)
    parser.add_argument("--limit", type=_nonneg_int, default=None, help="only process the first N examples (debugging)")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _feature_key(features: dict[str, float]) -> tuple:
    """A hashable, bit-exact key for a feature vector.

    ``repr`` of each value preserves the full float bit pattern, so two vectors
    only compare equal when every component is identical to the last bit -- no
    rounding band.  Sorting by name makes the key independent of insertion order.
    """

    return tuple(sorted((name, repr(value)) for name, value in features.items()))


def _make_granularities(spec_ids: tuple[str, ...], extractors: dict[str, FeatureExtractor]) -> list[Granularity]:
    """Build the lexical/feature granularities the --specs select.

    ``decomposed_v1`` contributes the lexical-14 granularity (Opus's 24.9% is
    defined at exactly this granularity); the A8 spec contributes lexical-9 and
    the full lexical+transition (25-dimension) granularity, which is the only
    one whose tie means *any* weight vector cannot rank the pair.  The path-cost
    granularity is always added.
    """

    granularities: list[Granularity] = []
    if "decomposed_v1" in spec_ids:
        decomposed = extractors["decomposed_v1"]
        decomposed_lexical = set(decomposed.spec.lexical_features)

        def lexical14(path: Any) -> dict[str, float]:
            return {name: value for name, value in path_features(path, decomposed).items() if name in decomposed_lexical}

        granularities.append(Granularity("lex14", lexical14, True))
    if "decomposed_v1-no_script-surface_len" in spec_ids:
        a8 = extractors["decomposed_v1-no_script-surface_len"]
        a8_lexical = set(a8.spec.lexical_features)

        def lexical9(path: Any) -> dict[str, float]:
            return {name: value for name, value in path_features(path, a8).items() if name in a8_lexical}

        def full25(path: Any) -> dict[str, float]:
            return path_features(path, a8)

        granularities.append(Granularity("lex9", lexical9, True))
        granularities.append(Granularity("full25", full25, True))
    # The path-cost granularity is always present: it is the only one whose tie
    # is decided by the trained weights' actual scoring, independent of which
    # --specs were selected.
    granularities.append(Granularity("cost", None, False, is_cost=True))
    if not granularities:
        raise SystemExit("no selected --spec matches decomposed_v1 or decomposed_v1-no_script-surface_len")
    return granularities


def _tie_booleans(
    path_keys: list[dict[str, Any]],
    gold_keys: dict[str, Any] | None,
    gold_index: int | None,
) -> dict[str, tuple[bool, bool, bool]]:
    """Return ``(any_tied_pair, top2_tied, gold_vs_nongold_tied)`` per granularity.

    ``path_keys`` is the per-path key map for one candidate set (already ordered
    by ascending cost).  ``gold_keys`` / ``gold_index`` are ``None`` when the gold
    surface is unreachable; a reachable gold that is simply absent from the k-best
    still compares against every candidate via ``gold_keys``.
    """

    count = len(path_keys)
    result: dict[str, tuple[bool, bool, bool]] = {}
    for name in path_keys[0]:
        column = [keys[name] for keys in path_keys]
        if name == "cost":
            any_tied = any(
                math.isclose(column[i], column[i + 1], rel_tol=0, abs_tol=_COST_TIE_ABS_TOL)
                for i in range(count - 1)
            )
            top2_tied = count >= 2 and math.isclose(column[0], column[1], rel_tol=0, abs_tol=_COST_TIE_ABS_TOL)
            if gold_keys is None or gold_index is None:
                gold_tied = False
            else:
                gold_tied = any(
                    other != gold_index
                    and math.isclose(gold_keys[name], column[other], rel_tol=0, abs_tol=_COST_TIE_ABS_TOL)
                    for other in range(count)
                )
        else:
            any_tied = len(set(column)) < count
            top2_tied = count >= 2 and column[0] == column[1]
            if gold_keys is None or gold_index is None:
                gold_tied = False
            else:
                gold_tied = any(
                    other != gold_index and column[other] == gold_keys[name] for other in range(count)
                )
        result[name] = (any_tied, top2_tied, gold_tied)
    return result


def _merge_histogram(histogram: dict[str, int], gold_rank: int | None) -> dict[str, int]:
    merged = dict(histogram)
    rank = gold_rank + 1 if gold_rank is not None else -1
    if rank == 1:
        merged["1"] += 1
    elif rank == 2:
        merged["2"] += 1
    elif rank == 3:
        merged["3"] += 1
    elif 4 <= rank <= 10:
        merged["4-10"] += 1
    elif 11 <= rank <= 64:
        merged["11-64"] += 1
    else:
        merged["not_in_kbest"] += 1
    return merged


def _rate_section(
    total: int,
    gold_unreachable: int,
    any_counts: dict[str, int],
    top2_counts: dict[str, int],
    gold_counts: dict[str, int],
    denominator: float,
    *,
    extra: dict[str, Any],
) -> dict[str, Any]:
    section: dict[str, Any] = {"cases": total, "gold_unreachable": gold_unreachable}
    for name in any_counts:
        section[name] = {
            "any_tied_pair_rate": any_counts[name] / denominator if denominator else 0.0,
            "top2_tied_rate": top2_counts[name] / denominator if denominator else 0.0,
            "gold_vs_nongold_tied_rate": gold_counts[name] / denominator if denominator else 0.0,
        }
    section.update(extra)
    return section


class _RunningStats:
    """Welford running mean/std plus min/max without storing every value."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def summary(self) -> dict[str, float]:
        variance = self.m2 / self.count if self.count > 0 else 0.0
        return {
            "mean": self.mean,
            "std": math.sqrt(variance),
            "min": self.min if self.count else 0.0,
            "max": self.max if self.count else 0.0,
            "count": float(self.count),
        }


def _analyze_candidate_set(
    paths: tuple[Any, ...],
    gold_path: Any | None,
    gold_surface: str | None,
    granularities: list[Granularity],
    *,
    is_lattice: bool,
    candidate_unreachable: int,
    dec_distribution: dict[str, _RunningStats] | None,
) -> dict[str, Any]:
    """Return the per-case S1 booleans plus the gold-rank artefacts for one set.

    The returned ``booleans`` map each granularity to ``(any_tied_pair,
    top2_tied, gold_vs_nongold_tied)`` for this single case; ``_process_split``
    accumulates them into rates.  ``gold_rank`` is the gold's 0-based index in
    the ordered candidate set (``None`` when the gold surface is absent), used to
    build the histogram and the ``misranked_not_tied`` count.
    """

    gold_keys: dict[str, Any] | None = None
    if gold_path is not None:
        gold_keys = {}
        for granularity in granularities:
            if granularity.is_cost:
                gold_keys[granularity.name] = gold_path.cost
            else:
                gold_keys[granularity.name] = _feature_key(granularity.fn(gold_path))

    ordered = sorted(paths, key=lambda path: (path.cost, path.surface))
    if not ordered:
        # An empty candidate set has no pairs, so nothing can tie at any
        # granularity.  (The lattice k-best is empty when the reading is not
        # connectable; the JSONL set is empty when every candidate surface is
        # unreachable.)
        return {
            "booleans": {granularity.name: (False, False, False) for granularity in granularities},
            "gold_rank": None,
            "misranked_not_tied": 0,
            "candidate_unreachable": candidate_unreachable,
        }
    path_keys: list[dict[str, Any]] = []
    for path in ordered:
        keys: dict[str, Any] = {}
        dec_dict: dict[str, float] | None = None
        for granularity in granularities:
            if granularity.is_cost:
                continue
            features = granularity.fn(path)
            keys[granularity.name] = _feature_key(features)
            if granularity.is_lexicographic and granularity.name == "lex14":
                dec_dict = features
        keys["cost"] = path.cost
        path_keys.append(keys)
        if dec_dict is not None and dec_distribution is not None:
            for name, value in dec_dict.items():
                dec_distribution[name].update(float(value))

    gold_index: int | None = None
    if gold_surface is not None:
        for index, path in enumerate(ordered):
            if path.surface == gold_surface:
                gold_index = index
                break

    booleans = _tie_booleans(path_keys, gold_keys, gold_index)
    misranked = 0
    if is_lattice and gold_index is not None and gold_index > 0 and gold_keys is not None:
        full25 = path_keys[gold_index].get("full25")
        top_full25 = path_keys[0].get("full25") if path_keys else None
        if full25 is not None and top_full25 is not None and full25 != top_full25:
            misranked += 1
    return {
        "booleans": booleans,
        "gold_rank": gold_index,
        "misranked_not_tied": misranked,
        "candidate_unreachable": candidate_unreachable,
    }


def _process_split(
    examples: tuple[TrainingExample, ...],
    dictionary: Dictionary,
    cost_scorer: LinearScorer,
    granularities: list[Granularity],
    beam_width: int,
    limit: int | None,
    dec_distribution: dict[str, _RunningStats] | None,
    gold_pair_multiset: Counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = examples[:limit] if limit is not None else examples
    total = len(cases)

    # Per-set, per-granularity counts of cases whose single-case booleans were
    # True; divided by the reachable denominator below to form the rates.
    any_counts: dict[str, dict[str, int]] = {"lattice": defaultdict(int), "jsonl": defaultdict(int)}
    top2_counts: dict[str, dict[str, int]] = {"lattice": defaultdict(int), "jsonl": defaultdict(int)}
    gold_counts: dict[str, dict[str, int]] = {"lattice": defaultdict(int), "jsonl": defaultdict(int)}
    gold_unreachable = 0
    gold_reachable = 0
    lattice_connectable = 0
    gold_pair_in_dictionary = 0
    token_histogram = {"1": 0, "2": 0, "3": 0, "4+": 0}
    lattice_histogram: dict[str, int] = {"1": 0, "2": 0, "3": 0, "4-10": 0, "11-64": 0, "not_in_kbest": 0}
    lattice_misranked = 0
    jsonl_candidate_unreachable = 0
    # Seed every granularity so a set with no True booleans (e.g. an empty
    # candidate set) still emits all three rates rather than missing keys.
    for granularity in granularities:
        for set_name in ("lattice", "jsonl"):
            any_counts[set_name][granularity.name] = 0
            top2_counts[set_name][granularity.name] = 0
            gold_counts[set_name][granularity.name] = 0
    readings: set[str] = set()
    pairs: set[tuple[str, str]] = set()

    lattice_cache: dict[str, Lattice] = {}

    for example in cases:
        readings.add(example.reading)
        pairs.add((example.reading, example.target_surface))
        try:
            lattice = lattice_cache.get(example.reading)
            if lattice is None:
                lattice = build_lattice(example.reading, dictionary, allow_literals=True)
                lattice_cache[example.reading] = lattice
        except ValueError:
            # Non-hiragana reading: no lattice can be built; treated as
            # unreachable / unconnectable, consistent with the scorer.
            gold_unreachable += 1
            continue

        if lattice.k_best(beam_width, cost_scorer):
            lattice_connectable += 1

        gold_path = constrained_best_path(lattice, example.target_surface, cost_scorer)
        gold_surface = gold_path.surface if gold_path is not None else None
        if gold_path is None:
            gold_unreachable += 1
        else:
            gold_reachable += 1
            # Exactly once per case -- the two candidate sets below share this
            # gold path, so counting inside _analyze_candidate_set would double
            # every occurrence in coverage_pairs.jsonl.
            gold_pair_multiset[(gold_path.reading, gold_path.surface)] += 1
            entries = dictionary.entries_for_reading(example.reading)
            if any(entry.surface == example.target_surface for entry in entries):
                gold_pair_in_dictionary += 1
            token_count = min(len(gold_path.tokens), 4)
            token_histogram["4+" if token_count == 4 else str(token_count)] += 1

        lattice_case = _analyze_candidate_set(
            lattice.k_best(beam_width, cost_scorer),
            gold_path,
            gold_surface,
            granularities,
            is_lattice=True,
            candidate_unreachable=0,
            dec_distribution=dec_distribution,
        )
        # All three rates share the gold-reachable denominator, so a case whose
        # gold cannot be built contributes to none of the numerators either.
        for granularity in granularities if gold_path is not None else ():
            any_tied, top2_tied, gold_tied = lattice_case["booleans"][granularity.name]
            if any_tied:
                any_counts["lattice"][granularity.name] += 1
            if top2_tied:
                top2_counts["lattice"][granularity.name] += 1
            if gold_tied:
                gold_counts["lattice"][granularity.name] += 1
        lattice_histogram = _merge_histogram(lattice_histogram, lattice_case["gold_rank"])
        lattice_misranked += lattice_case["misranked_not_tied"]

        reachable: list[Any] = []
        candidate_unreachable = 0
        for surface in example.candidates:
            path = constrained_best_path(lattice, surface, cost_scorer)
            if path is None:
                candidate_unreachable += 1
            else:
                reachable.append(path)
        jsonl_case = _analyze_candidate_set(
            tuple(reachable),
            gold_path,
            gold_surface,
            granularities,
            is_lattice=False,
            candidate_unreachable=candidate_unreachable,
            dec_distribution=dec_distribution,
        )
        jsonl_candidate_unreachable += jsonl_case["candidate_unreachable"]
        for granularity in granularities if gold_path is not None else ():
            any_tied, top2_tied, gold_tied = jsonl_case["booleans"][granularity.name]
            if any_tied:
                any_counts["jsonl"][granularity.name] += 1
            if top2_tied:
                top2_counts["jsonl"][granularity.name] += 1
            if gold_tied:
                gold_counts["jsonl"][granularity.name] += 1

    denominator = gold_reachable if gold_reachable else 0
    tie_result = {
        "cases": total,
        "gold_unreachable": gold_unreachable,
        "lattice": _rate_section(
            total, gold_unreachable, any_counts["lattice"], top2_counts["lattice"], gold_counts["lattice"], denominator,
            extra={"gold_rank_histogram": lattice_histogram, "misranked_not_tied": lattice_misranked},
        ),
        "jsonl": _rate_section(
            total, gold_unreachable, any_counts["jsonl"], top2_counts["jsonl"], gold_counts["jsonl"], denominator,
            extra={"candidate_unreachable": jsonl_candidate_unreachable},
        ),
    }

    coverage = {
        "reading_in_dictionary_rate": _rate(lambda ex: bool(dictionary.entries_for_reading(ex.reading)), cases),
        "reading_lattice_connectable_rate": lattice_connectable / total if total else 0.0,
        "gold_pair_in_dictionary_rate": gold_pair_in_dictionary / total if total else 0.0,
        "gold_reachable_rate": gold_reachable / total if total else 0.0,
        "gold_unreachable_count": gold_unreachable,
        "gold_token_count_histogram": token_histogram,
        "unique_readings": len(readings),
        "unique_gold_pairs": len(pairs),
    }
    return tie_result, coverage


def _rate(predicate, examples) -> float:
    count = sum(1 for example in examples if predicate(example))
    return count / len(examples) if examples else 0.0


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".measure-local-ceiling-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _split_name(data_path: Path) -> str:
    stem = data_path.stem
    if stem in {"train", "dev", "test"}:
        return stem
    return stem


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _write_coverage_pairs(path: Path, multiset: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"reading": reading, "surface": surface, "count": count}, ensure_ascii=False)
        for (reading, surface), count in sorted(multiset.items())
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.k != args.beam_width:
        # Lattice.k_best(beam_width, scorer) has no separate k: the beam *is* the
        # k.  Reject the mismatch rather than record a --k the run never used.
        raise SystemExit(f"--k ({args.k}) must equal --beam-width ({args.beam_width})")

    spec_ids = tuple(slug.strip() for slug in args.specs.split(","))
    dictionary = Dictionary.from_tsv(args.dictionary)
    weights = Weights.load(args.weights, feature_spec=None)
    cost_scorer = LinearScorer(extractor_for(weights.feature_spec, dictionary.reading_conditional), weights)

    extractors: dict[str, FeatureExtractor] = {}
    for spec_id in spec_ids:
        extractors[spec_id] = extractor_for(spec_id, dictionary.reading_conditional)
    # decomposed_v1 is always needed for the lexical-14 granularity and the
    # per-feature distribution, even when not named in --specs.
    extractors.setdefault("decomposed_v1", extractor_for("decomposed_v1", dictionary.reading_conditional))
    granularities = _make_granularities(tuple(extractors), extractors)

    ties: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    dec_distribution: dict[str, _RunningStats] | None = {name: _RunningStats() for name in DECOMPOSED_V1_SPEC.lexical_features}
    gold_pair_multiset: Counter = Counter()

    inputs = []
    for data_path in args.data:
        examples = load_jsonl(data_path)
        tie_section, coverage_section = _process_split(
            examples,
            dictionary,
            cost_scorer,
            granularities,
            args.beam_width,
            args.limit,
            dec_distribution,
            gold_pair_multiset,
        )
        ties[_split_name(data_path)] = tie_section
        coverage[_split_name(data_path)] = coverage_section
        inputs.append({"path": _display_path(data_path), "sha256": _sha256(data_path), "examples": len(examples)})

    feature_distributions: dict[str, Any] = {}
    if dec_distribution is not None:
        for name in sorted(dec_distribution):
            feature_distributions[name] = dec_distribution[name].summary()

    document = {
        "cli": {
            "data": [_display_path(path) for path in args.data],
            "dictionary": _display_path(args.dictionary),
            "weights": _display_path(args.weights),
            "specs": args.specs,
            "k": args.k,
            "beam_width": args.beam_width,
            "limit": args.limit,
            "output": _display_path(args.output),
        },
        "git_commit": _git_commit(),
        "dictionary_sha256": _sha256(args.dictionary),
        "weights_sha256": _sha256(args.weights),
        "weights_feature_spec": weights.feature_spec,
        "granularities": [granularity.name for granularity in granularities],
        "k": args.k,
        "beam_width": args.beam_width,
        "inputs": inputs,
        "ties": ties,
        "coverage": coverage,
        "feature_distributions": feature_distributions,
    }
    _atomic_json(args.output, document)

    output_path = Path(args.output)
    pairs_path = output_path.with_name("coverage_pairs.jsonl")
    _write_coverage_pairs(pairs_path, gold_pair_multiset)

    print(f"cases measured: {sum(item['examples'] for item in inputs)}")
    print(f"ties -> {_display_path(args.output)}")
    print(f"gold pairs -> {_display_path(pairs_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
