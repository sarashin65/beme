#!/usr/bin/env python3
"""Train a reproducible averaged structured perceptron."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from beme import __version__  # noqa: E402
from beme.dictionary import Dictionary  # noqa: E402
from beme.features import extractor_for  # noqa: E402
from beme.lattice import build_lattice  # noqa: E402
from beme.training import evaluate_split, evaluate_test, train_scorer  # noqa: E402
from beme.training_data import build_manifest, load_splits  # noqa: E402
from beme.scorer import LinearScorer  # noqa: E402
from beme.weights import DEFAULT_WEIGHTS_PATH, Weights  # noqa: E402


DEFAULT_DICTIONARY = REPOSITORY_ROOT / "data" / "dictionary_v2.tsv"
DEFAULT_INITIAL_WEIGHTS = REPOSITORY_ROOT / "data" / "weights" / "decomposed_v1.json"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the BeME scorer deterministically")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--initial-weights", type=Path, default=DEFAULT_INITIAL_WEIGHTS)
    parser.add_argument("--epochs", type=_positive_int, required=True)
    parser.add_argument("--learning-rate", type=_positive_float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-weights", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_path(path: Path) -> str:
    """Keep report provenance useful without embedding machine-specific paths."""

    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return path.name


def _stable_manifest(manifest: dict[str, object], paths: dict[str, Path]) -> dict[str, object]:
    result = json.loads(json.dumps(manifest, ensure_ascii=False))
    for name, path in paths.items():
        result["splits"][name]["path"] = _stable_path(path)
    return result


def _metrics(value: object) -> dict[str, object]:
    return asdict(value)


def _revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, document: object) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise OSError(f"output parent directory does not exist: {parent}")
    fd, temporary = tempfile.mkstemp(prefix=".beme-tmp-", dir=parent)
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


def _atomic_weights(path: Path, weights: Weights) -> None:
    _atomic_json(path, weights.to_dict())


def run(args: argparse.Namespace) -> dict[str, object]:
    paths = {"train": args.train, "dev": args.dev, "test": args.test}
    splits, raw_manifest = load_splits(paths)
    manifest = _stable_manifest(raw_manifest, paths)
    dictionary = Dictionary.from_tsv(args.dictionary)
    initial_weights = Weights.load(args.initial_weights, feature_spec=None)
    extractor = extractor_for(initial_weights.feature_spec, dictionary.reading_conditional)
    dictionary_sha256 = _sha256(args.dictionary)
    initial_weights_sha256 = _sha256(args.initial_weights)
    metadata = {
        "dictionary_sha256": dictionary_sha256,
        "initial_weights_sha256": initial_weights_sha256,
        "split_manifest": manifest,
    }

    def lattice_builder(reading: str):
        return build_lattice(reading, dictionary, allow_literals=True)

    baseline_scorer = LinearScorer(extractor, initial_weights)
    baseline_dev = evaluate_split(splits["dev"], lattice_builder, baseline_scorer)
    baseline_test = evaluate_split(splits["test"], lattice_builder, baseline_scorer)
    training_scorer = LinearScorer(extractor, initial_weights)
    result = train_scorer(
        splits["train"],
        splits["dev"],
        lattice_builder,
        training_scorer,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        extractor=extractor,
        metadata=metadata,
    )
    selected_test = evaluate_test(
        result, splits["test"], lattice_builder, training_scorer
    )
    report = {
        "cli": {
            "train": _stable_path(args.train),
            "dev": _stable_path(args.dev),
            "test": _stable_path(args.test),
            "dictionary": _stable_path(args.dictionary),
            "initial_weights": _stable_path(args.initial_weights),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "output_weights": args.output_weights.name,
            "output_report": args.output_report.name,
        },
        "git_commit": _revision(),
        "python_version": platform.python_version(),
        "beme_version": __version__,
        "feature_spec": initial_weights.feature_spec,
        "dictionary_sha256": dictionary_sha256,
        "initial_weights_sha256": initial_weights_sha256,
        "manifest": manifest,
        "baseline": {"dev": _metrics(baseline_dev), "test": _metrics(baseline_test)},
        "epochs": [
            {
                "epoch": item.epoch,
                "train": _metrics(item.train),
                "dev": _metrics(item.dev),
            }
            for item in result.epochs
        ],
        "selected_epoch": result.selected_epoch,
        "selected_model": {
            "test": _metrics(selected_test),
            "updates": result.averaged_weights.metadata["updates"],
            "averaging_steps": result.averaged_weights.metadata["averaging_steps"],
        },
    }
    _atomic_weights(args.output_weights, result.averaged_weights)
    _atomic_json(args.output_report, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_weights == args.output_report:
        parser.error("--output-weights and --output-report must differ")
    try:
        run(args)
    except (OSError, ValueError, NotImplementedError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
