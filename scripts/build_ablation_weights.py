#!/usr/bin/env python3
"""Build deterministic initial weights for the feature ablations."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beme.features import registered_spec_ids, resolve_spec  # noqa: E402
from beme.weights import Weights  # noqa: E402


def main() -> int:
    source = ROOT / "data" / "weights" / "decomposed_v1.json"
    output_dir = ROOT / "data" / "weights" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Weights.load(source, feature_spec="decomposed_v1")
    for spec_id in registered_spec_ids():
        if not (spec_id.startswith("decomposed_v1-") or spec_id.startswith("decomposed_v1+")):
            continue
        spec = resolve_spec(spec_id)
        known = set(spec.lexical_features) | set(spec.transition_features)
        values = {name: value for name, value in base.values.items() if name in known}
        if "trans.matrix_cost" in known:
            values["trans.matrix_cost"] = 0.0
        for name in ("lex.reading_cond_logprob", "lex.reading_candidate_count"):
            if name in known:
                values[name] = 0.0
        Weights(values, base.default_weight, spec_id, base.metadata).dump(
            output_dir / f"{spec_id}.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
