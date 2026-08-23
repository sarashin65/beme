"""Tests for cost-scale measurement helpers and self-consistency."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_quantiles_and_null_confidences() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("measure_cost_scale", ROOT / "scripts/measure_cost_scale.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = module.confidence_summary([1.0, 2.0, 3.0, None], 2.0)
    assert result["null_confidence_count"] == 1
    assert result["p50"] == 2.0


def test_handmade_selfcheck_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [sys.executable, "scripts/measure_cost_scale.py", "--suite", "handmade", "--dictionary", "data/dictionary_v2.tsv", "--baseline-weights", "data/weights/decomposed_v1.json", "--learned-weights", "data/weights/decomposed_v1.json"]
    for output in (first, second):
        subprocess.run([*command, "--output", str(output)], cwd=ROOT, check=True)
    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_text(encoding="utf-8"))
    section = document["confidence_distributions"]["handmade"]
    assert section["scale_ratio"] == 1.0
    assert section["threshold_learned"] == 2.0


def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("measure_cost_scale", ROOT / "scripts/measure_cost_scale.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_design_findings_are_derived_from_the_current_source() -> None:
    """The artifact must not carry line numbers that the source has moved past."""
    module = _module()
    expected = []
    for name in module.SOURCES:
        path = ROOT / "src" / "beme" / f"{name}.py"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "needs_rerank" in line:
                expected.append(f"src/beme/{name}.py:{number}")
    finding = module._needs_rerank_finding()
    assert finding["references"] == expected
    assert expected, "needs_rerank should still exist in the source"
    assert finding["gating_uses"] == []
    assert finding["confirmed"] is True


def test_a_conditional_on_needs_rerank_would_refute_the_finding(tmp_path: Path) -> None:
    module = _module()
    source = (ROOT / "src" / "beme" / "converter.py").read_text(encoding="utf-8")
    patched = source.replace(
        "        reranked = not context.is_empty and self.reranker is not None\n",
        "        reranked = not context.is_empty and self.reranker is not None\n        if needs_rerank:\n            pass\n",
        1,
    )
    assert patched != source
    package = tmp_path / "src" / "beme"
    package.mkdir(parents=True)
    (package / "converter.py").write_text(patched, encoding="utf-8")
    module.ROOT = tmp_path
    try:
        assert module.needs_rerank_gating_uses()
    finally:
        module.ROOT = ROOT
