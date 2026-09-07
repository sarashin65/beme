"""Tests for SparseLexicalWeights."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from beme.sparse_weights import SparseLexicalWeights


def test_empty() -> None:
    w = SparseLexicalWeights.empty()
    assert w.get("foo", "bar") == 0.0
    assert w.num_nonzero == {"pair": 0, "surface": 0, "reading": 0}


def test_pair_weight() -> None:
    w = SparseLexicalWeights.empty()
    w.set_pair("foo", "bar", 1.5)
    assert w.get("foo", "bar") == 1.5
    assert w.get("foo", "baz") == 0.0  # different pair → backoff to 0


def test_backoff_surface() -> None:
    w = SparseLexicalWeights.empty()
    w.set_surface("bar", 2.0)
    # No pair weight, but surface exists → backoff to surface
    assert w.get("foo", "bar") == 2.0


def test_backoff_reading() -> None:
    w = SparseLexicalWeights.empty()
    w.set_reading("foo", 3.0)
    # No pair or surface, but reading exists → backoff to reading
    assert w.get("foo", "bar") == 3.0


def test_backoff_priority() -> None:
    """Pair should take priority over surface, which takes priority over reading."""
    w = SparseLexicalWeights.empty()
    w.set_pair("foo", "bar", 1.0)
    w.set_surface("bar", 2.0)
    w.set_reading("foo", 3.0)
    assert w.get("foo", "bar") == 1.0  # pair wins


def test_prune_zeros() -> None:
    w = SparseLexicalWeights.empty()
    w.set_pair("foo", "bar", 1e-13)  # below threshold, should be pruned
    assert w.get("foo", "bar") == 0.0  # pruned as zero
    assert w.num_nonzero["pair"] == 0


def test_small_nonzero_not_pruned() -> None:
    """Values >= 1e-12 are not pruned."""
    w = SparseLexicalWeights.empty()
    w.set_pair("foo", "bar", 1e-6)
    assert w.get("foo", "bar") == pytest.approx(1e-6)
    assert w.num_nonzero["pair"] == 1


def test_update_pair() -> None:
    w = SparseLexicalWeights.empty()
    w.update_pair("foo", "bar", 1.0)
    assert w.get("foo", "bar") == 1.0
    w.update_pair("foo", "bar", 0.5)
    assert w.get("foo", "bar") == 1.5


def test_update_surface() -> None:
    w = SparseLexicalWeights.empty()
    w.update_surface("bar", 2.0)
    assert w.get_surface("bar") == 2.0


def test_update_reading() -> None:
    w = SparseLexicalWeights.empty()
    w.update_reading("foo", 3.0)
    assert w.get_reading("foo") == 3.0


def test_l2_regularization() -> None:
    w = SparseLexicalWeights.empty()
    w.set_pair("foo", "bar", 1.0)
    w.set_surface("bar", 2.0)
    w.set_reading("foo", 3.0)
    w.apply_l2(0.1)  # w -= 0.1 * w
    assert w.get("foo", "bar") == pytest.approx(0.9)
    assert w.get_surface("bar") == pytest.approx(1.8)
    assert w.get_reading("foo") == pytest.approx(2.7)


def test_dump_load_roundtrip() -> None:
    original = SparseLexicalWeights.empty()
    original.set_pair("foo", "bar", 1.5)
    original.set_pair("hello", "world", -0.5)
    original.set_surface("baz", 2.0)
    original.set_reading("qux", 3.0)

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "weights"
        original.dump(base)

        # Verify both files exist
        assert base.with_suffix(".sparse.json").exists()
        assert base.with_suffix(".sparse.bin").exists()

        loaded = SparseLexicalWeights.load(base)
        assert loaded.get("foo", "bar") == pytest.approx(1.5)
        assert loaded.get("hello", "world") == pytest.approx(-0.5)
        assert loaded.get("foo", "baz") == pytest.approx(2.0)  # backoff to surface
        assert loaded.get("qux", "unknown") == pytest.approx(3.0)  # backoff to reading
        assert loaded.get("unknown", "unknown") == 0.0


def test_dump_load_empty() -> None:
    original = SparseLexicalWeights.empty()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "empty"
        original.dump(base)
        loaded = SparseLexicalWeights.load(base)
        assert loaded.get("foo", "bar") == 0.0
        assert loaded.num_nonzero == {"pair": 0, "surface": 0, "reading": 0}


def test_to_dict_from_dict() -> None:
    original = SparseLexicalWeights.empty()
    original.set_pair("foo", "bar", 1.5)
    original.set_surface("baz", 2.0)
    data = original.to_dict()
    assert isinstance(data, dict)
    loaded = SparseLexicalWeights.from_dict(data)
    assert loaded.get("foo", "bar") == 1.5
    assert loaded.get("foo", "baz") == 2.0


def test_num_nonzero() -> None:
    w = SparseLexicalWeights.empty()
    w.set_pair("a", "b", 1.0)
    w.set_pair("c", "d", 2.0)
    w.set_surface("e", 3.0)
    stats = w.num_nonzero
    assert stats["pair"] == 2
    assert stats["surface"] == 1
    assert stats["reading"] == 0


# Need pytest.approx
import pytest  # noqa: E402 (must be after other imports)

__all__: list[str] = []