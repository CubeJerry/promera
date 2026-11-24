"""Exercise the real refold guard through a minimal base implementation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from promera_angles.refold import UnconditionedRefoldMixin


class Model:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def pairformer_forward(self, features: dict[str, Any], *, marker: str) -> str:
        self.calls.append(features)
        return marker


class RefoldBase:
    def _refold_batch(
        self,
        model: Model,
        features: dict[str, Any],
        *,
        downstream_error: bool = False,
    ) -> str:
        result = model.pairformer_forward(features, marker="unrestrained-refold")
        if downstream_error:
            raise ValueError("downstream refold failure")
        return result


class GuardedRefold(UnconditionedRefoldMixin, RefoldBase):
    pass


def test_fresh_unrestrained_features_reach_refolding_and_method_is_restored() -> None:
    model = Model()
    original = model.pairformer_forward
    features = {
        "restype": np.zeros(7),
        "distogram_mask": np.zeros((7, 7), dtype=bool),
        "is_epitope": np.zeros(7, dtype=bool),
    }
    assert GuardedRefold()._refold_batch(model, features) == "unrestrained-refold"
    assert model.calls == [features]
    assert model.pairformer_forward == original
    assert not features["distogram_mask"].any()
    assert not features["is_epitope"].any()


@pytest.mark.parametrize("contaminated_key", ["distogram_mask", "is_epitope"])
def test_generation_guidance_cannot_enter_refolding(contaminated_key: str) -> None:
    model = Model()
    original = model.pairformer_forward
    features = {
        "distogram_mask": np.zeros((7, 7), dtype=bool),
        "is_epitope": np.zeros(7, dtype=bool),
    }
    features[contaminated_key].flat[0] = True
    with pytest.raises(RuntimeError, match="validation contamination"):
        GuardedRefold()._refold_batch(model, features)
    assert model.calls == []
    assert model.pairformer_forward == original
    # An accidental guide is rejected, rather than silently hidden from accounting.
    assert features[contaminated_key].any()


def test_refold_guard_also_accepts_absent_guidance_features() -> None:
    model = Model()
    features = {"restype": np.zeros(7)}
    assert GuardedRefold()._refold_batch(model, features) == "unrestrained-refold"


def test_original_forward_is_restored_after_downstream_refold_error() -> None:
    model = Model()
    original = model.pairformer_forward
    with pytest.raises(ValueError, match="downstream refold failure"):
        GuardedRefold()._refold_batch(model, {"restype": np.zeros(7)}, downstream_error=True)
    assert len(model.calls) == 1
    assert model.pairformer_forward == original
