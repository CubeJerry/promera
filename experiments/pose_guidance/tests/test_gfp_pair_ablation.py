"""CPU checks for the fixed GFP-only ablation."""

import numpy as np
import pytest

from promera_angles.gfp_pair_ablation import (
    GFP,
    GFP_PAIR_COUNTS,
    retain_gfp_pairs,
    validate_gfp_sequences,
)
from promera_angles.run import split_jobs
from promera_angles.settings import study_arms, validate_settings


def test_gfp_scope_and_budget(tmp_path):
    validate_gfp_sequences({"A": GFP})
    for sequences in ({"A": "A" * 238}, {"A": GFP, "B": GFP}, {"A": GFP[:100]}):
        with pytest.raises(ValueError):
            validate_gfp_sequences(sequences)
    spec = validate_settings(dict(
        enabled=True, gfp_pair_ablation=True, elevation_interval_deg=[70, 90]
    ))
    arms = study_arms(spec)
    assert arms == list(GFP_PAIR_COUNTS)
    jobs = split_jobs(tmp_path, 200, 10, False, True, 0, arms=arms)
    assert len(jobs) == 50
    assert sum(job["count"] for job in jobs) == 1000
    for arm in arms:
        rows = [job for job in jobs if job["arm"] == arm]
        assert [job["seed"] for job in rows] == list(range(10))
        assert [job["start"] for job in rows] == list(range(0, 200, 20))
        assert all(job["gfp_pair_arm"] == arm for job in rows)


def test_nested_subsets_keep_bins_and_symmetry():
    records = [
        {"target_token": i, "binder_token": 4 + (i + offset) % 4}
        for offset in range(4) for i in range(4)
    ]
    bins = np.zeros((8, 8), dtype=int)
    mask = np.zeros((8, 8), dtype=bool)
    for i, row in enumerate(records):
        a, b = row["target_token"], row["binder_token"]
        bins[a, b] = bins[b, a] = i + 1
        mask[a, b] = mask[b, a] = True
    for arm, count in GFP_PAIR_COUNTS.items():
        if not count:
            continue
        b, m = bins.copy(), mask.copy()
        assert retain_gfp_pairs(b, m, records, arm) == records[:count]
        assert m.sum() == 2 * count
        assert np.array_equal(m, m.T)
        assert np.array_equal(b[m], bins[m])
        assert not b[~m].any()
