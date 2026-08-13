# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for the frozen shortcut-battery extension implementation.

Covers: fold-assignment determinism, row feature-index logic on synthetic
clips (prefix / mask / shuffle edge cases: N=1, N=2, N<4, m<4), round-half
rules, the abort-on-fingerprint-mismatch path, seeded draw determinism, slot
permutation mechanics, and rank equivalence with the pilot.
"""

from __future__ import annotations

import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_temporal_shortcuts as pilot  # noqa: E402
import run_shortcut_battery_extension as battery  # noqa: E402


# --- fold assignment -------------------------------------------------------


def test_fold_assignment_matches_protocol_rule() -> None:
    scenes = [f"scene_{i:04d}" for i in range(392)]
    presented = list(reversed(scenes)) * 4  # scrambled order with duplicates
    mapping = battery.assign_folds(presented)
    expected_order = sorted(set(presented))
    rng = random.Random(20260811)
    rng.shuffle(expected_order)
    expected = {scene: index % 5 for index, scene in enumerate(expected_order)}
    assert mapping == expected


def test_fold_assignment_deterministic_and_balanced() -> None:
    scenes = [f"s{i}" for i in range(392)]
    a = battery.assign_folds(scenes)
    b = battery.assign_folds(list(reversed(scenes)))
    assert a == b
    counts = [sum(1 for fold in a.values() if fold == f) for f in range(5)]
    assert counts == [79, 79, 78, 78, 78]


# --- prefix rows (8/9) ------------------------------------------------------


def test_prefix_indices_repeat_when_m_below_4() -> None:
    # N=1 -> m=1 for both fractions -> four copies of frame 0.
    assert battery.prefix_frame_indices(1, 0.25) == [0, 0, 0, 0]
    assert battery.prefix_frame_indices(1, 0.50) == [0, 0, 0, 0]
    # N=2: 25% -> m=max(1,0)=1; 50% -> m=1.
    assert battery.prefix_frame_indices(2, 0.25) == [0, 0, 0, 0]
    assert battery.prefix_frame_indices(2, 0.50) == [0, 0, 0, 0]
    # N=3, 50% -> m=1; N=6, 50% -> m=3 -> [0, round(2/3), round(4/3), 2] = [0,1,1,2]
    assert battery.prefix_frame_indices(3, 0.50) == [0, 0, 0, 0]
    assert battery.prefix_frame_indices(6, 0.50) == [0, 1, 1, 2]
    # N=8, 25% -> m=2 -> [0, round(1/3), round(2/3), 1] = [0,0,1,1]
    assert battery.prefix_frame_indices(8, 0.25) == [0, 0, 1, 1]


def test_prefix_indices_equal_spacing_when_m_at_least_4() -> None:
    # N=16, 25% -> m=4 -> exactly [0,1,2,3]
    assert battery.prefix_frame_indices(16, 0.25) == [0, 1, 2, 3]
    # N=13, 50% -> m=6 -> [0, round(5/3)=2, round(10/3)=3, 5]
    assert battery.prefix_frame_indices(13, 0.50) == [0, 2, 3, 5]
    # No index may reach m; indices nondecreasing.
    for n in range(1, 700):
        for fraction in (0.25, 0.50):
            m = max(1, math.floor(fraction * n))
            indices = battery.prefix_frame_indices(n, fraction)
            assert all(0 <= i < m for i in indices)
            assert indices == sorted(indices)
            assert indices[0] == 0 and indices[-1] == m - 1


def test_prefix_round_half_cases_cannot_occur() -> None:
    """j*(m-1)/3 has fractional part in {0, 1/3, 2/3}; exact .5 never occurs,
    so Python's round (half-to-even) equals round-half-up here for every m."""
    for m in range(1, 5001):
        for j in range(4):
            exact = Fraction(j * (m - 1), 3)
            assert exact - Fraction(math.floor(exact)) != Fraction(1, 2)
            half_up = math.floor(exact + Fraction(1, 2))
            assert round(j * (m - 1) / 3) == half_up


# --- endpoint-masked rows (10/11) -------------------------------------------


def test_masked_terminal_index_rules() -> None:
    # N=10: 10% -> k=1 -> index 8; 25% -> k=ceil(2.5)=3 -> index 6.
    assert battery.masked_terminal_index(10, 0.10) == 8
    assert battery.masked_terminal_index(10, 0.25) == 6
    # N=2, 10% -> k=1 -> index 0 (single surviving frame).
    assert battery.masked_terminal_index(2, 0.10) == 0
    assert battery.masked_terminal_index(2, 0.25) == 0
    # N=4, 25% -> k=1 -> index 2; N=5, 25% -> k=2 -> index 2.
    assert battery.masked_terminal_index(4, 0.25) == 2
    assert battery.masked_terminal_index(5, 0.25) == 2
    # N=40, 10% -> k=4 -> index 35.
    assert battery.masked_terminal_index(40, 0.10) == 35


def test_masked_terminal_index_aborts_when_no_frame_survives() -> None:
    with pytest.raises(ValueError):
        battery.masked_terminal_index(1, 0.10)
    with pytest.raises(ValueError):
        battery.masked_terminal_index(1, 0.25)
    with pytest.raises(ValueError):
        battery.masked_terminal_index(0, 0.10)


# --- seeded draws ------------------------------------------------------------


def test_random_frame_indices_deterministic_and_in_range() -> None:
    counts = [1, 2, 3, 600, 47, 1]
    a = battery.draw_random_frame_indices(counts)
    b = battery.draw_random_frame_indices(counts)
    assert a == b
    assert all(0 <= index < n for index, n in zip(a, counts))
    rng = random.Random(20260811)
    assert a == [rng.randrange(n) for n in counts]
    # N=1 always draws frame 0.
    assert a[0] == 0 and a[-1] == 0


def test_clip_permutation_deterministic_valid_and_id_sensitive() -> None:
    p2 = battery.clip_permutation("traversal_critic_train/media/video_00001.mp4", 2)
    p4 = battery.clip_permutation("traversal_critic_train/media/video_00001.mp4", 4)
    assert p2 == battery.clip_permutation("traversal_critic_train/media/video_00001.mp4", 2)
    assert sorted(p2) == [0, 1] and sorted(p4) == [0, 1, 2, 3]
    others = {tuple(battery.clip_permutation(f"traversal_critic_val/media/video_{i:05d}.mp4", 4)) for i in range(1, 40)}
    assert len(others) > 1  # ID-sensitive


def test_label_permutation_deterministic() -> None:
    a = battery.label_permutation(1568)
    assert a == battery.label_permutation(1568)
    assert sorted(a) == list(range(1568))
    assert a != list(range(1568))


def test_bootstrap_draws_shape_range_deterministic() -> None:
    a = battery.bootstrap_scene_draws(108, n_draws=50)
    b = battery.bootstrap_scene_draws(108, n_draws=50)
    assert a.shape == (50, 108)
    assert a.min() >= 0 and a.max() < 108
    assert np.array_equal(a, b)
    rng = random.Random(20260811)
    assert a[0].tolist() == [rng.randrange(108) for _ in range(108)]


# --- fingerprint replay -------------------------------------------------------


def test_verify_fingerprints_passes_on_match() -> None:
    fingerprints = {"train": "aa", "val": "bb"}
    battery.verify_fingerprints(dict(fingerprints), dict(fingerprints))


def test_verify_fingerprints_aborts_on_mismatch() -> None:
    with pytest.raises(SystemExit) as excinfo:
        battery.verify_fingerprints({"train": "aa", "val": "XX"}, {"train": "aa", "val": "bb"})
    assert "MEDIA FINGERPRINT MISMATCH" in str(excinfo.value)
    with pytest.raises(SystemExit):
        battery.verify_fingerprints({"train": "ZZ", "val": "bb"}, {"train": "aa", "val": "bb"})
    with pytest.raises(SystemExit):
        battery.verify_fingerprints({}, {"train": "aa", "val": "bb"})


# --- slot permutation mechanics -----------------------------------------------


def test_apply_slot_permutation_moves_slots_only() -> None:
    frames = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    perms = np.asarray([[2, 0, 1], [0, 1, 2]])
    out = battery.apply_slot_permutation(frames, perms)
    assert out.shape == (2, 12)
    assert np.array_equal(out[0], np.concatenate([frames[0, 2], frames[0, 0], frames[0, 1]]))
    assert np.array_equal(out[1], frames[1].reshape(-1))  # identity permutation
    # Pixel content unchanged: sorted values per clip identical.
    assert np.array_equal(np.sort(out[0]), np.sort(frames[0].reshape(-1)))


def test_apply_slot_permutation_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        battery.apply_slot_permutation(np.zeros((2, 3, 4)), np.zeros((2, 2), dtype=np.int64))


# --- rounded-metric rule and ranks ---------------------------------------------


def test_rounded_metric_rule_matches_pilot() -> None:
    prediction = np.asarray([0.4, 5.7, 2.5, 3.5, 1.49, -1.0])
    rounded = np.clip(np.round(prediction), 1, 5)
    # numpy half-to-even at .5, then clip to [1, 5] - frozen pilot behavior.
    assert rounded.tolist() == [1.0, 5.0, 2.0, 4.0, 1.0, 1.0]
    labels = np.asarray([1.0, 5.0, 2.0, 4.0, 1.0, 1.0])
    metrics = pilot.metrics(prediction, labels)
    assert metrics["pearson_rounded_clipped"] == pytest.approx(1.0)


def test_average_ranks_fast_matches_pilot() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        values = rng.integers(0, 6, size=97).astype(np.float64)  # heavy ties
        assert np.allclose(battery.average_ranks_fast(values), pilot._average_ranks(values))
    values = rng.normal(size=432)
    assert np.allclose(battery.average_ranks_fast(values), pilot._average_ranks(values))


# --- estimator determinism -------------------------------------------------------


def test_fit_row_deterministic_and_uses_protocol_folds() -> None:
    rng = np.random.default_rng(1)
    scenes = [f"s{i % 40}" for i in range(200)]
    fold_map = battery.assign_folds(scenes)
    fold_ids = np.asarray([fold_map[s] for s in scenes])
    Xtr = rng.normal(size=(200, 6))
    ytr = Xtr[:, 0] * 2.0 + rng.normal(scale=0.1, size=200)
    Xva = rng.normal(size=(50, 6))
    cv_a, lam_a, pred_a = battery.fit_row(Xtr, ytr, fold_ids, Xva)
    cv_b, lam_b, pred_b = battery.fit_row(Xtr, ytr, fold_ids, Xva)
    assert cv_a == cv_b and lam_a == lam_b
    assert np.array_equal(pred_a, pred_b)
    assert lam_a in battery.LAMBDAS
    # Scenes share folds: every clip of a scene sits in one fold.
    for scene in set(scenes):
        folds = {fold_ids[i] for i, s in enumerate(scenes) if s == scene}
        assert len(folds) == 1
