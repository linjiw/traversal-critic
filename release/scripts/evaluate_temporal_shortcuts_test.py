# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
from evaluate_temporal_shortcuts import (
    _average_ranks,
    duration_features,
    grouped_folds,
    metrics,
)


def test_duration_features_include_polynomial_and_timeout_indicator():
    actual = duration_features(np.asarray([12.0, 24.0]))

    assert actual.tolist() == [
        [0.5, 0.25, 0.125, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]


def test_grouped_folds_never_split_a_scene():
    scenes = np.asarray(["a", "a", "b", "c", "c", "d", "e"])
    fold = grouped_folds(scenes)

    for scene in set(scenes):
        assert len(set(fold[scenes == scene].tolist())) == 1


def test_average_ranks_and_metrics_handle_ties_and_constant_predictions():
    assert _average_ranks(np.asarray([3.0, 1.0, 1.0])).tolist() == [2.0, 0.5, 0.5]
    result = metrics(np.asarray([2.0, 2.0, 2.0, 2.0, 2.0]), np.asarray([1, 2, 3, 4, 5]))

    assert result["pearson_continuous"] == 0.0
    assert result["accuracy_rounded_clipped"] == 0.2
