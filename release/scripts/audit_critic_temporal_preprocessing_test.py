# SPDX-License-Identifier: OpenMDW-1.1

from audit_critic_temporal_preprocessing import (
    evaluation_indices,
    evaluation_timestamps,
    training_indices,
    training_timestamps,
)


def test_capped_training_and_evaluation_routes_are_not_equivalent():
    train, effective_fps = training_indices(600, 25.0)
    evaluate = evaluation_indices(600, 25.0)

    assert len(train) == 32
    assert len(evaluate) == 48
    assert train[0] == evaluate[0] == 0
    assert train[-1] == 576
    assert evaluate[-1] == 599
    assert training_timestamps(train, effective_fps)[-1] == 14.879999999999999
    assert evaluation_timestamps(evaluate, 25.0)[-1] == 23.96


def test_short_training_and_evaluation_routes_still_choose_different_frames():
    train, effective_fps = training_indices(57, 25.0)
    evaluate = evaluation_indices(57, 25.0)

    assert effective_fps == 25.0 / 12.0
    assert train == [0, 12, 24, 36, 48]
    assert evaluate == [0, 19, 37, 56]


def test_training_sampler_does_not_repeat_indices_when_capped():
    train, _ = training_indices(10_000, 25.0)

    assert len(train) == 32
    assert train == sorted(set(train))
