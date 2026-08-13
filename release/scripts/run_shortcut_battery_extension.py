# SPDX-License-Identifier: OpenMDW-1.1
# Public release copy (stage A); private-path defaults replaced by <REPO>/<DATA> placeholders — see RELEASE_MANIFEST.md
"""Execute the prospectively frozen shortcut-battery extension.

Authority: docs/reviews/shortcut_battery_extension_protocol_2026-08-11.md
(frozen 2026-08-11 before any extended-battery row was computed).  This script
computes the 16 preregistered battery rows (rows 1-11, four order-shuffled
companions, one shuffled-label control) plus the reprinted post-hoc pilot row,
on the unchanged 1,568-clip / 432-clip scene-disjoint split with frozen
labeler-v3 labels.

Data-loading, labeling, scene-grouping, fingerprinting, ridge, and metric
machinery is imported READ-ONLY from the pilot implementation
``scripts/evaluate_temporal_shortcuts.py`` (SHA-256
0e7f422c5a3829fd543c2ffe2a3a8a92a1c161df7df2456361e319060957a79e at pilot
time); that file is never edited.

CPU-only by construction: run with CUDA_VISIBLE_DEVICES="" and under
``nice -n 19 ionice -c3``; decoding uses at most --workers (default 4)
processes.

Example::

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= nice -n 19 ionice -c3 \
      <REPO>/cosmos-framework/.venv/bin/python \
      scripts/run_shortcut_battery_extension.py \
        --data-root <DATA>/traversal-critic/data/critic_v5 \
        --dataset-audit <DATA>/traversal-critic/data/critic_v5_dataset_audit.json \
        --critic-selection <DATA>/traversal-critic/data/critic_v5_selected.json \
        --pilot-results autoresearch/run-260811-1753/temporal_shortcut_results.json \
        --pilot-cache autoresearch/run-260811-1753/temporal_shortcut_features.npz \
        --protocol docs/reviews/shortcut_battery_extension_protocol_2026-08-11.md \
        --out-dir autoresearch/run-260812-HHMM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_temporal_shortcuts as pilot  # noqa: E402  (read-only reuse of pilot machinery)

MASTER_SEED = 20260811
THUMBNAIL_SIZE = 8
FRAME_DIMS = THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3
LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
N_BOOTSTRAP = 10_000
PREFIX_K = 4

# Frozen references (protocol "Frozen reporting rules").
CRITIC_REFERENCE_PEARSON = 0.5646  # mismatched-interface historical record (E5 audit)
PROBE_REFERENCE_PEARSON = 0.7053  # frozen-tower probe result of record

# Pilot authorities (protocol "Fixed data, labels, and split").
EXPECTED_FINGERPRINTS = {
    "train": "77fa2c4a4eb9544159a41f457b8b07a8b18c92f1933bab5fd2a8ac49bfbbb698",
    "val": "eaa47ef334a285bcc379dc396bdc1f8ace4c7c319337a7c3670f4686b102bba4",
}
EXPECTED_PILOT_CACHE_SHA256 = "f8a9b5f81c94bca5bd240e0af8a31cf6692aab8808fcd36bb83f7636144fb852"
EXPECTED_PILOT_RESULTS_SHA256 = "02aa81fd26857744b47e49c769e256354909d27e8d380f1ab29b34073f893ca2"
EXPECTED_N_TRAIN = 1568
EXPECTED_N_VAL = 432
EXPECTED_N_TRAIN_SCENES = 392
EXPECTED_N_VAL_SCENES = 108

# Sanity gate (execution directive of 2026-08-12).
PILOT_ROW5_PEARSON = 0.6827086890665492
SANITY_ROW5_TOLERANCE = 0.02
SANITY_SHUFFLED_LABEL_MAX = 0.15


# ---------------------------------------------------------------------------
# Seeded, protocol-frozen index logic (pure functions; unit-tested).
# ---------------------------------------------------------------------------


def assign_folds(scene_ids: list[str], n_folds: int = N_FOLDS, seed: int = MASTER_SEED) -> dict[str, int]:
    """Protocol fold rule: sorted unique scenes, random.Random(seed) shuffle, round-robin."""
    unique = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    return {scene: index % n_folds for index, scene in enumerate(unique)}


def prefix_frame_indices(n_frames: int, fraction: float) -> list[int]:
    """Rows 8/9: K=4 equally spaced indices within the prefix of m frames.

    m = max(1, floor(fraction * N)); indices round(j*(m-1)/3), j=0..3 (may repeat
    when m < 4).  No index at or beyond m may be produced.
    """
    if n_frames < 1:
        raise ValueError(f"invalid frame count {n_frames}")
    m = max(1, math.floor(fraction * n_frames))
    indices = [round(j * (m - 1) / 3) for j in range(PREFIX_K)]
    if any(index < 0 or index >= m for index in indices):
        raise ValueError(f"prefix index out of range: n={n_frames} fraction={fraction} -> {indices}")
    return indices


def masked_terminal_index(n_frames: int, fraction: float) -> int:
    """Rows 10/11: drop final k = max(1, ceil(fraction*N)) frames; terminal is N-1-k."""
    if n_frames < 1:
        raise ValueError(f"invalid frame count {n_frames}")
    k = max(1, math.ceil(fraction * n_frames))
    index = n_frames - 1 - k
    if index < 0:
        raise ValueError(f"endpoint mask removes every frame: n={n_frames} fraction={fraction} k={k}")
    return index


def clip_permutation(clip_id: str, n_slots: int, seed: int = MASTER_SEED) -> list[int]:
    """Rows 12-15: per-clip slot permutation derived from the master seed and clip ID."""
    digest = hashlib.sha256(f"{seed}|{clip_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    permutation = list(range(n_slots))
    rng.shuffle(permutation)
    return permutation


def draw_random_frame_indices(frame_counts: list[int], seed: int = MASTER_SEED) -> list[int]:
    """Row 7: one uniform frame index per clip, drawn in dataset order (train then val)."""
    rng = random.Random(seed)
    return [rng.randrange(n) for n in frame_counts]


def label_permutation(n_labels: int, seed: int = MASTER_SEED) -> list[int]:
    """Row 16: single training-label permutation from the master seed."""
    rng = random.Random(seed)
    permutation = list(range(n_labels))
    rng.shuffle(permutation)
    return permutation


def bootstrap_scene_draws(n_scenes: int, n_draws: int = N_BOOTSTRAP, seed: int = MASTER_SEED) -> np.ndarray:
    """Scene-clustered bootstrap draw matrix: n_draws x n_scenes scene indices."""
    rng = random.Random(seed)
    return np.asarray(
        [[rng.randrange(n_scenes) for _ in range(n_scenes)] for _ in range(n_draws)],
        dtype=np.int32,
    )


def verify_fingerprints(computed: dict[str, str], expected: dict[str, str]) -> None:
    """Mandatory fingerprint replay: abort with a clear report on any mismatch."""
    mismatches = {
        split: {"expected": expected[split], "computed": computed.get(split, "<missing>")}
        for split in expected
        if computed.get(split) != expected[split]
    }
    if mismatches:
        raise SystemExit(
            "MEDIA FINGERPRINT MISMATCH - aborting before any fitting per protocol "
            "(docs/reviews/shortcut_battery_extension_protocol_2026-08-11.md, 'Fixed data, labels, and split'): "
            + json.dumps(mismatches, indent=2)
        )


def apply_slot_permutation(frames: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    """Reorder per-clip frame slots.

    ``frames`` has shape (n_clips, n_slots, frame_dims); ``permutations`` has
    shape (n_clips, n_slots).  Row i of the output concatenates
    frames[i, permutations[i][0]], frames[i, permutations[i][1]], ...
    Pixel content and dimensionality are unchanged; only slot order moves.
    """
    n_clips, n_slots, dims = frames.shape
    if permutations.shape != (n_clips, n_slots):
        raise ValueError(f"permutation shape {permutations.shape} != {(n_clips, n_slots)}")
    out = frames[np.arange(n_clips)[:, None], permutations, :]
    return out.reshape(n_clips, n_slots * dims)


def average_ranks_fast(values: np.ndarray) -> np.ndarray:
    """Vectorized average ranks, exactly matching the pilot's _average_ranks."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    group = np.empty(len(values), dtype=np.int64)
    if len(values):
        group[0] = 0
        np.cumsum(sorted_values[1:] != sorted_values[:-1], out=group[1:])
    counts = np.bincount(group)
    ends = np.cumsum(counts)
    starts = ends - counts
    group_rank = (starts + ends - 1) / 2.0
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = group_rank[group]
    return ranks


# ---------------------------------------------------------------------------
# Estimator (protocol-frozen; reuses the pilot's ridge/standardization).
# ---------------------------------------------------------------------------


def fit_row(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    fold_ids: np.ndarray,
    Xva: np.ndarray,
    lambdas: tuple[float, ...] = LAMBDAS,
) -> tuple[dict[str, float], float, np.ndarray]:
    """Train-only five-fold scene-grouped lambda selection, refit, one validation pass."""
    cv: dict[str, float] = {}
    for lam in lambdas:
        oof = np.empty(len(ytr), dtype=np.float64)
        for fold in range(N_FOLDS):
            train = fold_ids != fold
            heldout = ~train
            mean, std = pilot._standardize_fit(Xtr[train])
            coef = pilot.ridge_fit(pilot._design(Xtr[train], mean, std), ytr[train], lam)
            oof[heldout] = pilot._design(Xtr[heldout], mean, std) @ coef
        cv[str(lam)] = pilot.pearson(oof, ytr)
    # Pilot-inherited tie-break: maximum pooled OOF Pearson, smallest lambda on ties.
    selected = max(lambdas, key=lambda value: (cv[str(value)], -value))
    mean, std = pilot._standardize_fit(Xtr)
    coef = pilot.ridge_fit(pilot._design(Xtr, mean, std), ytr, selected)
    prediction = pilot._design(Xva, mean, std) @ coef
    return cv, float(selected), prediction


def bootstrap_intervals(
    prediction: np.ndarray,
    labels: np.ndarray,
    draw_item_indices: list[np.ndarray],
) -> dict[str, Any]:
    """Descriptive 95% percentile intervals from the saved scene-clustered draws."""
    rounded = np.clip(np.round(prediction), 1, 5)
    stats = {"pearson_continuous": [], "spearman_continuous": [], "pearson_rounded_clipped": []}
    for idx in draw_item_indices:
        y = labels[idx]
        p = prediction[idx]
        stats["pearson_continuous"].append(pilot.pearson(p, y))
        stats["spearman_continuous"].append(pilot.pearson(average_ranks_fast(p), average_ranks_fast(y)))
        stats["pearson_rounded_clipped"].append(pilot.pearson(rounded[idx], y))
    out: dict[str, Any] = {}
    for name, values in stats.items():
        arr = np.asarray(values, dtype=np.float64)
        low, high = np.percentile(arr, [2.5, 97.5])
        out[name] = {
            "ci95_low": float(low),
            "ci95_high": float(high),
            "bootstrap_mean": float(arr.mean()),
            "n_draws": int(len(arr)),
        }
    return out


# ---------------------------------------------------------------------------
# Decoding (CPU-only, <= --workers processes).
# ---------------------------------------------------------------------------


def _decode_worker(task: tuple[str, tuple[int, ...]]) -> tuple[str, int, float, dict[int, list[float]]]:
    """Decode selected frame indices of one clip to 8x8 bilinear thumbnails.

    Thumbnail pipeline is byte-identical to the pilot's _extract: torchcodec RGB
    frame -> uint8 HWC -> PIL bilinear 8x8 -> float64/255 flattened (192 dims).
    """
    from PIL import Image
    from torchcodec.decoders import VideoDecoder

    path, indices = task
    decoder = VideoDecoder(path)
    n_frames = int(decoder.metadata.num_frames or 0)
    fps = float(decoder.metadata.average_fps or 0.0)
    if n_frames <= 0 or fps <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    features: dict[int, list[float]] = {}
    if indices:
        wanted = sorted(set(indices))
        if wanted[0] < 0 or wanted[-1] >= n_frames:
            raise ValueError(f"frame index out of range for {path}: {wanted}")
        frames = decoder.get_frames_at(indices=wanted).data
        for slot, index in enumerate(wanted):
            rgb = frames[slot].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
            thumb = Image.fromarray(rgb).resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.BILINEAR)
            features[index] = (np.asarray(thumb, dtype=np.float64).reshape(-1) / 255.0).tolist()
    return path, n_frames, fps, features


def _run_pool(
    tasks: list[tuple[str, tuple[int, ...]]], workers: int, stage: str
) -> dict[str, tuple[int, float, dict[int, np.ndarray]]]:
    results: dict[str, tuple[int, float, dict[int, np.ndarray]]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for done, (path, n_frames, fps, features) in enumerate(pool.map(_decode_worker, tasks, chunksize=16), start=1):
            results[path] = (
                n_frames,
                fps,
                {index: np.asarray(vec, dtype=np.float64) for index, vec in features.items()},
            )
            if done % 200 == 0 or done == len(tasks):
                print(f"[battery] {stage} {done}/{len(tasks)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Battery assembly.
# ---------------------------------------------------------------------------


def _clip_id(row: dict[str, Any], data_root: Path) -> str:
    return str(Path(row["media"]).resolve().relative_to(data_root))


def build_battery(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pilot_results_path = Path(args.pilot_results).expanduser().resolve()
    pilot_cache_path = Path(args.pilot_cache).expanduser().resolve()
    protocol_path = Path(args.protocol).expanduser().resolve()
    dataset_audit_path = Path(args.dataset_audit).expanduser().resolve()
    critic_selection_path = Path(args.critic_selection).expanduser().resolve()

    # --- Data authorities -------------------------------------------------
    audit_payload = json.loads(dataset_audit_path.read_text())
    if not audit_payload.get("passed"):
        raise SystemExit(f"dataset audit is not passing: {dataset_audit_path}")
    if Path(audit_payload["_meta"]["dataset_root"]).resolve() != data_root:
        raise SystemExit("dataset audit does not bind --data-root")
    pilot_results_sha = pilot.sha256(pilot_results_path)
    if pilot_results_sha != EXPECTED_PILOT_RESULTS_SHA256:
        raise SystemExit(f"pilot results SHA-256 mismatch: {pilot_results_sha}")
    pilot_cache_sha = pilot.sha256(pilot_cache_path)
    if pilot_cache_sha != EXPECTED_PILOT_CACHE_SHA256:
        raise SystemExit(f"pilot feature-cache SHA-256 mismatch: {pilot_cache_sha}")
    pilot_results = json.loads(pilot_results_path.read_text())

    # --- Load rows and replay media fingerprints (abort on mismatch) ------
    train_rows = pilot._load_original_rows(data_root, "train")
    val_rows = pilot._load_original_rows(data_root, "val")
    print(f"[battery] loaded {len(train_rows)} train / {len(val_rows)} val rows", flush=True)
    computed_fingerprints = {
        "train": pilot._media_fingerprint(train_rows),
        "val": pilot._media_fingerprint(val_rows),
    }
    verify_fingerprints(computed_fingerprints, EXPECTED_FINGERPRINTS)
    print("[battery] media fingerprint replay: MATCH", flush=True)

    train_scenes = [row["scene_id"] for row in train_rows]
    val_scenes = [row["scene_id"] for row in val_rows]
    if len(train_rows) != EXPECTED_N_TRAIN or len(val_rows) != EXPECTED_N_VAL:
        raise SystemExit(f"split size mismatch: {len(train_rows)} train / {len(val_rows)} val")
    if len(set(train_scenes)) != EXPECTED_N_TRAIN_SCENES or len(set(val_scenes)) != EXPECTED_N_VAL_SCENES:
        raise SystemExit(f"scene count mismatch: {len(set(train_scenes))} train / {len(set(val_scenes))} val scenes")
    if set(train_scenes) & set(val_scenes):
        raise SystemExit("train/val scenes overlap")

    ytr = np.asarray([row["label"] for row in train_rows], dtype=np.float64)
    yva = np.asarray([row["label"] for row in val_rows], dtype=np.float64)
    all_rows = train_rows + val_rows
    clip_ids = [_clip_id(row, data_root) for row in all_rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise SystemExit("clip IDs are not unique")

    # --- Pass 1: metadata (frame counts, fps) -----------------------------
    meta_tasks = [(str(row["media"]), ()) for row in all_rows]
    meta = _run_pool(meta_tasks, args.workers, "metadata")
    frame_counts = [meta[str(row["media"])][0] for row in all_rows]
    fps_values = [meta[str(row["media"])][1] for row in all_rows]
    durations = np.asarray([n / f for n, f in zip(frame_counts, fps_values)], dtype=np.float64)

    # --- Seeded quantities (all from the master seed), saved before fitting
    fold_by_scene = assign_folds(train_scenes)
    fold_ids = np.asarray([fold_by_scene[scene] for scene in train_scenes], dtype=np.int64)
    random_frame_indices = draw_random_frame_indices(frame_counts)
    perms2 = np.asarray([clip_permutation(clip_id, 2) for clip_id in clip_ids], dtype=np.int64)
    perms4 = np.asarray([clip_permutation(clip_id, 4) for clip_id in clip_ids], dtype=np.int64)
    train_label_perm = np.asarray(label_permutation(len(ytr)), dtype=np.int64)
    sorted_val_scenes = sorted(set(val_scenes))
    draw_matrix = bootstrap_scene_draws(len(sorted_val_scenes))

    prefix25 = [prefix_frame_indices(n, 0.25) for n in frame_counts]
    prefix50 = [prefix_frame_indices(n, 0.50) for n in frame_counts]
    mask10 = [masked_terminal_index(n, 0.10) for n in frame_counts]
    mask25 = [masked_terminal_index(n, 0.25) for n in frame_counts]

    seed_artifact_path = out_dir / "shortcut_battery_seed_index_artifact.npz"
    np.savez_compressed(
        seed_artifact_path,
        master_seed=np.int64(MASTER_SEED),
        clip_ids=np.asarray(clip_ids),
        n_train=np.int64(len(train_rows)),
        frame_counts=np.asarray(frame_counts, dtype=np.int64),
        fps=np.asarray(fps_values, dtype=np.float64),
        train_scene_ids=np.asarray(train_scenes),
        val_scene_ids=np.asarray(val_scenes),
        fold_scene_order=np.asarray(sorted(fold_by_scene, key=lambda s: (fold_by_scene[s], s))),
        train_fold_ids=fold_ids,
        fold_by_scene_keys=np.asarray(sorted(fold_by_scene)),
        fold_by_scene_values=np.asarray([fold_by_scene[s] for s in sorted(fold_by_scene)], dtype=np.int64),
        random_frame_indices=np.asarray(random_frame_indices, dtype=np.int64),
        prefix25_indices=np.asarray(prefix25, dtype=np.int64),
        prefix50_indices=np.asarray(prefix50, dtype=np.int64),
        mask10_terminal_index=np.asarray(mask10, dtype=np.int64),
        mask25_terminal_index=np.asarray(mask25, dtype=np.int64),
        frame_order_permutations_2=perms2,
        frame_order_permutations_4=perms4,
        train_label_permutation=train_label_perm,
        bootstrap_sorted_val_scenes=np.asarray(sorted_val_scenes),
        bootstrap_draw_matrix=draw_matrix,
    )
    seed_artifact_sha = pilot.sha256(seed_artifact_path)
    print(f"[battery] seed/index artifact saved: {seed_artifact_path} ({seed_artifact_sha})", flush=True)

    # --- Pass 2: decode all needed frames ---------------------------------
    needed: list[tuple[str, tuple[int, ...]]] = []
    for position, row in enumerate(all_rows):
        n = frame_counts[position]
        indices = {0, n - 1, random_frame_indices[position], mask10[position], mask25[position]}
        indices.update(prefix25[position])
        indices.update(prefix50[position])
        needed.append((str(row["media"]), tuple(sorted(indices))))
    decoded = _run_pool(needed, args.workers, "decode")

    def frame(position: int, index: int) -> np.ndarray:
        return decoded[str(all_rows[position]["media"])][2][index]

    n_total = len(all_rows)
    first = np.stack([frame(i, 0) for i in range(n_total)])
    terminal = np.stack([frame(i, frame_counts[i] - 1) for i in range(n_total)])
    randomf = np.stack([frame(i, random_frame_indices[i]) for i in range(n_total)])
    masked10 = np.stack([frame(i, mask10[i]) for i in range(n_total)])
    masked25 = np.stack([frame(i, mask25[i]) for i in range(n_total)])
    pref25 = np.stack([np.stack([frame(i, j) for j in prefix25[i]]) for i in range(n_total)])
    pref50 = np.stack([np.stack([frame(i, j) for j in prefix50[i]]) for i in range(n_total)])
    duration_feats = pilot.duration_features(durations)

    features_path = out_dir / "shortcut_battery_features.npz"
    np.savez_compressed(
        features_path,
        durations=durations,
        duration_features=duration_feats,
        first=first,
        terminal=terminal,
        random_frame=randomf,
        masked10=masked10,
        masked25=masked25,
        prefix25=pref25,
        prefix50=pref50,
        labels=np.concatenate([ytr, yva]),
        n_train=np.int64(len(train_rows)),
        train_fingerprint=EXPECTED_FINGERPRINTS["train"],
        val_fingerprint=EXPECTED_FINGERPRINTS["val"],
    )
    features_sha = pilot.sha256(features_path)

    # --- Cross-check fresh terminal/duration against the verified pilot cache
    cache = np.load(pilot_cache_path)
    n_train = len(train_rows)
    cache_terminal = np.concatenate([cache["Xtr"][:, 1:], cache["Xva"][:, 1:]])
    cache_duration = np.concatenate([cache["Xtr"][:, 0], cache["Xva"][:, 0]])
    terminal_max_diff = float(np.abs(terminal - cache_terminal).max())
    duration_max_diff = float(np.abs(durations - cache_duration).max())
    print(
        f"[battery] pilot-cache cross-check: terminal max|diff|={terminal_max_diff:.3e}, "
        f"duration max|diff|={duration_max_diff:.3e}",
        flush=True,
    )

    # --- Assemble the 16 computed rows ------------------------------------
    def split(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X[:n_train], X[n_train:]

    flat25 = pref25.reshape(n_total, PREFIX_K * FRAME_DIMS)
    flat50 = pref50.reshape(n_total, PREFIX_K * FRAME_DIMS)
    first_terminal_frames = np.stack([first, terminal], axis=1)
    row4 = first_terminal_frames.reshape(n_total, 2 * FRAME_DIMS)
    row6 = np.column_stack([duration_feats, row4])
    row12 = apply_slot_permutation(first_terminal_frames, perms2)
    row13 = np.column_stack([duration_feats, row12])
    row14 = apply_slot_permutation(pref25, perms4)
    row15 = apply_slot_permutation(pref50, perms4)

    rows: list[dict[str, Any]] = [
        {"number": 1, "key": "duration_only", "X": duration_feats, "description": "four duration features"},
        {"number": 2, "key": "first_frame_only", "X": first, "description": "frame 0 (192 dims)"},
        {
            "number": 3,
            "key": "terminal_frame_only",
            "X": terminal,
            "description": "frame N-1 (192 dims); recomputes pilot",
        },
        {"number": 4, "key": "first_plus_terminal", "X": row4, "description": "frames 0 and N-1 (384 dims)"},
        {
            "number": 5,
            "key": "duration_plus_terminal",
            "X": np.column_stack([duration_feats, terminal]),
            "description": "duration features plus frame N-1; recomputes pilot",
        },
        {
            "number": 6,
            "key": "duration_first_terminal",
            "X": row6,
            "description": "duration + first + terminal (388 dims)",
        },
        {
            "number": 7,
            "key": "random_single_frame",
            "X": randomf,
            "description": "one uniform master-seed frame per clip (192 dims)",
        },
        {
            "number": 8,
            "key": "onset_prefix_25",
            "X": flat25,
            "description": "K=4 frames in first 25% of decoded frames (768 dims)",
        },
        {
            "number": 9,
            "key": "onset_prefix_50",
            "X": flat50,
            "description": "K=4 frames in first 50% of decoded frames (768 dims)",
        },
        {
            "number": 10,
            "key": "endpoint_masked_last_10",
            "X": masked10,
            "description": "terminal frame at N-1-k, k=max(1,ceil(0.10N)) (192 dims)",
        },
        {
            "number": 11,
            "key": "endpoint_masked_last_25",
            "X": masked25,
            "description": "terminal frame at N-1-k, k=max(1,ceil(0.25N)) (192 dims)",
        },
        {
            "number": 12,
            "key": "shuffled_order_first_plus_terminal",
            "X": row12,
            "description": "row 4 with per-clip slot permutation",
        },
        {
            "number": 13,
            "key": "shuffled_order_duration_first_terminal",
            "X": row13,
            "description": "row 6 with per-clip frame-slot permutation (duration block fixed)",
        },
        {
            "number": 14,
            "key": "shuffled_order_onset_prefix_25",
            "X": row14,
            "description": "row 8 with per-clip slot permutation",
        },
        {
            "number": 15,
            "key": "shuffled_order_onset_prefix_50",
            "X": row15,
            "description": "row 9 with per-clip slot permutation",
        },
        {
            "number": 16,
            "key": "shuffled_label_control",
            "X": row6,
            "description": "row-6 features, train labels permuted once with master seed; negative control",
            "label_permutation": True,
        },
    ]

    # --- Fit every row (validation touched once per row) ------------------
    results_rows: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for row in rows:
        Xtr, Xva = split(np.asarray(row["X"], dtype=np.float64))
        labels_train = ytr[train_label_perm] if row.get("label_permutation") else ytr
        cv, selected_lambda, prediction = fit_row(Xtr, labels_train, fold_ids, Xva)
        predictions[row["key"]] = prediction
        validation = pilot.metrics(prediction, yva)
        results_rows[row["key"]] = {
            "row_number": row["number"],
            "description": row["description"],
            "n_features": int(Xtr.shape[1]),
            "selection": {
                "rule": "maximum pooled scene-grouped five-fold train OOF Pearson; smallest lambda breaks ties",
                "cv_pearson": cv,
                "selected_lambda": selected_lambda,
                "oof_pearson_at_selected_lambda": cv[str(selected_lambda)],
            },
            "validation": validation,
            "descriptive_ratios": {
                "pearson_fraction_of_selected_critic_0.5646_mismatched_interface_historical_record": validation[
                    "pearson_continuous"
                ]
                / CRITIC_REFERENCE_PEARSON,
                "pearson_fraction_of_frozen_tower_probe_0.7053": validation["pearson_continuous"]
                / PROBE_REFERENCE_PEARSON,
            },
        }
        print(
            f"[battery] row {row['number']:>2} {row['key']}: r={validation['pearson_continuous']:.4f} "
            f"lambda={selected_lambda}",
            flush=True,
        )

    # --- Sanity gate (before intervals/finalization) -----------------------
    row5_r = results_rows["duration_plus_terminal"]["validation"]["pearson_continuous"]
    row16_r = results_rows["shuffled_label_control"]["validation"]["pearson_continuous"]
    gate = {
        "row5_recomputed_pearson": row5_r,
        "row5_pilot_pearson": PILOT_ROW5_PEARSON,
        "row5_abs_difference": abs(row5_r - PILOT_ROW5_PEARSON),
        "row5_tolerance": SANITY_ROW5_TOLERANCE,
        "row5_pass": abs(row5_r - PILOT_ROW5_PEARSON) <= SANITY_ROW5_TOLERANCE,
        "shuffled_label_pearson": row16_r,
        "shuffled_label_max": SANITY_SHUFFLED_LABEL_MAX,
        "shuffled_label_pass": abs(row16_r) <= SANITY_SHUFFLED_LABEL_MAX,
    }
    if not (gate["row5_pass"] and gate["shuffled_label_pass"]):
        failure_path = out_dir / "SANITY_GATE_FAILURE.json"
        failure_path.write_text(json.dumps({"sanity_gate": gate, "rows": results_rows}, indent=2) + "\n")
        raise SystemExit(f"SANITY GATE FAILED - stopping before finalization; see {failure_path}")
    print("[battery] sanity gate: PASS", flush=True)

    # --- Bootstrap intervals from the pre-saved draw matrix ----------------
    scene_items = {
        scene: np.asarray([i for i, s in enumerate(val_scenes) if s == scene], dtype=np.int64)
        for scene in sorted_val_scenes
    }
    draw_item_indices = [np.concatenate([scene_items[sorted_val_scenes[s]] for s in draw]) for draw in draw_matrix]
    for row in rows:
        key = row["key"]
        results_rows[key]["bootstrap_ci95_descriptive"] = bootstrap_intervals(predictions[key], yva, draw_item_indices)
        print(f"[battery] intervals done for row {results_rows[key]['row_number']}", flush=True)

    # --- Row 17: post-hoc pilot reprint ------------------------------------
    def pilot_reprint(block: str) -> dict[str, Any]:
        entry = pilot_results[block]
        return {
            "pearson_continuous": entry["validation"]["pearson_continuous"],
            "spearman_continuous": entry["validation"]["spearman_continuous"],
            "pearson_rounded_clipped": entry["validation"]["pearson_rounded_clipped"],
            "selected_lambda": entry["selection"]["selected_lambda"],
            "oof_pearson_at_selected_lambda": entry["selection"]["cv_pearson"][
                str(entry["selection"]["selected_lambda"])
            ],
        }

    post_hoc_pilot = {
        "row_number": 17,
        "status": "post-hoc pilot",
        "note": (
            "Values reprinted from autoresearch/run-260811-1753/temporal_shortcut_results.json "
            "(post-hoc descriptive pilot per protocol; not confirmatory; fold rule differs "
            "from this battery: pilot shuffled scenes with numpy default_rng(0))."
        ),
        "duration_only": pilot_reprint("duration_only"),
        "terminal_frame_only": pilot_reprint("terminal_frame_only"),
        "duration_plus_terminal_frame": pilot_reprint("duration_plus_terminal_frame"),
        "fixed_seed_permuted_training_labels": pilot_reprint("fixed_seed_permuted_training_labels"),
    }

    # --- Persist ------------------------------------------------------------
    script_path = Path(__file__).resolve()
    payload = {
        "_meta": {
            "labeler_version": 3,
            "dynamics": "physics",
            "scene_range": [1000, 1499],
            "data_root": str(data_root),
            "protocol_authority": pilot.file_record(protocol_path),
            "source": pilot.file_record(script_path),
            "pilot_source_module": pilot.file_record(Path(pilot.__file__).resolve()),
            "dataset_audit": pilot.file_record(dataset_audit_path),
            "critic_selection": pilot.file_record(critic_selection_path),
            "pilot_results": pilot.file_record(pilot_results_path),
            "pilot_feature_cache": pilot.file_record(pilot_cache_path),
            "battery_feature_cache": {"path": str(features_path), "sha256": features_sha},
            "seed_index_artifact": {"path": str(seed_artifact_path), "sha256": seed_artifact_sha},
            "media_fingerprints": computed_fingerprints,
            "master_seed": MASTER_SEED,
            "cpu_only": True,
            "pilot_cache_cross_check": {
                "terminal_feature_max_abs_diff": terminal_max_diff,
                "duration_max_abs_diff": duration_max_diff,
            },
            "implementation_notes": [
                "Protocol total says '17 rows (rows 1-11, four order-shuffled companions, one "
                "shuffled-label control)'; that enumeration counts 16 computed rows, so the "
                "reprinted post-hoc pilot is reported as row 17, reconciling the stated total.",
                "Pixel scaling: thumbnails divided by 255 exactly as in the pilot; fold-local "
                "standardization makes this affine scaling mathematically irrelevant to ridge.",
                "Lambda tie-break (unspecified by protocol): maximum pooled OOF Pearson, smallest "
                "lambda on ties - inherited from the pilot; no tie occurred.",
                "Row 7 random draws: random.Random(20260811), one randrange(N) per clip in dataset "
                "order (train rows 1..1568 then val rows 1..432).",
                "Rows 12-15 permutations: per clip, random.Random(int.from_bytes(sha256("
                "f'{20260811}|{clip_id}').digest()[:8],'big')).shuffle over slot indices; clip_id is "
                "the media path relative to the data root; rows 12/13 share the 2-slot permutation, "
                "rows 14/15 share the 4-slot permutation (permutation depends only on master seed, "
                "clip ID, and slot count).",
                "Row 16 label permutation: random.Random(20260811).shuffle over train indices (the "
                "pilot used numpy default_rng(20260811); the protocol specifies only 'the master seed').",
                "Bootstrap: 10,000 draws of 108 scene indices with replacement via "
                "random.Random(20260811).randrange(108), scenes ordered by sorted unique val scene ID; "
                "draw matrix saved to the seed/index artifact before any interval was computed; "
                "intervals are 2.5/97.5 percentiles (numpy linear interpolation), labeled descriptive.",
                "Rounding for 'rounded Pearson' uses numpy round (half-to-even) then clip to [1,5], "
                "as in the pilot; prefix indices use Python round - exact .5 cases cannot occur for "
                "j*(m-1)/3 (fractional parts are thirds), verified by test.",
                "Frames are decoded at native resolution/fps with torchcodec exact-index access over "
                "the source order 0..N-1 (N = decoder.metadata.num_frames), identical to the pilot's "
                "frame basis; all features used for fitting were freshly decoded in this run and the "
                "terminal/duration columns were cross-checked against the verified pilot cache.",
            ],
        },
        "protocol": {
            "authority": "docs/reviews/shortcut_battery_extension_protocol_2026-08-11.md",
            "status": "prospectively frozen 2026-08-11; executed 2026-08-12",
            "estimator": "ridge with fold-local standardization and unpenalized intercept",
            "lambda_grid": list(LAMBDAS),
            "fold_rule": "392 sorted train scene IDs shuffled once with random.Random(20260811), dealt round-robin into 5 folds",
            "selection": "train-only pooled out-of-fold Pearson; refit on all 1,568; single validation pass",
            "references": {
                "selected_v5_critic_pearson": CRITIC_REFERENCE_PEARSON,
                "selected_v5_critic_label": "mismatched-interface historical record (E5 audit)",
                "frozen_tower_probe_pearson": PROBE_REFERENCE_PEARSON,
            },
            "interval_label": "descriptive (validation set participated in the historical checkpoint sweep)",
        },
        "fingerprint_replay": {
            "expected": EXPECTED_FINGERPRINTS,
            "computed": computed_fingerprints,
            "match": True,
        },
        "data": {
            "n_train": len(train_rows),
            "n_train_scenes": len(set(train_scenes)),
            "n_val": len(val_rows),
            "n_val_scenes": len(set(val_scenes)),
            "train_duration_s": {
                "min": float(durations[:n_train].min()),
                "max": float(durations[:n_train].max()),
            },
            "val_duration_s": {
                "min": float(durations[n_train:].min()),
                "max": float(durations[n_train:].max()),
            },
            "frame_count_range": [int(min(frame_counts)), int(max(frame_counts))],
        },
        "sanity_gate": gate,
        "rows": results_rows,
        "post_hoc_pilot": post_hoc_pilot,
    }
    results_path = out_dir / "shortcut_battery_results.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    results_sha = pilot.sha256(results_path)

    report_lines = [
        "# Shortcut-Battery Extension Run Report",
        "",
        "- Executed: 2026-08-12 (protocol frozen 2026-08-11)",
        f"- Authority: {protocol_path}",
        f"- Script: {script_path}",
        "- Fingerprint replay: MATCH (train and val)",
        f"- Sanity gate: PASS (row 5 recomputed {row5_r:.4f} vs pilot {PILOT_ROW5_PEARSON:.4f}; "
        f"shuffled-label control {row16_r:.4f})",
        "",
        "## Artifact SHA-256",
        "",
        f"- `shortcut_battery_results.json`: `{results_sha}`",
        f"- `shortcut_battery_seed_index_artifact.npz`: `{seed_artifact_sha}`",
        f"- `shortcut_battery_features.npz`: `{features_sha}`",
        f"- script `run_shortcut_battery_extension.py`: `{pilot.sha256(script_path)}`",
        f"- pilot results (input authority): `{pilot_results_sha}`",
        f"- pilot feature cache (input authority): `{pilot_cache_sha}`",
        f"- protocol document: `{pilot.sha256(protocol_path)}`",
        "",
        "## Row summary (validation Pearson [95% CI descriptive], Spearman, rounded Pearson, lambda)",
        "",
        "| Row | Key | Pearson | 95% CI | Spearman | Rounded Pearson | Lambda | OOF sel. Pearson |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        entry = results_rows[row["key"]]
        ci = entry["bootstrap_ci95_descriptive"]["pearson_continuous"]
        v = entry["validation"]
        report_lines.append(
            f"| {entry['row_number']} | {row['key']} | {v['pearson_continuous']:.4f} | "
            f"[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}] | {v['spearman_continuous']:.4f} | "
            f"{v['pearson_rounded_clipped']:.4f} | {entry['selection']['selected_lambda']:g} | "
            f"{entry['selection']['oof_pearson_at_selected_lambda']:.4f} |"
        )
    pr = post_hoc_pilot["duration_plus_terminal_frame"]
    report_lines.append(
        f"| 17 | post_hoc_pilot (duration+terminal) | {pr['pearson_continuous']:.4f} | n/a (reprint) | "
        f"{pr['spearman_continuous']:.4f} | {pr['pearson_rounded_clipped']:.4f} | "
        f"{pr['selected_lambda']:g} | {pr['oof_pearson_at_selected_lambda']:.4f} |"
    )
    report_lines += [
        "",
        "Row 17 reprints the post-hoc pilot; its other reprinted rows (duration-only,",
        "terminal-only, permuted-labels) are in `post_hoc_pilot` inside the results JSON.",
        "All intervals are descriptive. Ratios vs critic 0.5646 (mismatched-interface",
        "historical record) and probe 0.7053 are per row in the results JSON.",
        "",
    ]
    (out_dir / "run_report.md").write_text("\n".join(report_lines))
    print(f"[battery] results -> {results_path} ({results_sha})", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--critic-selection", required=True)
    parser.add_argument("--pilot-results", required=True)
    parser.add_argument("--pilot-cache", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers > 4:
        raise SystemExit("worker cap is 4 (frozen PPO matrix is running on this machine)")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('battery is CPU-only: run with CUDA_VISIBLE_DEVICES=""')
    return build_battery(args)


if __name__ == "__main__":
    raise SystemExit(main())
