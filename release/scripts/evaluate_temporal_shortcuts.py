# SPDX-License-Identifier: OpenMDW-1.1
# Public release copy (stage A); private-path defaults replaced by <REPO>/<DATA> placeholders — see RELEASE_MANIFEST.md
"""Measure v5 label predictability from duration and a terminal frame.

This is a deliberately cheap shortcut diagnostic, not a replacement critic.
It fits train-only, scene-grouped ridge probes to (1) video duration and (2)
duration plus an 8x8 RGB terminal-frame thumbnail, then evaluates once on the
unchanged 108-scene validation split.  Original media are used once each; the
balanced training-manifest duplicates are not treated as independent clips.

Run inside the Cosmos environment so torchcodec reads the same MP4 bytes as
the critic paths::

    cd ~/cosmos-framework
    LD_LIBRARY_PATH=$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
      uv run python ~/traversal-critic-research/scripts/evaluate_temporal_shortcuts.py \
        --data-root <DATA>/traversal-critic/data/critic_v5 \
        --dataset-audit <DATA>/traversal-critic/data/critic_v5_dataset_audit.json \
        --critic-selection <DATA>/traversal-critic/data/critic_v5_selected.json \
        --cache /path/to/temporal_shortcut_features.npz \
        --out /path/to/temporal_shortcut_results.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

THUMBNAIL_SIZE = 8
RIDGE_LAMBDAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
CV_SEED = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _load_original_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    split_root = data_root / f"traversal_critic_{split}"
    labels = list(csv.DictReader((split_root / "labels.csv").open()))
    rows = []
    for index, label in enumerate(labels, start=1):
        media = split_root / "media" / f"video_{index:05d}.mp4"
        conversation = split_root / "text" / f"conversation_{index:05d}.json"
        if not media.is_file():
            raise FileNotFoundError(media)
        messages = json.loads(conversation.read_text())["conversations"]
        assistant = next(message for message in messages if message.get("role") == "assistant")
        conversation_label = int(assistant["content"][0]["text"])
        if conversation_label != int(label["overall"]):
            raise ValueError(
                f"labels/conversation mismatch for {split} original {index}: {label['overall']} != {conversation_label}"
            )
        rows.append(
            {
                "media": media,
                "label": int(label["overall"]),
                "scene_id": label["scene_id"],
                "episode_id": label["episode_id"],
            }
        )
    return rows


def _media_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        path = Path(row["media"])
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\0")
        digest.update(str(row["label"]).encode())
        digest.update(b"\0")
        digest.update(row["scene_id"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _extract(rows: list[dict[str, Any]], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from PIL import Image
    from torchcodec.decoders import VideoDecoder

    features = []
    labels = []
    scenes = []
    for index, row in enumerate(rows, start=1):
        decoder = VideoDecoder(str(row["media"]))
        n_frames = int(decoder.metadata.num_frames or 0)
        fps = float(decoder.metadata.average_fps or 0.0)
        if n_frames <= 0 or fps <= 0:
            raise ValueError(f"invalid video metadata: {row['media']}")
        frame = decoder.get_frames_at(indices=[n_frames - 1]).data[0]
        rgb = frame.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
        thumb = Image.fromarray(rgb).resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.BILINEAR)
        duration = n_frames / fps
        features.append(np.concatenate(([duration], np.asarray(thumb, dtype=np.float64).reshape(-1) / 255.0)))
        labels.append(row["label"])
        scenes.append(row["scene_id"])
        if index % 200 == 0 or index == len(rows):
            print(f"[shortcut] decoded {split} {index}/{len(rows)}", flush=True)
    return np.asarray(features), np.asarray(labels, dtype=np.float64), np.asarray(scenes)


def _load_or_extract(
    data_root: Path,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    train_rows = _load_original_rows(data_root, "train")
    val_rows = _load_original_rows(data_root, "val")
    fingerprints = {
        "train": _media_fingerprint(train_rows),
        "val": _media_fingerprint(val_rows),
    }
    if cache_path.is_file():
        cache = np.load(cache_path)
        cached_train = str(cache["train_fingerprint"].item())
        cached_val = str(cache["val_fingerprint"].item())
        if cached_train != fingerprints["train"] or cached_val != fingerprints["val"]:
            raise SystemExit(f"stale temporal-shortcut cache: {cache_path}")
        print(f"[shortcut] loaded cache {cache_path}", flush=True)
        return (
            cache["Xtr"],
            cache["ytr"],
            cache["str"],
            cache["Xva"],
            cache["yva"],
            cache["sva"],
            fingerprints,
        )
    Xtr, ytr, str_ = _extract(train_rows, "train")
    Xva, yva, sva = _extract(val_rows, "val")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        Xtr=Xtr,
        ytr=ytr,
        str=str_,
        Xva=Xva,
        yva=yva,
        sva=sva,
        train_fingerprint=fingerprints["train"],
        val_fingerprint=fingerprints["val"],
    )
    return Xtr, ytr, str_, Xva, yva, sva, fingerprints


def duration_features(duration_seconds: np.ndarray) -> np.ndarray:
    scaled = np.asarray(duration_seconds, dtype=np.float64) / 24.0
    return np.column_stack([scaled, scaled**2, scaled**3, (duration_seconds >= 23.99).astype(np.float64)])


def grouped_folds(scenes: np.ndarray, n_folds: int = CV_FOLDS, seed: int = CV_SEED) -> np.ndarray:
    unique = np.asarray(sorted(set(scenes.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    mapping = {scene: index % n_folds for index, scene in enumerate(unique)}
    return np.asarray([mapping[scene] for scene in scenes], dtype=np.int64)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X.mean(axis=0), X.std(axis=0) + 1e-8


def _design(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.column_stack([(X - mean) / std, np.ones(len(X), dtype=np.float64)])


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)


def pearson(prediction: np.ndarray, labels: np.ndarray) -> float:
    if np.std(prediction) <= 1e-12 or np.std(labels) <= 1e-12:
        return 0.0
    return float(np.corrcoef(prediction, labels)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def metrics(prediction: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    rounded = np.clip(np.round(prediction), 1, 5)
    per_class_mae = {
        str(score): float(np.abs(prediction[labels == score] - score).mean())
        for score in range(1, 6)
        if np.any(labels == score)
    }
    return {
        "pearson_continuous": pearson(prediction, labels),
        "spearman_continuous": pearson(_average_ranks(prediction), _average_ranks(labels)),
        "mae_continuous": float(np.abs(prediction - labels).mean()),
        "macro_mae_continuous": float(np.mean(list(per_class_mae.values()))),
        "per_class_mae_continuous": per_class_mae,
        "pearson_rounded_clipped": pearson(rounded, labels),
        "accuracy_rounded_clipped": float((rounded == labels).mean()),
        "within_one_rounded_clipped": float((np.abs(rounded - labels) <= 1).mean()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
    }


def fit_scene_grouped_probe(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    train_scenes: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
) -> dict[str, Any]:
    folds = grouped_folds(train_scenes)
    cv = {}
    for lam in lambdas:
        prediction = np.empty(len(ytr), dtype=np.float64)
        for fold in range(CV_FOLDS):
            train = folds != fold
            heldout = ~train
            mean, std = _standardize_fit(Xtr[train])
            coef = ridge_fit(_design(Xtr[train], mean, std), ytr[train], lam)
            prediction[heldout] = _design(Xtr[heldout], mean, std) @ coef
        cv[str(lam)] = pearson(prediction, ytr)
    selected_lambda = max(lambdas, key=lambda value: (cv[str(value)], -value))
    mean, std = _standardize_fit(Xtr)
    coef = ridge_fit(_design(Xtr, mean, std), ytr, selected_lambda)
    val_prediction = _design(Xva, mean, std) @ coef
    return {
        "selection": {
            "rule": "maximum pooled scene-grouped five-fold train OOF Pearson; smallest lambda breaks ties",
            "cv_pearson": cv,
            "selected_lambda": selected_lambda,
            "fold_rows": {str(fold): int((folds == fold).sum()) for fold in range(CV_FOLDS)},
            "fold_scenes": {str(fold): len(set(train_scenes[folds == fold].tolist())) for fold in range(CV_FOLDS)},
        },
        "validation": metrics(val_prediction, yva),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--critic-selection", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_audit = Path(args.dataset_audit).expanduser().resolve()
    critic_selection = Path(args.critic_selection).expanduser().resolve()
    cache_path = Path(args.cache).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    audit_payload = json.loads(dataset_audit.read_text())
    if not audit_payload.get("passed"):
        raise SystemExit(f"dataset audit is not passing: {dataset_audit}")
    if Path(audit_payload["_meta"]["dataset_root"]).resolve() != data_root:
        raise SystemExit("dataset audit does not bind --data-root")

    Xtr_raw, ytr, train_scenes, Xva_raw, yva, val_scenes, fingerprints = _load_or_extract(data_root, cache_path)
    duration_train = duration_features(Xtr_raw[:, 0])
    duration_val = duration_features(Xva_raw[:, 0])
    duration_terminal_train = np.column_stack([duration_train, Xtr_raw[:, 1:]])
    duration_terminal_val = np.column_stack([duration_val, Xva_raw[:, 1:]])

    duration_result = fit_scene_grouped_probe(duration_train, ytr, train_scenes, duration_val, yva)
    terminal_result = fit_scene_grouped_probe(
        duration_terminal_train,
        ytr,
        train_scenes,
        duration_terminal_val,
        yva,
    )
    terminal_only_result = fit_scene_grouped_probe(Xtr_raw[:, 1:], ytr, train_scenes, Xva_raw[:, 1:], yva)
    permutation_rng = np.random.default_rng(20260811)
    permutation_result = fit_scene_grouped_probe(
        duration_terminal_train,
        permutation_rng.permutation(ytr),
        train_scenes,
        duration_terminal_val,
        yva,
    )
    constant = np.full(len(yva), ytr.mean(), dtype=np.float64)
    selected = json.loads(critic_selection.read_text())["selected"]
    critic_r = float(selected["metrics"]["pearson_correlation"])
    duration_r = duration_result["validation"]["pearson_continuous"]
    terminal_r = terminal_result["validation"]["pearson_continuous"]

    train_duration = Xtr_raw[:, 0]
    val_duration = Xva_raw[:, 0]
    payload = {
        "_meta": {
            "labeler_version": 3,
            "dynamics": "physics",
            "scene_range": [1000, 1499],
            "data_root": str(data_root),
            "source": file_record(Path(__file__)),
            "dataset_audit": file_record(dataset_audit),
            "critic_selection": file_record(critic_selection),
            "feature_cache": file_record(cache_path),
            "media_fingerprints": fingerprints,
        },
        "protocol": {
            "training_unit": "unique original media (balanced-manifest duplicates excluded)",
            "validation_unit": "unique media from 108 scene-disjoint validation scenes",
            "duration_features": ["duration/24", "(duration/24)^2", "(duration/24)^3", "duration>=23.99s"],
            "terminal_frame_features": f"final decoded RGB frame resized bilinearly to {THUMBNAIL_SIZE}x{THUMBNAIL_SIZE} and flattened",
            "model": "ridge regression with fold-local standardization and unpenalized intercept",
            "lambda_candidates": list(RIDGE_LAMBDAS),
            "selection_data": "train-only scene-grouped five-fold out-of-fold predictions",
            "status": "post-hoc release-blocking shortcut diagnostic; descriptive, not confirmatory",
        },
        "data": {
            "n_train": len(ytr),
            "n_train_scenes": len(set(train_scenes.tolist())),
            "n_val": len(yva),
            "n_val_scenes": len(set(val_scenes.tolist())),
            "train_duration_s": {"min": float(train_duration.min()), "max": float(train_duration.max())},
            "val_duration_s": {"min": float(val_duration.min()), "max": float(val_duration.max())},
            "train_mean_duration_by_label": {
                str(score): float(train_duration[ytr == score].mean()) for score in range(1, 6)
            },
            "val_mean_duration_by_label": {
                str(score): float(val_duration[yva == score].mean()) for score in range(1, 6)
            },
        },
        "constant_train_mean": metrics(constant, yva),
        "duration_only": duration_result,
        "terminal_frame_only": terminal_only_result,
        "duration_plus_terminal_frame": terminal_result,
        "fixed_seed_permuted_training_labels": {
            "seed": 20260811,
            "purpose": "pipeline negative control; validation labels and features remain unchanged",
            **permutation_result,
        },
        "critic_reference": {
            "checkpoint": selected["checkpoint"],
            "iteration": selected["iteration"],
            "pearson": critic_r,
        },
        "descriptive_comparison": {
            "duration_pearson_fraction_of_selected_critic": duration_r / critic_r,
            "duration_plus_terminal_pearson_fraction_of_selected_critic": terminal_r / critic_r,
            "terminal_frame_incremental_pearson": terminal_r - duration_r,
            "note": "fractions share a validation set but compare different model classes and are not variance-explained estimates",
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[shortcut] duration r={duration_r:.4f}; duration+terminal r={terminal_r:.4f}; "
        f"critic r={critic_r:.4f} -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
