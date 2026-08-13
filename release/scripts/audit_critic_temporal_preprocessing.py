# SPDX-License-Identifier: OpenMDW-1.1
# Public release copy (stage A); private-path defaults replaced by <REPO>/<DATA> placeholders — see RELEASE_MANIFEST.md
"""Audit traversal-critic training and inference video preprocessing.

The v5 SFT dataflow decodes bytes to a pre-sampled PIL-frame list before it
calls the Cosmos3-Edge processor.  The validation evaluator instead gives the
processor an MP4 pathname.  Those routes exercise different sampling and
metadata code.  This audit compares their source indices, timestamps, video
grid, prompt tokens, and pixel tensors on deterministic validation strata.

Run inside the Cosmos environment because the real processor and torchcodec
are part of the evidence under test::

    cd ~/cosmos-framework
    LD_LIBRARY_PATH=$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib \
      uv run python ~/traversal-critic-research/scripts/audit_critic_temporal_preprocessing.py \
        --data-root <DATA>/traversal-critic/data/critic_v5 \
        --rollout-root <DATA>/traversal-critic/data/rollouts_phys_v5 \
        --processor-checkpoint /path/to/hf_exports/iter_000000100 \
        --dataset-audit <DATA>/traversal-critic/data/critic_v5_dataset_audit.json \
        --training-source /path/to/sft_process/cosmos_source.json \
        --out /path/to/temporal_preprocessing_audit.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

TARGET_VIDEO_FPS = 2.0
TRAIN_MAX_VIDEO_FRAMES = 32
EVAL_MIN_VIDEO_FRAMES = 4
EVAL_MAX_VIDEO_FRAMES = 768
_TIMESTAMP_RE = re.compile(r"<([0-9]+(?:\.[0-9]+)?) seconds>")
TRAINING_SOURCE_FILES = (
    "cosmos_framework/configs/base/reasoner/experiment/videophy2_dataflow_roles.py",
    "cosmos_framework/configs/base/reasoner/experiment/traversal_critic_dataflow_roles.py",
    "cosmos_framework/data/generator/processors/cosmos3_edge_processing.py",
    "cosmos_framework/model/generator/reasoner/qwen3_vl/video_processing_qwen3_vl.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def training_indices(
    total_frames: int,
    source_fps: float,
    *,
    target_fps: float = TARGET_VIDEO_FPS,
    max_frames: int = TRAIN_MAX_VIDEO_FRAMES,
) -> tuple[list[int], float]:
    """Reproduce ``VideoPhy2Processor`` byte-decoder sampling exactly."""
    if total_frames <= 0 or source_fps <= 0:
        raise ValueError("positive total_frames and source_fps are required")
    stride = max(1, int(round(source_fps / target_fps)))
    indices = list(range(0, total_frames, stride))
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]
    return indices, float(source_fps / stride)


def evaluation_indices(
    total_frames: int,
    source_fps: float,
    *,
    target_fps: float = TARGET_VIDEO_FPS,
    min_frames: int = EVAL_MIN_VIDEO_FRAMES,
    max_frames: int = EVAL_MAX_VIDEO_FRAMES,
) -> list[int]:
    """Reproduce the Edge file-path video processor's default sampler."""
    if total_frames <= 0 or source_fps <= 0:
        raise ValueError("positive total_frames and source_fps are required")
    num_frames = int(total_frames / source_fps * target_fps)
    num_frames = min(min(max(num_frames, min_frames), max_frames), total_frames)
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()


def training_timestamps(indices: list[int], effective_fps: float) -> list[float]:
    """Timestamps the processor sees after the source indices are discarded."""
    return [i / effective_fps for i in range(len(indices))]


def evaluation_timestamps(indices: list[int], source_fps: float) -> list[float]:
    return [i / source_fps for i in indices]


def _load_rows(data_root: Path, rollout_root: Path) -> list[dict[str, Any]]:
    split_root = data_root / "traversal_critic_val"
    labels = list(csv.DictReader((split_root / "labels.csv").open()))
    meta = json.loads((split_root / "meta.json").read_text())
    if len(labels) != len(meta):
        raise ValueError(f"validation labels/meta mismatch: {len(labels)} != {len(meta)}")
    rows = []
    for index, (label, entry) in enumerate(zip(labels, meta), start=1):
        privileged_path = rollout_root / label["episode_id"] / "privileged.json"
        privileged = json.loads(privileged_path.read_text())
        duration = len(privileged["frames"]) / float(privileged["fps"])
        rows.append(
            {
                "index": index,
                "episode_id": label["episode_id"],
                "scene_id": label["scene_id"],
                "label": int(label["overall"]),
                "success": bool(privileged.get("success")),
                "fall": bool(privileged.get("fall")),
                "privileged_duration_s": duration,
                "media": split_root / entry["media"],
                "conversation": split_root / entry["conversation"],
                "privileged": privileged_path,
            }
        )
    return rows


def select_representatives(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Choose deterministic, distinct short/capped/success/fall/timeout rows."""
    chosen: dict[str, dict[str, Any]] = {}
    used: set[str] = set()

    def take(name: str, candidates: list[dict[str, Any]]) -> None:
        for row in candidates:
            if row["episode_id"] not in used:
                chosen[name] = row
                used.add(row["episode_id"])
                return
        raise ValueError(f"no distinct representative found for {name}")

    def stable(row: dict[str, Any]) -> tuple[float, str]:
        return row["privileged_duration_s"], row["episode_id"]

    take(
        "short",
        sorted(
            [r for r in rows if r["privileged_duration_s"] < 12 and not r["success"] and not r["fall"]],
            key=stable,
        ),
    )
    take(
        "capped",
        sorted(
            [r for r in rows if 16 < r["privileged_duration_s"] < 23.9 and not r["success"] and not r["fall"]],
            key=stable,
        ),
    )
    take("successful", sorted([r for r in rows if r["success"]], key=stable))
    take("falling", sorted([r for r in rows if r["fall"]], key=stable))
    take(
        "timeout",
        sorted(
            [r for r in rows if r["privileged_duration_s"] >= 23.9 and not r["success"] and not r["fall"]],
            key=lambda row: row["episode_id"],
        ),
    )
    return chosen


def _user_messages(conversation_path: Path, video_value: Any) -> list[dict[str, Any]]:
    messages = json.loads(conversation_path.read_text())["conversations"]
    messages = copy.deepcopy([message for message in messages if message.get("role") != "assistant"])
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                item["video"] = video_value
    return messages


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _timestamps_from_input(processor: Any, inputs: dict[str, Any]) -> list[float]:
    decoded = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    return [float(value) for value in _TIMESTAMP_RE.findall(decoded)]


def _apply(processor: Any, messages: list[dict[str, Any]], *, videos_kwargs: dict | None = None) -> dict:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": "pt",
        "return_dict": True,
        "enable_thinking": False,
    }
    if videos_kwargs is not None:
        kwargs["videos_kwargs"] = videos_kwargs
    inputs = processor.apply_chat_template(messages, **kwargs)
    return {
        "inputs": inputs,
        "input_ids": _tensor_record(inputs["input_ids"]),
        "pixel_values_videos": _tensor_record(inputs["pixel_values_videos"]),
        "video_grid_thw": _tensor_record(inputs["video_grid_thw"]),
        "timestamps_s": _timestamps_from_input(processor, inputs),
        "video_grid_thw_values": inputs["video_grid_thw"].tolist(),
    }


def _same_tensors(left: dict, right: dict) -> dict[str, bool]:
    import torch

    keys = ("input_ids", "pixel_values_videos", "video_grid_thw")
    return {key: bool(torch.equal(left["inputs"][key], right["inputs"][key])) for key in keys}


def _public_path_record(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "inputs"}


def audit_sample(processor: Any, row: dict[str, Any]) -> dict[str, Any]:
    from torchcodec.decoders import VideoDecoder

    from cosmos_framework.configs.base.reasoner.experiment.videophy2_dataflow_roles import (
        _decode_video_to_pil_frames,
    )

    video_path = Path(row["media"])
    decoder = VideoDecoder(str(video_path))
    total_frames = int(decoder.metadata.num_frames or 0)
    source_fps = float(decoder.metadata.average_fps or 0.0)
    if total_frames <= 0 or source_fps <= 0:
        raise ValueError(f"invalid video metadata: {video_path}")

    train_indices, effective_fps = training_indices(total_frames, source_fps)
    eval_expected = evaluation_indices(total_frames, source_fps)

    frames, decoded_effective_fps = _decode_video_to_pil_frames(video_path.read_bytes())
    if not math.isclose(decoded_effective_fps, effective_fps, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("training decoder effective fps differs from reproduced planner")
    train_messages = _user_messages(row["conversation"], frames)
    for message in train_messages:
        if not isinstance(message.get("content"), list):
            continue
        for item in message["content"]:
            if isinstance(item, dict) and item.get("type") == "video":
                item["fps"] = effective_fps
    training = _apply(processor, train_messages)

    scorer_metadata = {
        "fps": effective_fps,
        "total_num_frames": len(frames),
        "frames_indices": list(range(len(frames))),
    }
    scorer = _apply(
        processor,
        copy.deepcopy(train_messages),
        videos_kwargs={"do_sample_frames": False, "video_metadata": scorer_metadata},
    )

    captured: list[dict[str, Any]] = []
    original_sample_frames = processor.video_processor.sample_frames

    def spy(metadata: Any, *args: Any, **kwargs: Any) -> Any:
        indices = original_sample_frames(metadata, *args, **kwargs)
        captured.append({"metadata": metadata, "indices": indices.tolist(), "kwargs": kwargs})
        return indices

    processor.video_processor.sample_frames = spy
    try:
        evaluation = _apply(processor, _user_messages(row["conversation"], str(video_path)))
    finally:
        processor.video_processor.sample_frames = original_sample_frames
    if len(captured) != 1:
        raise AssertionError(f"expected one evaluator sampling event, observed {len(captured)}")
    eval_actual = captured[0]["indices"]
    sampled_metadata = captured[0]["metadata"]
    eval_metadata_indices = getattr(sampled_metadata, "frames_indices", None)
    if hasattr(eval_metadata_indices, "tolist"):
        eval_metadata_indices = eval_metadata_indices.tolist()

    train_vs_scorer = _same_tensors(training, scorer)
    train_vs_eval = _same_tensors(training, evaluation)
    return {
        "stratum": row["stratum"],
        "episode_id": row["episode_id"],
        "scene_id": row["scene_id"],
        "label": row["label"],
        "success": row["success"],
        "fall": row["fall"],
        "privileged_duration_s": row["privileged_duration_s"],
        "media": file_record(video_path),
        "conversation": file_record(Path(row["conversation"])),
        "privileged": file_record(Path(row["privileged"])),
        "source_video": {
            "total_frames": total_frames,
            "fps": source_fps,
            "duration_s_num_frames_over_fps": total_frames / source_fps,
        },
        "training_route": {
            "source_indices": train_indices,
            "effective_fps": effective_fps,
            "metadata_frame_indices": list(range(len(train_indices))),
            "expected_timestamps_s": training_timestamps(train_indices, effective_fps),
            **_public_path_record(training),
        },
        "scorer_route": {
            "source_indices": train_indices,
            "metadata": scorer_metadata,
            **_public_path_record(scorer),
        },
        "evaluation_route": {
            "expected_source_indices": eval_expected,
            "actual_source_indices": eval_actual,
            "metadata_frame_indices": eval_metadata_indices,
            "expected_timestamps_s": evaluation_timestamps(eval_actual, source_fps),
            **_public_path_record(evaluation),
        },
        "checks": {
            "evaluation_sampler_matches_reimplementation": eval_actual == eval_expected,
            "training_scorer_tensor_equality": train_vs_scorer,
            "training_evaluation_tensor_equality": train_vs_eval,
            "training_scorer_all_equal": all(train_vs_scorer.values()),
            "training_evaluation_all_equal": all(train_vs_eval.values()),
            "source_indices_equal": train_indices == eval_actual,
            "timestamps_equal": training["timestamps_s"] == evaluation["timestamps_s"],
        },
    }


def _checkpoint_processor_record(checkpoint: Path) -> dict[str, Any]:
    names = (
        "tokenizer.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    return {
        "path": str(checkpoint.resolve()),
        "files": {name: file_record(checkpoint / name) for name in names},
    }


def verify_training_source(training_source: Path) -> dict[str, Any]:
    """Require the executed preprocessing modules to equal the SFT launch capture."""
    import cosmos_framework

    payload = json.loads(training_source.read_text())
    captured_files = payload.get("source_files")
    if not isinstance(captured_files, dict):
        raise ValueError("training source capture has no source_files map")
    repository = Path(cosmos_framework.__file__).resolve().parent.parent
    records = {}
    for relative in TRAINING_SOURCE_FILES:
        current = repository / relative
        current_sha = sha256(current)
        captured_sha = captured_files.get(relative)
        if current_sha != captured_sha:
            raise ValueError(
                f"current preprocessing source differs from SFT capture: {relative} "
                f"current={current_sha} captured={captured_sha}"
            )
        records[relative] = {
            "current": file_record(current),
            "captured_sha256": captured_sha,
            "exact": True,
        }
    return {
        "capture": file_record(training_source),
        "captured_at": payload.get("_meta", {}).get("captured_at"),
        "source_tree_sha256": payload.get("source_tree_sha256"),
        "verified_files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--processor-checkpoint", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument(
        "--training-source",
        required=True,
        help="Immutable cosmos_source.json captured at the v5 SFT process launch.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--require-alignment",
        action="store_true",
        help="Exit nonzero when training and evaluator tensors differ.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    rollout_root = Path(args.rollout_root).expanduser().resolve()
    checkpoint = Path(args.processor_checkpoint).expanduser().resolve()
    dataset_audit = Path(args.dataset_audit).expanduser().resolve()
    training_source = Path(args.training_source).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    audit_payload = json.loads(dataset_audit.read_text())
    if not audit_payload.get("passed"):
        raise SystemExit(f"dataset audit is not passing: {dataset_audit}")
    if Path(audit_payload["_meta"]["dataset_root"]).resolve() != data_root:
        raise SystemExit("dataset audit does not bind --data-root")

    from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
        build_cosmos3_edge_processor,
    )

    processor = build_cosmos3_edge_processor(str(checkpoint))
    training_source_record = verify_training_source(training_source)
    representatives = select_representatives(_load_rows(data_root, rollout_root))
    results = []
    for stratum, selected in representatives.items():
        row = dict(selected)
        row["stratum"] = stratum
        print(f"[temporal-audit] {stratum}: {row['episode_id']}", flush=True)
        results.append(audit_sample(processor, row))

    scorer_aligned = all(result["checks"]["training_scorer_all_equal"] for result in results)
    evaluation_aligned = all(result["checks"]["training_evaluation_all_equal"] for result in results)
    source_indices_aligned = all(result["checks"]["source_indices_equal"] for result in results)
    timestamps_aligned = all(result["checks"]["timestamps_equal"] for result in results)
    samplers_reproduced = all(result["checks"]["evaluation_sampler_matches_reimplementation"] for result in results)
    errors = []
    if not scorer_aligned:
        errors.append("training and scorer preprocessing differ on at least one representative")
    if not evaluation_aligned:
        errors.append("training and validation-evaluator tensors differ on at least one representative")
    if not source_indices_aligned:
        errors.append("training and validation-evaluator source frame indices differ")
    if not timestamps_aligned:
        errors.append("training and validation-evaluator timestamp tokens differ")
    if not samplers_reproduced:
        errors.append("observed evaluator sampling differs from the audited reimplementation")

    payload = {
        "_meta": {
            "labeler_version": 3,
            "dynamics": "physics",
            "scene_range": [1000, 1499],
            "data_root": str(data_root),
            "rollout_root": str(rollout_root),
            "audit_source": file_record(Path(__file__)),
            "dataset_audit": file_record(dataset_audit),
            "sft_launch_source": training_source_record,
            "processor_checkpoint": _checkpoint_processor_record(checkpoint),
        },
        "protocol": {
            "target_fps": TARGET_VIDEO_FPS,
            "training_max_frames": TRAIN_MAX_VIDEO_FRAMES,
            "evaluation_min_frames": EVAL_MIN_VIDEO_FRAMES,
            "evaluation_max_frames": EVAL_MAX_VIDEO_FRAMES,
            "strata": list(representatives),
            "comparison": "same user prompt and generation template; only video materialization route differs",
        },
        "summary": {
            "n": len(results),
            "training_scorer_exact": scorer_aligned,
            "training_evaluation_exact": evaluation_aligned,
            "source_indices_exact": source_indices_aligned,
            "timestamps_exact": timestamps_aligned,
            "evaluation_sampler_reimplementation_exact": samplers_reproduced,
            "semantic_alignment_passed": not errors,
        },
        "results": results,
        "errors": errors,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[temporal-audit] semantic_alignment_passed={not errors} -> {out_path}",
        flush=True,
    )
    if args.require_alignment and errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
