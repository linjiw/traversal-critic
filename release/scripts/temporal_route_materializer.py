# SPDX-License-Identifier: OpenMDW-1.1
"""Shared temporal materializer for every traversal-critic video route.

The release-blocking E5 audit
(``autoresearch/run-260811-1753/temporal_preprocessing_audit.json``) found three
incompatible temporal preprocessing interfaces in the v5 pipeline.  Execution
report "Exact next actions" item 3 requires ONE shared materializer for all
corrected future work.  This module is that single entry point: given a video
source and a declared route, it returns the selected PIL frames PLUS their
original source frame indices, the source fps, and per-frame timestamps, and it
routes every processor invocation through ``do_sample_frames=False`` with
explicit video metadata so the Cosmos3-Edge processor can never silently
resample or default to a 24 fps clock.

Declared routes (config-declared and hashable; runs bind to the config hash):

``sft_exact_v5``
    Replays the captured v5 SFT launch dataflow exactly — the two-stage
    double-sampling in which the byte decoder pre-sampled toward 2 fps/<=32
    frames and the Edge processor then re-sampled that list with no metadata,
    defaulting to a 24 fps clock and the 4-frame minimum.  Needed by the frozen
    E1 comparator's SFT-exact cells.
``historical_validation``
    Reproduces the historical validation evaluator's file-path selection
    (4-768 frames toward 2 fps on the true source clock).
``policy_scorer``
    Reproduces the policy scorer daemon's predecoded route (stride sampling
    plus renumbered frame indices on the effective-fps clock).
``corrected_v6``
    The intended single-pass 2 fps/<=32 route with explicit source-coordinate
    metadata — the route all future corrected work uses.

This module is ADDITIVE.  It modifies nothing the frozen v5 pipeline, the
running PPO matrix, its scorer, the SFT capture, or the historical evaluators
depend on.  Frozen logic is copied here (never imported-and-monkeypatched),
with each copy's source and capture authority named next to it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

MATERIALIZER_VERSION = "traversal-temporal-routes-v1"

ROUTE_SFT_EXACT_V5 = "sft_exact_v5"
ROUTE_HISTORICAL_VALIDATION = "historical_validation"
ROUTE_POLICY_SCORER = "policy_scorer"
ROUTE_CORRECTED_V6 = "corrected_v6"


@dataclass(frozen=True)
class RouteConfig:
    """A declared temporal route: name + exact numeric contract + authority.

    ``config_hash`` binds ``MATERIALIZER_VERSION``, the route name, and the
    numeric parameters (not the human-readable authority text), so a run that
    records the hash is bound to the exact selection arithmetic.
    """

    name: str
    params: Mapping[str, float | int]
    authority: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "materializer_version": MATERIALIZER_VERSION,
            "name": self.name,
            "params": dict(self.params),
        }

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


ROUTE_CONFIGS: dict[str, RouteConfig] = {
    ROUTE_SFT_EXACT_V5: RouteConfig(
        name=ROUTE_SFT_EXACT_V5,
        params={
            "stage1_target_fps": 2.0,
            "stage1_max_frames": 32,
            "stage1_missing_fps_fallback": 30.0,
            "stage2_metadata_default_fps": 24.0,
            "stage2_target_fps": 2.0,
            "stage2_min_frames": 4,
            "stage2_max_frames": 768,
        },
        authority=(
            "Stage 1: cosmos_framework/configs/base/reasoner/experiment/videophy2_dataflow_roles.py"
            "::_decode_video_to_pil_frames, byte-verified against the immutable v5 SFT launch capture "
            "sft_attempt_1_20260809T070843-0400_pid18287 (capture sha256 "
            "5e8aa5f647a0246b5af84f629021ea9c0306964be36d47f2234f05a6e31809c3). "
            "Stage 2: Qwen3VLVideoProcessor.sample_frames default path (metadata fps None -> 24, "
            "self.fps=2, min_frames=4, max_frames=768) in "
            "cosmos_framework/model/generator/reasoner/qwen3_vl/video_processing_qwen3_vl.py, "
            "as exercised by the captured launch (per-item 'fps' key is dropped by apply_chat_template)."
        ),
    ),
    ROUTE_HISTORICAL_VALIDATION: RouteConfig(
        name=ROUTE_HISTORICAL_VALIDATION,
        params={
            "target_fps": 2.0,
            "min_frames": 4,
            "max_frames": 768,
        },
        authority=(
            "Historical validation evaluator: MP4 pathname handed to the Edge processor; "
            "Qwen3VLVideoProcessor.sample_frames with true container metadata "
            "(num_frames=int(total/source_fps*2), clamp [4,768], np.linspace(...).round()). "
            "Reimplementation verified exact by the E5 auditor "
            "(scripts/audit_critic_temporal_preprocessing.py::evaluation_indices; "
            "summary.evaluation_sampler_reimplementation_exact=true in "
            "autoresearch/run-260811-1753/temporal_preprocessing_audit.json)."
        ),
    ),
    ROUTE_POLICY_SCORER: RouteConfig(
        name=ROUTE_POLICY_SCORER,
        params={
            "target_fps": 2.0,
            "max_frames": 32,
            "missing_fps_fallback": 25.0,
        },
        authority=(
            "Policy scorer daemon: sim_rl/critic_scorer_daemon.py::_decode_frames and ::score_clip "
            "(stride sampling like training stage 1 but with a 25.0 fps fallback, then "
            "do_sample_frames=False with metadata fps=effective_fps and RENUMBERED frames_indices "
            "0..n-1). Hash-verified against the E5 audit scorer_route records."
        ),
    ),
    ROUTE_CORRECTED_V6: RouteConfig(
        name=ROUTE_CORRECTED_V6,
        params={
            "target_fps": 2.0,
            "max_frames": 32,
        },
        authority=(
            "Corrected single-pass interface: selection arithmetic copied from "
            "cosmos_overlay/cosmos_framework/data/generator/processors/traversal_critic_temporal.py"
            "::select_source_indices (MATERIALIZER_VERSION 'traversal-temporal-v1'), the module the "
            "separately named corrected generation trains/validates/scores through. Metadata keeps "
            "source coordinates: fps=source fps, total_num_frames=source total, "
            "frames_indices=original source indices."
        ),
    ),
}


@dataclass(frozen=True)
class TemporalRoutePlan:
    """Deterministic frame-selection plan for one video under one route.

    ``source_indices`` are always ORIGINAL source-video frame indices of the
    frames handed to the model.  ``prompt_frames_indices``/``prompt_clock_fps``
    are what the processor metadata must carry so the emitted
    ``<t seconds>`` tokens equal the route's historical (or intended) tokens;
    for the legacy routes these clocks are intentionally NOT the true source
    clock — that mismatch is the audited defect being replayed, and
    ``source_timestamps_s`` always reports the true clock alongside it.
    """

    route: str
    route_config_hash: str
    total_source_frames: int
    source_fps_raw: float
    source_fps: float
    source_indices: tuple[int, ...]
    prompt_clock_fps: float
    prompt_frames_indices: tuple[int, ...]
    prompt_metadata_total_num_frames: int
    prompt_metadata_include_duration: bool
    stage_detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.prompt_frames_indices) != len(self.source_indices):
            raise ValueError("prompt_frames_indices and source_indices must have the same length")
        if not self.source_indices:
            raise ValueError("a route plan must select at least one frame")
        if any(a >= b for a, b in zip(self.source_indices, self.source_indices[1:])):
            raise ValueError("source_indices must be strictly increasing")
        if any(i < 0 or i >= self.total_source_frames for i in self.source_indices):
            raise ValueError("source index is outside the source video")
        object.__setattr__(self, "stage_detail", MappingProxyType(dict(self.stage_detail)))

    @property
    def frame_count(self) -> int:
        return len(self.source_indices)

    @property
    def source_timestamps_s(self) -> tuple[float, ...]:
        """True per-frame timestamps on the source-video clock."""
        return tuple(index / self.source_fps for index in self.source_indices)

    @property
    def prompt_timestamps_s(self) -> tuple[float, ...]:
        """Per-frame timestamps the processor will render into the prompt."""
        return tuple(index / self.prompt_clock_fps for index in self.prompt_frames_indices)

    @property
    def prompt_timestamp_token_values(self) -> tuple[float, ...]:
        """The ``<t seconds>`` token values after the processor's ``:.1f`` format."""
        return tuple(float(f"{value:.1f}") for value in self.prompt_timestamps_s)

    @property
    def processor_metadata(self) -> dict[str, Any]:
        """Explicit video metadata for ``do_sample_frames=False`` invocations."""
        metadata: dict[str, Any] = {
            "fps": self.prompt_clock_fps,
            "total_num_frames": self.prompt_metadata_total_num_frames,
            "frames_indices": list(self.prompt_frames_indices),
        }
        if self.prompt_metadata_include_duration:
            metadata["duration"] = self.prompt_metadata_total_num_frames / self.prompt_clock_fps
        return metadata


@dataclass(frozen=True)
class TemporalRouteBundle:
    """A plan plus the decoded PIL frames it selected."""

    plan: TemporalRoutePlan
    frames: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.frames) != self.plan.frame_count:
            raise ValueError("decoded frame count does not match the route plan")

    @property
    def route(self) -> str:
        return self.plan.route

    @property
    def route_config_hash(self) -> str:
        return self.plan.route_config_hash

    @property
    def source_indices(self) -> tuple[int, ...]:
        return self.plan.source_indices

    @property
    def source_fps(self) -> float:
        return self.plan.source_fps

    @property
    def timestamps_s(self) -> tuple[float, ...]:
        return self.plan.source_timestamps_s


def _stride_indices(total_frames: int, source_fps: float, target_fps: float, max_frames: int) -> tuple[int, ...]:
    """v5 stride sampler, copied verbatim in arithmetic from
    ``videophy2_dataflow_roles._decode_video_to_pil_frames`` (see the
    ``sft_exact_v5`` authority above).  Also the scorer daemon's sampler."""
    stride = max(1, int(round(source_fps / target_fps)))
    indices = list(range(0, total_frames, stride))
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(i * step)] for i in range(max_frames)]
    return tuple(indices)


def _processor_default_sample(
    total_frames: int, metadata_fps: float, target_fps: float, min_frames: int, max_frames: int
) -> tuple[int, ...]:
    """``Qwen3VLVideoProcessor.sample_frames`` fps path, copied in arithmetic
    from ``video_processing_qwen3_vl.py`` (byte-verified against the SFT
    launch capture by the E5 auditor)."""
    num_frames = int(total_frames / metadata_fps * target_fps)
    num_frames = min(min(max(num_frames, min_frames), max_frames), total_frames)
    return tuple(np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist())


def _corrected_select_source_indices(
    total_frames: int, source_fps: float, target_fps: float, max_frames: int
) -> tuple[int, ...]:
    """Copied from ``cosmos_overlay/cosmos_framework/data/generator/processors/``
    ``traversal_critic_temporal.py::select_source_indices`` (corrected
    generation's 'traversal-temporal-v1' interface); copied rather than
    imported so this module never touches the sealed overlay."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    stride = max(1, int(round(source_fps / target_fps)))
    candidates = tuple(range(0, total_frames, stride))
    if len(candidates) <= max_frames:
        return candidates
    if max_frames == 1:
        return (candidates[0],)
    last = len(candidates) - 1
    positions = tuple((slot * last) // (max_frames - 1) for slot in range(max_frames))
    return tuple(candidates[position] for position in positions)


def _require_positive(total_frames: int, source_fps: float, route: str) -> None:
    if total_frames <= 0:
        raise ValueError(f"{route}: total_frames must be positive")
    if source_fps <= 0:
        raise ValueError(f"{route}: source_fps must be positive")


def plan_sft_exact_v5(total_frames: int, source_fps_raw: float) -> TemporalRoutePlan:
    config = ROUTE_CONFIGS[ROUTE_SFT_EXACT_V5]
    p = config.params
    source_fps = source_fps_raw if source_fps_raw > 0 else float(p["stage1_missing_fps_fallback"])
    _require_positive(total_frames, source_fps, ROUTE_SFT_EXACT_V5)
    stage1 = _stride_indices(total_frames, source_fps, float(p["stage1_target_fps"]), int(p["stage1_max_frames"]))
    stride = max(1, int(round(source_fps / float(p["stage1_target_fps"]))))
    effective_fps = source_fps / stride
    positions = _processor_default_sample(
        len(stage1),
        float(p["stage2_metadata_default_fps"]),
        float(p["stage2_target_fps"]),
        int(p["stage2_min_frames"]),
        int(p["stage2_max_frames"]),
    )
    return TemporalRoutePlan(
        route=ROUTE_SFT_EXACT_V5,
        route_config_hash=config.config_hash,
        total_source_frames=total_frames,
        source_fps_raw=source_fps_raw,
        source_fps=source_fps,
        source_indices=tuple(stage1[position] for position in positions),
        prompt_clock_fps=float(p["stage2_metadata_default_fps"]),
        prompt_frames_indices=positions,
        prompt_metadata_total_num_frames=len(stage1),
        prompt_metadata_include_duration=False,
        stage_detail={
            "stage1_source_indices": list(stage1),
            "stage1_effective_fps": effective_fps,
            "stage2_positions_in_stage1": list(positions),
        },
    )


def plan_historical_validation(total_frames: int, source_fps_raw: float) -> TemporalRoutePlan:
    config = ROUTE_CONFIGS[ROUTE_HISTORICAL_VALIDATION]
    p = config.params
    _require_positive(total_frames, source_fps_raw, ROUTE_HISTORICAL_VALIDATION)
    indices = _processor_default_sample(
        total_frames, source_fps_raw, float(p["target_fps"]), int(p["min_frames"]), int(p["max_frames"])
    )
    return TemporalRoutePlan(
        route=ROUTE_HISTORICAL_VALIDATION,
        route_config_hash=config.config_hash,
        total_source_frames=total_frames,
        source_fps_raw=source_fps_raw,
        source_fps=source_fps_raw,
        source_indices=indices,
        prompt_clock_fps=source_fps_raw,
        prompt_frames_indices=indices,
        prompt_metadata_total_num_frames=total_frames,
        prompt_metadata_include_duration=False,
        stage_detail={},
    )


def plan_policy_scorer(total_frames: int, source_fps_raw: float) -> TemporalRoutePlan:
    config = ROUTE_CONFIGS[ROUTE_POLICY_SCORER]
    p = config.params
    source_fps = source_fps_raw if source_fps_raw > 0 else float(p["missing_fps_fallback"])
    _require_positive(total_frames, source_fps, ROUTE_POLICY_SCORER)
    indices = _stride_indices(total_frames, source_fps, float(p["target_fps"]), int(p["max_frames"]))
    stride = max(1, int(round(source_fps / float(p["target_fps"]))))
    effective_fps = source_fps / stride
    return TemporalRoutePlan(
        route=ROUTE_POLICY_SCORER,
        route_config_hash=config.config_hash,
        total_source_frames=total_frames,
        source_fps_raw=source_fps_raw,
        source_fps=source_fps,
        source_indices=indices,
        prompt_clock_fps=effective_fps,
        prompt_frames_indices=tuple(range(len(indices))),
        prompt_metadata_total_num_frames=len(indices),
        prompt_metadata_include_duration=False,
        stage_detail={"effective_fps": effective_fps},
    )


def plan_corrected_v6(total_frames: int, source_fps_raw: float) -> TemporalRoutePlan:
    config = ROUTE_CONFIGS[ROUTE_CORRECTED_V6]
    p = config.params
    _require_positive(total_frames, source_fps_raw, ROUTE_CORRECTED_V6)
    indices = _corrected_select_source_indices(
        total_frames, source_fps_raw, float(p["target_fps"]), int(p["max_frames"])
    )
    return TemporalRoutePlan(
        route=ROUTE_CORRECTED_V6,
        route_config_hash=config.config_hash,
        total_source_frames=total_frames,
        source_fps_raw=source_fps_raw,
        source_fps=source_fps_raw,
        source_indices=indices,
        prompt_clock_fps=source_fps_raw,
        prompt_frames_indices=indices,
        prompt_metadata_total_num_frames=total_frames,
        prompt_metadata_include_duration=True,
        stage_detail={},
    )


_PLANNERS = {
    ROUTE_SFT_EXACT_V5: plan_sft_exact_v5,
    ROUTE_HISTORICAL_VALIDATION: plan_historical_validation,
    ROUTE_POLICY_SCORER: plan_policy_scorer,
    ROUTE_CORRECTED_V6: plan_corrected_v6,
}


def resolve_route(route: str | RouteConfig) -> RouteConfig:
    if isinstance(route, RouteConfig):
        declared = ROUTE_CONFIGS.get(route.name)
        if declared is None or declared.config_hash != route.config_hash:
            raise ValueError(f"route config is not one of the declared routes: {route.name}")
        return declared
    if route not in ROUTE_CONFIGS:
        raise ValueError(f"unknown temporal route {route!r}; declared: {sorted(ROUTE_CONFIGS)}")
    return ROUTE_CONFIGS[route]


def plan_route(route: str | RouteConfig, total_frames: int, source_fps: float) -> TemporalRoutePlan:
    """Pure, decode-free frame-selection plan (unit-testable without video IO)."""
    return _PLANNERS[resolve_route(route).name](total_frames, source_fps)


def materialize_route(source: str | Path | bytes, route: str | RouteConfig) -> TemporalRouteBundle:
    """Single entry point: decode ``source`` once under the declared route.

    Returns the selected PIL frames plus original source frame indices, the
    source fps, per-frame timestamps (both the true source clock and the
    route's prompt clock), and the binding route-config hash.
    """
    from PIL import Image
    from torchcodec.decoders import VideoDecoder

    decoder_source: str | bytes = str(source) if isinstance(source, Path) else source
    decoder = VideoDecoder(decoder_source)
    total_frames = int(decoder.metadata.num_frames or 0)
    source_fps_raw = float(decoder.metadata.average_fps or 0.0)
    plan = plan_route(route, total_frames, source_fps_raw)
    frames_tensor = decoder.get_frames_at(indices=list(plan.source_indices)).data
    frames_np = frames_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
    frames = tuple(Image.fromarray(frame) for frame in frames_np)
    return TemporalRouteBundle(plan=plan, frames=frames)


def route_videos_kwargs(bundles: Sequence[TemporalRouteBundle]) -> dict[str, Any]:
    """``videos_kwargs`` that make silent resampling impossible."""
    if not bundles:
        raise ValueError("at least one materialized video is required")
    metadata = [bundle.plan.processor_metadata for bundle in bundles]
    return {
        "do_sample_frames": False,
        "video_metadata": metadata[0] if len(metadata) == 1 else metadata,
    }


def apply_route_chat_template(
    processor: Any,
    conversation: Any,
    bundles: Sequence[TemporalRouteBundle],
    **kwargs: Any,
) -> Any:
    """Invoke the processor with sampling disabled and explicit metadata.

    Every processor call in corrected work must go through here (or pass
    ``route_videos_kwargs`` itself); ``videos_kwargs`` is owned by this
    interface precisely so no caller can reintroduce silent resampling.
    """
    if "videos_kwargs" in kwargs:
        raise TypeError("videos_kwargs is owned by the shared temporal materializer")
    kwargs["videos_kwargs"] = route_videos_kwargs(bundles)
    return processor.apply_chat_template(conversation, **kwargs)
