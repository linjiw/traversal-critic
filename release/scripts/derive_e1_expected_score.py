# SPDX-License-Identifier: OpenMDW-1.1
# Public release copy (stage A); private-path defaults replaced by <REPO>/<DATA> placeholders — see RELEASE_MANIFEST.md
"""Derive the E1 expected-score readout from saved per-item candidate logits.

Implements the dated amendment
``docs/reviews/e1_amendment_expected_score_readout_2026-08-11.md``: a third,
post-hoc decoding readout computed from the per-item candidate logits that the
frozen E1 comparator saves for every constrained generation pass.  For each
completed constrained cell it takes a numerically stable softmax over exactly
the five recorded candidate token logits, computes ``E[s] = sum(s * p(s))`` in
``[1, 5]``, and reports rank correlations on the continuous values (primary),
rounded-score accuracy metrics (secondary, round-half-to-even clipped to
``[1, 5]``), a histogram of the continuous values, and descriptive
scene-clustered bootstrap intervals that reuse the frozen bootstrap index
artifact verbatim (no new random draws; this module never imports an RNG).

The tool is read-only on every source artifact.  It refuses tampered or
under-specified cells (missing or non-finite candidate logits, candidate
contracts that are not five distinct single-token IDs, manifest hash
mismatches), performs no model inference, and writes exactly one new derived
JSON.  If no constrained cell has completed yet, it states that and exits
cleanly with status 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

SCORE_KEYS = ("1", "2", "3", "4", "5")
SOURCE_INTERFACE = "constrained"
READOUT = "expected_score"
RESULT_VERSION = 1
PLAN_VERSION = 1
HISTOGRAM_START = 1.0
HISTOGRAM_END = 5.0
HISTOGRAM_BINS = 16
DEFAULT_ROOT = Path("<DATA>/traversal-critic/data/e1_factorial_v5")
DEFAULT_AMENDMENT = (
    Path(__file__).resolve().parent.parent / "docs/reviews/e1_amendment_expected_score_readout_2026-08-11.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"expected a regular non-symlink file: {candidate}")
    path = candidate.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _atomic_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing redirected output: {path}")
    if not replace and path.exists():
        raise FileExistsError(f"write-once derived artifact already exists: {path}")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _item_tree_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def expected_score(candidate_logits: dict[str, Any]) -> float:
    """Numerically stable softmax over exactly the five candidate logits, then E[s]."""
    if set(candidate_logits) != set(SCORE_KEYS):
        raise ValueError(
            f"candidate logits must contain exactly the five score keys {SCORE_KEYS}: {sorted(candidate_logits)}"
        )
    logits: list[float] = []
    for key in SCORE_KEYS:
        value = candidate_logits[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"candidate logit for score {key} is not a finite number: {value!r}")
        logits.append(float(value))
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return sum((index + 1) * weight for index, weight in enumerate(weights)) / total


def round_half_even_clip(value: float) -> int:
    """Round-half-to-even to the nearest integer, clipped to [1, 5]."""
    return min(5, max(1, round(value)))


def _select_constrained(logits: dict[str, Any]) -> int:
    """Replay of the frozen E1 constrained argmax (ties break to the lower score)."""
    return int(max(range(1, 6), key=lambda score: (float(logits[str(score)]), -score)))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _pearson_float(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    xss = sum((x - mx) ** 2 for x in xs)
    yss = sum((y - my) ** 2 for y in ys)
    denominator = math.sqrt(xss * yss)
    return numerator / denominator if denominator else None


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman_float(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson_float(_average_ranks(xs), _average_ranks(ys))


def _kendall_tau_b_float(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for left in range(len(xs) - 1):
        for right in range(left + 1, len(xs)):
            dx = (xs[left] > xs[right]) - (xs[left] < xs[right])
            dy = (ys[left] > ys[right]) - (ys[left] < ys[right])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def histogram(values: list[float]) -> dict[str, Any]:
    """Fixed-bin histogram of the continuous expected scores over [1, 5]."""
    width = (HISTOGRAM_END - HISTOGRAM_START) / HISTOGRAM_BINS
    counts = [0] * HISTOGRAM_BINS
    for value in values:
        if not HISTOGRAM_START <= value <= HISTOGRAM_END:
            raise ValueError(f"expected score outside [1, 5]: {value!r}")
        index = min(int((value - HISTOGRAM_START) / width), HISTOGRAM_BINS - 1)
        counts[index] += 1
    return {
        "values": "continuous_expected_scores",
        "bin_edges": [HISTOGRAM_START + width * index for index in range(HISTOGRAM_BINS + 1)],
        "counts": counts,
    }


def _require_single_token_contract(contract: Any, label: str) -> dict[str, int]:
    """Refuse any candidate contract that is not five distinct single-token IDs."""
    if not isinstance(contract, dict):
        raise ValueError(f"{label}: token contract is missing")
    candidate_ids = contract.get("candidate_token_ids")
    if not isinstance(candidate_ids, dict) or set(candidate_ids) != set(SCORE_KEYS):
        raise ValueError(f"{label}: candidate_token_ids must map exactly the five responses {SCORE_KEYS}")
    for key, value in candidate_ids.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{label}: candidate response {key!r} is not a single token ID "
                f"(multi-token or malformed candidate): {value!r}"
            )
    if len(set(candidate_ids.values())) != 5:
        raise ValueError(f"{label}: candidate token IDs are not five distinct single tokens")
    return {key: int(value) for key, value in candidate_ids.items()}


def load_bootstrap(plan: dict[str, Any]) -> dict[str, Any]:
    """Load the frozen scene-bootstrap index artifact and bind it to the plan bytes."""
    record = plan.get("bootstrap")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("plan has no bootstrap file record")
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"bootstrap index artifact is missing or redirected: {path}")
    if record.get("sha256") != sha256(path) or record.get("bytes") != path.stat().st_size:
        raise ValueError(f"bootstrap index artifact does not match the frozen plan record: {path}")
    bootstrap = _json(path)
    scene_count = plan.get("validation", {}).get("scene_count")
    if not isinstance(scene_count, int) or scene_count < 1:
        raise ValueError("plan validation scene_count is missing")
    scene_ids = bootstrap.get("scene_ids")
    if not isinstance(scene_ids, list) or len(scene_ids) != scene_count:
        raise ValueError("bootstrap scene_ids do not match the plan scene count")
    meta = bootstrap.get("_meta", {})
    draws = bootstrap.get("draw_indices")
    if meta.get("scene_count") != scene_count:
        raise ValueError("bootstrap _meta scene_count mismatch")
    if not isinstance(draws, list) or len(draws) != meta.get("draw_count"):
        raise ValueError("bootstrap draw count does not match its declared _meta")
    for draw in draws:
        if len(draw) != scene_count or any(
            not isinstance(index, int) or index < 0 or index >= scene_count for index in draw
        ):
            raise ValueError("bootstrap draw indices are malformed or out of range")
    return bootstrap


def load_constrained_cell(
    cell: dict[str, Any],
    plan_hash: str,
    sample_count: int,
    scene_ids: list[str],
) -> dict[str, Any]:
    """Load, hash-verify, and derive one completed constrained cell (read-only)."""
    cell_id = cell["cell_id"]
    cell_dir = Path(cell["path"])
    manifest_path = cell_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError(f"{cell_id}: manifest is redirected: {manifest_path}")
    manifest = _json(manifest_path)
    for name, expected in (
        ("plan_sha256", plan_hash),
        ("iteration", cell["iteration"]),
        ("route", cell["route"]),
        ("interface", SOURCE_INTERFACE),
        ("sample_count", sample_count),
    ):
        if manifest.get(name) != expected:
            raise ValueError(f"{cell_id}: manifest {name} mismatch")
    candidate_ids = _require_single_token_contract(manifest.get("token_contract"), cell_id)
    item_files = manifest.get("item_files")
    if not isinstance(item_files, dict) or len(item_files) != sample_count:
        raise ValueError(f"{cell_id}: manifest item_files is not the complete {sample_count}-item set")
    samples: list[dict[str, Any]] = []
    verified_files: dict[str, str] = {}
    for name, recorded_hash in sorted(item_files.items()):
        item_path = cell_dir / "items" / name
        if item_path.is_symlink() or not item_path.is_file():
            raise ValueError(f"{cell_id}: item is missing or redirected: {item_path}")
        actual_hash = sha256(item_path)
        if actual_hash != recorded_hash:
            raise ValueError(f"{cell_id}: item bytes do not match the cell manifest: {item_path}")
        verified_files[name] = actual_hash
        item = _json(item_path)
        for field, expected in (
            ("plan_sha256", plan_hash),
            ("iteration", cell["iteration"]),
            ("route", cell["route"]),
            ("interface", SOURCE_INTERFACE),
        ):
            if item.get(field) != expected:
                raise ValueError(f"{cell_id}: {name} {field} mismatch")
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or f"{sample_id}.json" != name:
            raise ValueError(f"{cell_id}: {name} sample id mismatch")
        ground_truth = item.get("ground_truth")
        if isinstance(ground_truth, bool) or ground_truth not in range(1, 6):
            raise ValueError(f"{cell_id}: {name} ground truth is not an integer in 1..5")
        scene_id = item.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"{cell_id}: {name} scene_id missing")
        if item.get("candidate_token_ids") != candidate_ids:
            raise ValueError(f"{cell_id}: {name} candidate token IDs do not match the cell contract")
        logits = item.get("candidate_logits")
        if logits is None:
            raise ValueError(f"{cell_id}: {name} has no saved candidate logits; cannot derive expected score")
        value = expected_score(logits)
        if item.get("selected_score") != _select_constrained(logits):
            raise ValueError(f"{cell_id}: {name} selected_score is not the argmax of its saved logits")
        samples.append(
            {
                "id": sample_id,
                "scene_id": scene_id,
                "ground_truth": int(ground_truth),
                "expected": value,
            }
        )
    if len({sample["id"] for sample in samples}) != sample_count:
        raise ValueError(f"{cell_id}: duplicate sample IDs")
    if {sample["scene_id"] for sample in samples} != set(scene_ids):
        raise ValueError(f"{cell_id}: item scene IDs do not cover the frozen bootstrap scene set")
    return {
        "samples": samples,
        "manifest_sha256": sha256(manifest_path),
        "item_tree_sha256": _item_tree_sha256(verified_files),
        "manifest_path": str(manifest_path.resolve()),
    }


def cell_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Amendment metrics: continuous rank correlations primary, rounded secondary."""
    xs = [sample["expected"] for sample in samples]
    ys = [float(sample["ground_truth"]) for sample in samples]
    rounded = [round_half_even_clip(value) for value in xs]
    count = len(samples)
    exact = within_one = 0
    per_class_total = {score: 0 for score in range(1, 6)}
    per_class_exact = {score: 0 for score in range(1, 6)}
    for prediction, sample in zip(rounded, samples):
        ground_truth = sample["ground_truth"]
        per_class_total[ground_truth] += 1
        exact += int(prediction == ground_truth)
        within_one += int(abs(prediction - ground_truth) <= 1)
        per_class_exact[ground_truth] += int(prediction == ground_truth)
    return {
        "num_samples": count,
        "pearson": _pearson_float(xs, ys),
        "spearman": _spearman_float(xs, ys),
        "kendall_tau_b": _kendall_tau_b_float(xs, ys),
        "exact_accuracy_rounded": exact / count if count else 0.0,
        "within_one_accuracy_rounded": within_one / count if count else 0.0,
        "per_class_recall_rounded": {
            str(score): (per_class_exact[score] / per_class_total[score] if per_class_total[score] else None)
            for score in range(1, 6)
        },
        "expected_score_mean": _mean(xs) if xs else None,
        "expected_score_min": min(xs) if xs else None,
        "expected_score_max": max(xs) if xs else None,
        "histogram": histogram(xs),
    }


def cell_bootstrap(samples: list[dict[str, Any]], bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Descriptive scene-clustered intervals from the frozen draw indices, verbatim."""
    by_scene: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        by_scene.setdefault(sample["scene_id"], []).append((sample["expected"], float(sample["ground_truth"])))
    scene_ids = bootstrap["scene_ids"]
    pearson_values: list[float] = []
    spearman_values: list[float] = []
    undefined = {"pearson": 0, "spearman": 0}
    for draw in bootstrap["draw_indices"]:
        xs: list[float] = []
        ys: list[float] = []
        for index in draw:
            for x, y in by_scene[scene_ids[index]]:
                xs.append(x)
                ys.append(y)
        pearson = _pearson_float(xs, ys)
        spearman = _spearman_float(xs, ys)
        if pearson is None:
            undefined["pearson"] += 1
        else:
            pearson_values.append(pearson)
        if spearman is None:
            undefined["spearman"] += 1
        else:
            spearman_values.append(spearman)

    def interval(values: list[float], undefined_count: int) -> dict[str, Any]:
        return {
            "lower_2_5pct": _percentile(values, 0.025) if values else None,
            "upper_97_5pct": _percentile(values, 0.975) if values else None,
            "defined_draws": len(values),
            "undefined_draws": undefined_count,
        }

    return {
        "interpretation": "descriptive",
        "pearson": interval(pearson_values, undefined["pearson"]),
        "spearman": interval(spearman_values, undefined["spearman"]),
    }


def derive(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    plan_path = (args.plan or root / "e1_plan.json").expanduser().resolve()
    plan_audit_path = (args.plan_audit or root / "e1_plan_audit.json").expanduser().resolve()
    out_path = (args.out or root / "e1_expected_score_derived.json").expanduser().resolve()
    amendment_record = file_record(args.amendment)

    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError(f"sealed E1 plan is missing or redirected: {plan_path}")
    plan_hash = sha256(plan_path)
    plan = _json(plan_path)
    if plan.get("_meta", {}).get("version") != PLAN_VERSION:
        raise ValueError(f"unsupported E1 plan version: {plan.get('_meta', {}).get('version')!r}")
    if plan_audit_path.is_symlink() or not plan_audit_path.is_file():
        raise ValueError(f"independent E1 plan audit is missing or redirected: {plan_audit_path}")
    plan_audit = _json(plan_audit_path)
    if plan_audit.get("passed") is not True:
        raise ValueError("independent E1 plan audit did not pass")
    if plan_audit.get("_meta", {}).get("plan_sha256") != plan_hash:
        raise ValueError("independent E1 plan audit does not bind the current plan bytes")
    sample_count = plan.get("validation", {}).get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("plan validation sample_count is missing")

    bootstrap = load_bootstrap(plan)
    scene_ids = bootstrap["scene_ids"]

    constrained_cells = [
        cell
        for cell in plan.get("factorial", {}).get("cells", [])
        if isinstance(cell, dict) and cell.get("interface") == SOURCE_INTERFACE
    ]
    if not constrained_cells:
        raise ValueError("plan contains no constrained cells")

    derived_cells: list[dict[str, Any]] = []
    source_cells: dict[str, Any] = {}
    pending: list[dict[str, Any]] = []
    for cell in constrained_cells:
        cell_dir = Path(cell["path"])
        if not (cell_dir / "manifest.json").is_file():
            items_dir = cell_dir / "items"
            pending.append(
                {
                    "cell_id": cell["cell_id"],
                    "path": str(cell_dir),
                    "status": (
                        "items_partial" if items_dir.is_dir() and any(items_dir.glob("*.json")) else "not_started"
                    ),
                }
            )
            continue
        loaded = load_constrained_cell(cell, plan_hash, sample_count, scene_ids)
        samples = loaded["samples"]
        derived_id = f"iter_{cell['iteration']:03d}__{cell['route']}__{READOUT}"
        derived_cells.append(
            {
                "cell_id": derived_id,
                "source_cell_id": cell["cell_id"],
                "iteration": cell["iteration"],
                "route": cell["route"],
                "readout": READOUT,
                "sample_count": len(samples),
                "metrics": cell_metrics(samples),
                "scene_cluster_bootstrap_95pct": cell_bootstrap(samples, bootstrap),
                "expected_scores": {sample["id"]: sample["expected"] for sample in samples},
            }
        )
        source_cells[derived_id] = {
            "source_cell_id": cell["cell_id"],
            "manifest_path": loaded["manifest_path"],
            "manifest_sha256": loaded["manifest_sha256"],
            "item_tree_sha256": loaded["item_tree_sha256"],
        }

    if not derived_cells:
        print(
            "[e1-expected-score] no completed constrained cell exists yet under "
            f"{root}; nothing to derive (expected before the factorial runs)"
        )
        return 0

    derived_at = args.derived_at or datetime.now().astimezone().isoformat()
    result = {
        "_meta": {
            "version": RESULT_VERSION,
            "derived_at": derived_at,
            "readout": READOUT,
            "definition": (
                "softmax over exactly the five recorded candidate token logits at the "
                "first score-token position; E[s] = sum(s * p(s)) in [1, 5]"
            ),
            "amendment": amendment_record,
            "plan": {"path": str(plan_path), "sha256": plan_hash},
            "plan_audit": {"path": str(plan_audit_path), "sha256": sha256(plan_audit_path)},
            "bootstrap": dict(plan["bootstrap"]),
            "script": file_record(Path(__file__)),
            "source_cells": source_cells,
            "source_interface": SOURCE_INTERFACE,
            "new_inference_performed": False,
            "rounding": "round_half_to_even_clipped_to_1_5_secondary_only",
            "bootstrap_metrics": ["pearson", "spearman"],
            "bootstrap_note": (
                "intervals reuse the frozen scene-cluster index artifact verbatim and "
                "mirror the frozen protocol's Pearson/Spearman interval set; Kendall "
                "tau-b is reported as a point estimate only; all intervals are "
                "descriptive because the same validation set participated in the "
                "historical checkpoint sweep"
            ),
            "selection_power": "none; the frozen iteration-100 matrix critic is unaffected",
        },
        "cell_count": len(derived_cells),
        "cells": derived_cells,
        "pending_source_cells": pending,
    }
    _atomic_json(out_path, result, replace=args.replace)
    print(f"[e1-expected-score] derived {len(derived_cells)} cell(s) ({len(pending)} pending): {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="E1 results root containing e1_plan.json and cells/",
    )
    parser.add_argument("--plan", type=Path, default=None, help="override: sealed E1 plan path")
    parser.add_argument("--plan-audit", type=Path, default=None, help="override: independent plan audit path")
    parser.add_argument(
        "--amendment",
        type=Path,
        default=DEFAULT_AMENDMENT,
        help="dated expected-score amendment document to bind into _meta",
    )
    parser.add_argument("--out", type=Path, default=None, help="override: derived JSON output path")
    parser.add_argument(
        "--derived-at",
        default=None,
        help="ISO-8601 derivation timestamp recorded in _meta (default: wall clock, "
        "matching the E1 evaluator's timestamp style)",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="allow overwriting an existing derived JSON (default: write-once)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return derive(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"[e1-expected-score] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
