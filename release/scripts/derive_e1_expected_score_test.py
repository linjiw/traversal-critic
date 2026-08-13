# SPDX-License-Identifier: OpenMDW-1.1
"""Adversarial tests for the E1 expected-score derivation tool.

All fixtures are synthetic but byte-for-byte match the schema the frozen E1
runner writes: per-item JSONs under ``<cell>/items/<id>.json`` with
``candidate_token_ids`` / ``candidate_logits`` keyed by the score strings
``"1"``-``"5"``, a hash-binding ``manifest.json`` per cell, the sealed plan,
its independent audit, and the frozen scene-bootstrap index artifact.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts/derive_e1_expected_score.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


derive = _load("derive_e1_expected_score", MODULE_PATH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sharp(*scores: int) -> dict[str, float]:
    """Logits whose softmax is exactly uniform over ``scores`` and zero elsewhere."""
    return {str(s): (0.0 if s in scores else -4000.0) for s in range(1, 6)}


def _uniform() -> dict[str, float]:
    return {str(s): 7.5 for s in range(1, 6)}


CANDIDATE_IDS = {"1": 101, "2": 102, "3": 103, "4": 104, "5": 105}

# id, scene, ground truth, logits -> exact expected scores
# [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 3.5]
FIXTURE_ITEMS = [
    ("conversation_00001", "scene_000", 1, _sharp(1)),
    ("conversation_00002", "scene_000", 2, _sharp(2)),
    ("conversation_00003", "scene_001", 3, _sharp(3)),
    ("conversation_00004", "scene_001", 4, _sharp(4)),
    ("conversation_00005", "scene_002", 5, _sharp(5)),
    ("conversation_00006", "scene_002", 1, _uniform()),
    ("conversation_00007", "scene_003", 2, _sharp(2)),
    ("conversation_00008", "scene_003", 3, _sharp(3, 4)),
]
EXPECTED_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 3.5]
SCENE_IDS = ["scene_000", "scene_001", "scene_002", "scene_003"]
DRAWS = [[0, 1, 2, 3], [0, 0, 1, 1], [2, 2, 3, 3]]
DERIVED_AT = "2026-08-12T00:00:00-04:00"


def _argmax(logits: dict[str, float]) -> int:
    return int(max(range(1, 6), key=lambda score: (logits[str(score)], -score)))


def _kendall_tau_b_reference(xs: list[float], ys: list[float]) -> float:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(xs) - 1):
        for j in range(i + 1, len(xs)):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
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
    return (concordant - discordant) / math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )


def _percentile_reference(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_fixture(
    root: Path,
    *,
    complete_cell: bool = True,
    item_mutator=None,
    contract_mutator=None,
) -> dict[str, Path]:
    """Materialize a schema-faithful synthetic E1 root with one constrained cell."""
    bootstrap_path = root / "e1_scene_bootstrap_indices.json"
    _write_json(
        bootstrap_path,
        {
            "_meta": {
                "version": 1,
                "method": "scene_cluster_resample_with_replacement",
                "rng": "python_random.Random",
                "seed": 20260811,
                "draw_count": len(DRAWS),
                "scene_count": len(SCENE_IDS),
            },
            "scene_ids": SCENE_IDS,
            "draw_indices": DRAWS,
        },
    )
    complete_dir = root / "cells/iter_100/filepath/constrained"
    pending_dir = root / "cells/iter_100/sft_exact/constrained"
    plan_path = root / "e1_plan.json"
    _write_json(
        plan_path,
        {
            "_meta": {"version": 1, "status": "prospective_outcome_blind_plan"},
            "validation": {"sample_count": len(FIXTURE_ITEMS), "scene_count": len(SCENE_IDS)},
            "bootstrap": {
                "path": str(bootstrap_path),
                "bytes": bootstrap_path.stat().st_size,
                "sha256": _sha(bootstrap_path),
            },
            "factorial": {
                "cells": [
                    {
                        "cell_id": "iter_100__filepath__free",
                        "iteration": 100,
                        "route": "filepath",
                        "interface": "free",
                        "evidence_kind": "historical_replay",
                        "path": str(root / "historical/iter_100"),
                    },
                    {
                        "cell_id": "iter_100__filepath__constrained",
                        "iteration": 100,
                        "route": "filepath",
                        "interface": "constrained",
                        "evidence_kind": "new",
                        "path": str(complete_dir),
                    },
                    {
                        "cell_id": "iter_100__sft_exact__constrained",
                        "iteration": 100,
                        "route": "sft_exact",
                        "interface": "constrained",
                        "evidence_kind": "new",
                        "path": str(pending_dir),
                    },
                ]
            },
        },
    )
    plan_hash = _sha(plan_path)
    _write_json(
        root / "e1_plan_audit.json",
        {"_meta": {"plan_path": str(plan_path), "plan_sha256": plan_hash}, "passed": True, "errors": []},
    )
    amendment_path = root / "amendment.md"
    amendment_path.write_text("# E1 Amendment — Expected-Score Decoding Readout — fixture\n")
    if complete_cell:
        item_files: dict[str, str] = {}
        for sample_id, scene_id, ground_truth, logits in FIXTURE_ITEMS:
            item = {
                "_meta": {"version": 1, "written_at": "2026-08-11T00:00:00-04:00"},
                "id": sample_id,
                "episode_id": f"episode_{sample_id}",
                "scene_id": scene_id,
                "ground_truth": ground_truth,
                "plan_sha256": plan_hash,
                "iteration": 100,
                "route": "filepath",
                "interface": "constrained",
                "checkpoint_tree_sha256": "tree-100",
                "route_metadata": {"media_input": "filepath"},
                "processor_metadata": {"sampling_events": [], "timestamps_s": []},
                "candidate_token_ids": CANDIDATE_IDS,
                "candidate_logits": logits,
                "selected_score": _argmax(logits),
            }
            if item_mutator is not None:
                item_mutator(item)
            item_path = complete_dir / "items" / f"{sample_id}.json"
            _write_json(item_path, item)
            item_files[item_path.name] = _sha(item_path)
        token_contract = {
            "witness_sample_id": FIXTURE_ITEMS[0][0],
            "generation_prefix_token_count": 2,
            "assistant_suffix_token_ids": [30],
            "candidate_token_ids": dict(CANDIDATE_IDS),
        }
        if contract_mutator is not None:
            contract_mutator(token_contract)
        _write_json(
            complete_dir / "manifest.json",
            {
                "_meta": {"version": 1, "completed_at": "2026-08-11T00:00:00-04:00"},
                "plan_sha256": plan_hash,
                "iteration": 100,
                "route": "filepath",
                "interface": "constrained",
                "checkpoint": {"iteration": 100, "path": "/fixture/checkpoint", "tree_sha256": "tree-100"},
                "token_contract": token_contract,
                "sample_count": len(FIXTURE_ITEMS),
                "item_files": dict(sorted(item_files.items())),
                "metrics": {},
            },
        )
    return {"root": root, "plan": plan_path, "amendment": amendment_path, "cell": complete_dir}


def _run(root: Path, *extra: str) -> tuple[int, str]:
    stream = io.StringIO()
    argv = [
        "--root",
        str(root),
        "--amendment",
        str(root / "amendment.md"),
        "--derived-at",
        DERIVED_AT,
        *extra,
    ]
    with contextlib.redirect_stdout(stream):
        code = derive.main(argv)
    return code, stream.getvalue()


class ExpectedScoreNumericsTest(unittest.TestCase):
    def test_uniform_logits_give_exactly_three(self):
        self.assertEqual(derive.expected_score(_uniform()), 3.0)

    def test_extreme_logits_are_numerically_stable(self):
        self.assertEqual(
            derive.expected_score({"1": -1e9, "2": -1e9, "3": -1e9, "4": -1e9, "5": 1e9}),
            5.0,
        )
        self.assertEqual(
            derive.expected_score({"1": 5000.0, "2": -5000.0, "3": -5000.0, "4": -5000.0, "5": -5000.0}),
            1.0,
        )

    def test_moderate_logits_match_direct_softmax(self):
        logits = {"1": 0.5, "2": -1.0, "3": 2.0, "4": 0.0, "5": -0.5}
        weights = [math.exp(logits[str(s)]) for s in range(1, 6)]
        expected = sum(s * w for s, w in zip(range(1, 6), weights)) / sum(weights)
        self.assertAlmostEqual(derive.expected_score(logits), expected, places=12)

    def test_two_way_tie_gives_exact_midpoint(self):
        self.assertEqual(derive.expected_score(_sharp(3, 4)), 3.5)

    def test_rejects_missing_extra_or_nonfinite_logits(self):
        with self.assertRaisesRegex(ValueError, "exactly the five score keys"):
            derive.expected_score({"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0})
        with self.assertRaisesRegex(ValueError, "exactly the five score keys"):
            derive.expected_score({str(s): 0.0 for s in range(1, 7)})
        with self.assertRaisesRegex(ValueError, "not a finite number"):
            derive.expected_score({"1": 0.0, "2": 0.0, "3": float("nan"), "4": 0.0, "5": 0.0})
        with self.assertRaisesRegex(ValueError, "not a finite number"):
            derive.expected_score({"1": 0.0, "2": 0.0, "3": True, "4": 0.0, "5": 0.0})


class RoundingTest(unittest.TestCase):
    def test_round_half_to_even_at_boundaries(self):
        self.assertEqual(derive.round_half_even_clip(1.5), 2)
        self.assertEqual(derive.round_half_even_clip(2.5), 2)
        self.assertEqual(derive.round_half_even_clip(3.5), 4)
        self.assertEqual(derive.round_half_even_clip(4.5), 4)

    def test_rounding_clips_to_score_range(self):
        self.assertEqual(derive.round_half_even_clip(0.4), 1)
        self.assertEqual(derive.round_half_even_clip(5.6), 5)
        self.assertEqual(derive.round_half_even_clip(3.0), 3)


class CorrelationTest(unittest.TestCase):
    def test_perfect_and_reversed_orderings(self):
        xs, ys = [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(derive._pearson_float(xs, ys), 1.0, places=12)
        self.assertAlmostEqual(derive._spearman_float(xs, ys), 1.0, places=12)
        self.assertAlmostEqual(derive._kendall_tau_b_float(xs, ys), 1.0, places=12)
        rev = list(reversed(ys))
        self.assertAlmostEqual(derive._pearson_float(xs, rev), -1.0, places=12)
        self.assertAlmostEqual(derive._spearman_float(xs, rev), -1.0, places=12)
        self.assertAlmostEqual(derive._kendall_tau_b_float(xs, rev), -1.0, places=12)

    def test_kendall_tau_b_known_tied_case(self):
        # pairs: 4 concordant, 0 discordant, 1 x-tie, 1 y-tie -> 4/sqrt(25) = 0.8
        self.assertAlmostEqual(derive._kendall_tau_b_float([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 3.0]), 0.8, places=12)

    def test_spearman_known_tied_case(self):
        self.assertAlmostEqual(derive._spearman_float([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 3.0]), 5.0 / 6.0, places=12)

    def test_zero_variance_is_undefined(self):
        self.assertIsNone(derive._pearson_float([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(derive._spearman_float([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(derive._kendall_tau_b_float([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))


class DerivationEndToEndTest(unittest.TestCase):
    def test_full_derivation_metrics_bootstrap_and_meta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = build_fixture(root)
            code, output = _run(root)
            self.assertEqual(code, 0, output)
            out_path = root / "e1_expected_score_derived.json"
            result = json.loads(out_path.read_text())

            self.assertEqual(result["cell_count"], 1)
            self.assertEqual(
                result["pending_source_cells"],
                [
                    {
                        "cell_id": "iter_100__sft_exact__constrained",
                        "path": str(root / "cells/iter_100/sft_exact/constrained"),
                        "status": "not_started",
                    }
                ],
            )
            cell = result["cells"][0]
            self.assertEqual(cell["cell_id"], "iter_100__filepath__expected_score")
            self.assertEqual(cell["source_cell_id"], "iter_100__filepath__constrained")
            self.assertEqual(cell["route"], "filepath")
            self.assertEqual(cell["sample_count"], len(FIXTURE_ITEMS))

            ids = [item[0] for item in FIXTURE_ITEMS]
            self.assertEqual(cell["expected_scores"], dict(zip(ids, EXPECTED_VALUES)))

            xs = EXPECTED_VALUES
            ys = [float(item[2]) for item in FIXTURE_ITEMS]
            metrics = cell["metrics"]
            self.assertAlmostEqual(metrics["pearson"], statistics.correlation(xs, ys), places=12)
            self.assertAlmostEqual(metrics["spearman"], statistics.correlation(xs, ys, method="ranked"), places=12)
            self.assertAlmostEqual(metrics["kendall_tau_b"], _kendall_tau_b_reference(xs, ys), places=12)
            # rounded (half-to-even): [1, 2, 3, 4, 5, 3, 2, 4] vs gt [1, 2, 3, 4, 5, 1, 2, 3]
            self.assertEqual(metrics["exact_accuracy_rounded"], 6 / 8)
            self.assertEqual(metrics["within_one_accuracy_rounded"], 7 / 8)
            self.assertEqual(
                metrics["per_class_recall_rounded"],
                {"1": 0.5, "2": 1.0, "3": 0.5, "4": 1.0, "5": 1.0},
            )
            self.assertEqual(metrics["expected_score_min"], 1.0)
            self.assertEqual(metrics["expected_score_max"], 5.0)

            hist = metrics["histogram"]
            self.assertEqual(hist["values"], "continuous_expected_scores")
            self.assertEqual(len(hist["bin_edges"]), 17)
            self.assertEqual(hist["bin_edges"][0], 1.0)
            self.assertEqual(hist["bin_edges"][-1], 5.0)
            self.assertEqual(sum(hist["counts"]), len(FIXTURE_ITEMS))
            expected_counts = [0] * 16
            for value in xs:
                expected_counts[min(int((value - 1.0) / 0.25), 15)] += 1
            self.assertEqual(hist["counts"], expected_counts)

            # Bootstrap intervals must come from exactly the three frozen draws.
            by_scene: dict[str, list[int]] = {}
            for index, (_, scene_id, _, _) in enumerate(FIXTURE_ITEMS):
                by_scene.setdefault(scene_id, []).append(index)
            draw_pearsons = []
            for draw in DRAWS:
                dxs, dys = [], []
                for scene_index in draw:
                    for item_index in by_scene[SCENE_IDS[scene_index]]:
                        dxs.append(xs[item_index])
                        dys.append(ys[item_index])
                draw_pearsons.append(statistics.correlation(dxs, dys))
            boot = cell["scene_cluster_bootstrap_95pct"]
            self.assertEqual(boot["interpretation"], "descriptive")
            self.assertEqual(boot["pearson"]["defined_draws"], len(DRAWS))
            self.assertEqual(boot["pearson"]["undefined_draws"], 0)
            self.assertAlmostEqual(
                boot["pearson"]["lower_2_5pct"], _percentile_reference(draw_pearsons, 0.025), places=12
            )
            self.assertAlmostEqual(
                boot["pearson"]["upper_97_5pct"], _percentile_reference(draw_pearsons, 0.975), places=12
            )
            self.assertEqual(boot["spearman"]["defined_draws"], len(DRAWS))

            meta = result["_meta"]
            self.assertEqual(meta["derived_at"], DERIVED_AT)
            self.assertEqual(meta["amendment"]["sha256"], _sha(fixture["amendment"]))
            self.assertEqual(meta["plan"]["sha256"], _sha(fixture["plan"]))
            self.assertEqual(meta["script"]["sha256"], _sha(MODULE_PATH))
            self.assertEqual(meta["source_interface"], "constrained")
            self.assertIs(meta["new_inference_performed"], False)
            self.assertEqual(meta["bootstrap_metrics"], ["pearson", "spearman"])
            self.assertEqual(meta["bootstrap"]["sha256"], _sha(root / "e1_scene_bootstrap_indices.json"))
            source = meta["source_cells"]["iter_100__filepath__expected_score"]
            self.assertEqual(source["manifest_sha256"], _sha(fixture["cell"] / "manifest.json"))
            self.assertIn("item_tree_sha256", source)
            self.assertEqual(meta["readout"], "expected_score")
            self.assertIn("round_half_to_even", meta["rounding"])

            # Source artifacts are untouched: every input byte is as written.
            self.assertEqual(_sha(fixture["plan"]), meta["plan"]["sha256"])
            self.assertEqual(_sha(fixture["cell"] / "manifest.json"), source["manifest_sha256"])

    def test_output_is_write_once_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            code, _ = _run(root)
            self.assertEqual(code, 0)
            code, output = _run(root)
            self.assertEqual(code, 1)
            self.assertIn("write-once", output)
            code, _ = _run(root, "--replace")
            self.assertEqual(code, 0)

    def test_no_completed_cells_exits_cleanly_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root, complete_cell=False)
            code, output = _run(root)
            self.assertEqual(code, 0)
            self.assertIn("no completed constrained cell", output)
            self.assertFalse((root / "e1_expected_score_derived.json").exists())

    def test_derivation_never_draws_random_numbers(self):
        source = MODULE_PATH.read_text()
        self.assertNotIn("import random", source)
        self.assertNotIn("numpy", source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            poison = mock.MagicMock(side_effect=AssertionError("RNG use is forbidden"))
            with (
                mock.patch("random.Random", poison),
                mock.patch("random.random", poison),
                mock.patch("random.randrange", poison),
                mock.patch("random.randint", poison),
                mock.patch("random.choice", poison),
                mock.patch("random.choices", poison),
                mock.patch("random.sample", poison),
                mock.patch("random.shuffle", poison),
            ):
                code, output = _run(root)
            self.assertEqual(code, 0, output)


class RefusalTest(unittest.TestCase):
    def _assert_refuses(self, root: Path, needle: str) -> None:
        code, output = _run(root)
        self.assertEqual(code, 1, output)
        self.assertIn(needle, output)
        self.assertFalse((root / "e1_expected_score_derived.json").exists())

    def test_refuses_item_without_candidate_logits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def drop_logits(item):
                if item["id"] == "conversation_00003":
                    del item["candidate_logits"]

            build_fixture(root, item_mutator=drop_logits)
            self._assert_refuses(root, "no saved candidate logits")

    def test_refuses_multi_token_candidate_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def multi_token(contract):
                contract["candidate_token_ids"] = {**CANDIDATE_IDS, "3": [103, 999]}

            build_fixture(root, contract_mutator=multi_token)
            self._assert_refuses(root, "multi-token")

    def test_refuses_duplicate_candidate_token_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def duplicate(contract):
                contract["candidate_token_ids"] = {**CANDIDATE_IDS, "5": 101}

            build_fixture(root, contract_mutator=duplicate)
            self._assert_refuses(root, "not five distinct")

    def test_refuses_tampered_item_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = build_fixture(root)
            item_path = fixture["cell"] / "items/conversation_00001.json"
            payload = json.loads(item_path.read_text())
            payload["candidate_logits"]["5"] = 9999.0
            _write_json(item_path, payload)
            self._assert_refuses(root, "do not match the cell manifest")

    def test_refuses_selected_score_that_is_not_the_argmax(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def wrong_selection(item):
                if item["id"] == "conversation_00002":
                    item["selected_score"] = 5

            build_fixture(root, item_mutator=wrong_selection)
            self._assert_refuses(root, "argmax")

    def test_refuses_tampered_bootstrap_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            bootstrap_path = root / "e1_scene_bootstrap_indices.json"
            payload = json.loads(bootstrap_path.read_text())
            payload["draw_indices"][0][0] = 3
            _write_json(bootstrap_path, payload)
            self._assert_refuses(root, "bootstrap index artifact does not match")

    def test_refuses_plan_audit_that_does_not_bind_plan_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_fixture(root)
            audit_path = root / "e1_plan_audit.json"
            payload = json.loads(audit_path.read_text())
            payload["_meta"]["plan_sha256"] = "0" * 64
            _write_json(audit_path, payload)
            self._assert_refuses(root, "does not bind the current plan bytes")

    def test_refuses_incomplete_item_set_behind_a_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = build_fixture(root)
            manifest_path = fixture["cell"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["item_files"].popitem()
            _write_json(manifest_path, manifest)
            self._assert_refuses(root, "not the complete")


if __name__ == "__main__":
    unittest.main()
