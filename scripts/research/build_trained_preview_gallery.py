# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Publish source-only metadata for existing trained UAQuad recordings.

Run after ``export_usage_animations.py`` with the tracked trained animation specs.
Original videos and previews remain in ignored output directories. This builder
checks completed native summaries and existing first/last decode receipts, hashes
every original video again, and writes an upload list for tagged Release assets.
It does not upload assets, run a simulator, or replace the stricter full training
and selected-weight verification workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

CAMERAS = ("scene_camera", "base_camera", "ee_camera")
SOURCES = (
    (
        "act",
        "selection",
        "act-corrected-eight-candidates-selection",
        "runtime_outputs/act-corrected-deployment-20261013/completed-080-videos-final.receipt.json",
    ),
    (
        "act",
        "sealed",
        "act-corrected-sealed30-final",
        "runtime_outputs/act-corrected-deployment-20261013/sealed30-videos.receipt.json",
    ),
    ("dp", "selection", "dp-native-selection80", "runtime_outputs/dp-native-evaluation/selection80-videos.json"),
    ("dp", "sealed", "dp-native-sealed30", "runtime_outputs/dp-native-evaluation/sealed30-videos.json"),
)


def sha256(path: Path) -> str:
    """Hash an artifact without loading the complete recording into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, count: int) -> list[float]:
    """Compute the two-sided Wilson interval with z=1.96."""
    proportion = successes / count
    denominator = 1 + 1.96**2 / count
    center = (proportion + 1.96**2 / (2 * count)) / denominator
    radius = 1.96 * math.sqrt(proportion * (1 - proportion) / count + 1.96**2 / (4 * count**2)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def main() -> int:
    """Build metadata and an upload plan from the unchanged native recordings."""
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "usage_assets/trained")
    parser.add_argument("--upload-plan", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", args.release_tag):
        parser.error("Release tag must contain only letters, digits, dots, underscores and hyphens")
    template = (args.output_dir / "index.template.html").read_text()
    assert template.count("/* VERIFIED_RECORDINGS */ null") == 1
    base = f"https://github.com/superboySB/ambench/releases/download/{args.release_tag}/"
    research = root / "outputs/research/fulltrain-20261013"
    validation_path = root / "usage_assets/trained/validation.json"
    validation = json.loads(validation_path.read_text())
    validation_reference = {
        "path": validation_path.relative_to(root).as_posix(),
        "sha256": sha256(validation_path),
        "url": base + "validation.json",
    }
    split = validation["data_split"]
    assert split["train_episode_indices"] == list(range(80))
    assert split["offline_validation_episode_indices"] == list(range(80, 100))
    assert split["normalization_uses_train_only"] is True
    assert validation["selection_seeds"] == list(range(23001, 23011))
    assert validation["final_seeds"] == list(range(33001, 33031))
    source_hashes = {item["path"]: item["sha256"] for item in validation["sources"]}
    for path, expected_hash in source_hashes.items():
        assert sha256(root / path) == expected_hash
    models = {item["family"]: item for item in validation["models"]}
    locks = {family: json.loads((root / model["selection_lock_path"]).read_text()) for family, model in models.items()}
    for family, lock in locks.items():
        model = models[family]
        assert lock["status"] == "locked_before_sealed30"
        assert lock["locked_at_utc"] == model["selection_locked_at_utc"] < model["final_started_at_utc"]
        assert lock["selected_spec"]["training_steps"] == model["selected_optimizer_updates"]
    dp_protocol_path = research / "dp-native-evaluation-plan/protocol.json"
    dp_protocol = json.loads(dp_protocol_path.read_text())
    assert dp_protocol["dataset_sha256"] == split["dp_zarr_sha256"]
    dp_source_path = research / "runtime_outputs/dp-native-evaluation/training-source-manifest.json"
    dp_source = json.loads(dp_source_path.read_text())
    assert dp_source["source_commit"] == models["dp"]["evaluation_source_commit"]
    # The frozen policy archive contains ambench_learn; simulator files remain
    # separately bound by the original native protocol's complete hash map.
    assert all(
        dp_source["files"][path] == digest
        for path, digest in dp_protocol["source_sha256"].items()
        if path.startswith("source/ambench_learn/")
    )
    cases, sources, uploads = [], [], []
    for family, phase, folder, receipt_relative in SOURCES:
        driver_path = research / folder / "results.json"
        receipt_path = research / receipt_relative
        driver = json.loads(driver_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        expected = 80 if phase == "selection" else 30
        assert driver["status"] == "completed" and len(driver["results"]) == expected
        assert receipt["status"] == "passed" and receipt["results_sha256"] == sha256(driver_path)
        decoded = {(row["name"], row["camera"]): row for row in receipt["videos"]}
        sources.append({
            "family": family,
            "phase": phase,
            "path": driver_path.relative_to(root).as_posix(),
            "sha256": sha256(driver_path),
            "decode_receipt_sha256": sha256(receipt_path),
        })
        for row_index, row in enumerate(driver["results"]):
            spec = row["spec"]
            model = models[family]
            lock = locks[family]
            summary_path = root / row["eval_summary"]
            summary = json.loads(summary_path.read_text())
            assert row["status"] == "completed" and row["eval_exit_code"] == 0
            assert sha256(summary_path) == row["eval_summary_sha256"]
            assert summary["status"] == "completed" and summary["error"] is None and len(summary["rollouts"]) == 1
            rollout = summary["rollouts"][0]
            assert (rollout["termination_reason"], rollout["executed_steps"]) == (row["outcome"], row["steps"])
            assert (summary["metadata"]["Seed"], summary["metadata"]["Task"]) == (spec["seed"], spec["task_id"])
            assert set(row["videos"]) == set(CAMERAS) and spec["num_rerenders_on_reset"] == 1
            assert spec["task_id"] == validation["task_id"]
            assert spec["action_frame"] == validation["action_frame"]
            assert spec["observation_ee_quaternion_frame"] == validation["observation_ee_quaternion_frame"]
            assert (
                spec.get("training_source_commit", spec.get("baseline_source_commit"))
                == model["training_source_commit"]
            )
            if phase == "sealed":
                assert spec["selection_lock_sha256"] == source_hashes[model["selection_lock_path"]]
                assert spec["training_steps"] == model["selected_optimizer_updates"]
                final = next(item for item in model["final_trials"] if item["trial_id"] == row["name"])
                assert (
                    final["seed"],
                    final["termination_reason"],
                    final["executed_steps"],
                    final["eval_summary_sha256"],
                ) == (spec["seed"], row["outcome"], row["steps"], row["eval_summary_sha256"])
            if family == "act":
                files = spec["checkpoint_files_sha256"]
                normalizer_name = "policy_preprocessor_step_5_normalizer_processor.safetensors"
                normalizer_hash = files[normalizer_name]
                assert normalizer_hash == files["policy_postprocessor_step_0_unnormalizer_processor.safetensors"]
                assert spec["canonical_info_sha256"] == split["canonical_info_sha256"]
                if phase == "sealed":
                    assert files == lock["selected_spec"]["checkpoint_files_sha256"]
                    assert files["model.safetensors"] == model["selected_weight_sha256"]
                normalizer_reference = {
                    "path": driver_path.relative_to(root).as_posix(),
                    "sha256": sha256(driver_path),
                    "json_path": ["results", row_index, "spec", "checkpoint_files_sha256", normalizer_name],
                }
            else:
                prefix = f"epoch{spec['candidate_completed_epochs']:03d}-actual{spec['training_steps']:06d}"
                candidate_root = research / "runtime_outputs/dp-native-evaluation/bound-candidates"
                budget_path = candidate_root / f"{prefix}-budget.json"
                inference_path = candidate_root / f"{prefix}-ema.manifest.json"
                budget = json.loads(budget_path.read_text())
                inference = json.loads(inference_path.read_text())
                assert sha256(budget_path) == spec["actual_budget_receipt_sha256"]
                assert sha256(inference_path) == spec["inference_manifest_sha256"]
                assert budget["status"] == "passed" and budget["actual_optimizer_updates"] == spec["training_steps"]
                assert budget["dataset_split"]["train_episode_indices"] == split["train_episode_indices"]
                assert budget["dataset_split"]["val_episode_indices"] == split["offline_validation_episode_indices"]
                assert inference["bundle_sha256"] == spec["inference_bundle_sha256"]
                assert inference["source_manifest_sha256"] == sha256(dp_source_path)
                assert spec["native_protocol_sha256"] == source_hashes[dp_protocol_path.relative_to(root).as_posix()]
                if phase == "sealed":
                    for field in ("inference_bundle_sha256", "inference_manifest_sha256", "parent_checkpoint_sha256"):
                        assert spec[field] == lock["selected_spec"][field]
                    assert spec["inference_bundle_sha256"] == model["selected_weight_sha256"]
                normalizer_hash = budget["normalizer_sha256"]
                normalizer_reference = {
                    "path": budget_path.relative_to(root).as_posix(),
                    "sha256": sha256(budget_path),
                    "json_path": ["normalizer_sha256"],
                }
            case = {
                "id": row["name"],
                "family": family,
                "phase": phase,
                "seed": spec["seed"],
                "outcome": row["outcome"],
                "steps": row["steps"],
                "simulation_duration_s": row["steps"] / 120,
                "task_id": spec["task_id"],
                "candidate_updates": spec["training_steps"],
                "candidate_epochs": spec.get("candidate_completed_epochs"),
                "run_budget": {
                    "updates": spec.get("run_budget_completed"),
                    "epochs": spec.get("training_run_budget_epochs"),
                },
                "selection_lock_sha256": spec.get("selection_lock_sha256"),
                "action_frame": spec["action_frame"],
                "observation_ee_quaternion_frame": spec["observation_ee_quaternion_frame"],
                "training_source_commit": spec.get("training_source_commit", spec.get("baseline_source_commit")),
                "evaluation_source_commit": spec.get("evaluation_source_commit", model["evaluation_source_commit"]),
                "evaluation_source_reference": {
                    **validation_reference,
                    "json_path": ["models", 0 if family == "act" else 1, "evaluation_source_commit"],
                },
                "native_protocol_sha256": spec.get("native_protocol_sha256"),
                "actual_budget_receipt_sha256": spec.get("actual_budget_receipt_sha256"),
                "inference_manifest_sha256": spec.get("inference_manifest_sha256"),
                "checkpoint_files_sha256": spec.get("checkpoint_files_sha256"),
                "data_split_reference": {**validation_reference, "json_path": ["data_split"]},
                "canonical_info_sha256": split["canonical_info_sha256"],
                "canonical_info_reference": {
                    **validation_reference,
                    "json_path": ["data_split", "canonical_info_sha256"],
                },
                "dp_zarr_sha256": split["dp_zarr_sha256"] if family == "dp" else None,
                "dp_zarr_reference": (
                    {**validation_reference, "json_path": ["data_split", "dp_zarr_sha256"]} if family == "dp" else None
                ),
                "normalizer_sha256": normalizer_hash,
                "normalizer_reference": normalizer_reference,
                "inference_seed": spec["inference_seed"],
                "inference_device": spec["inference_device"],
                "training_seed": spec["training_seed"],
                "attempt_selection": spec["attempt_selection"],
                "native_summary_sha256": row["eval_summary_sha256"],
                "tracking_sha256": row["tracking_sha256"],
                "model_sha256": spec.get(
                    "inference_bundle_sha256", spec.get("checkpoint_files_sha256", {}).get("model.safetensors")
                ),
                "views": [],
            }
            for camera in CAMERAS:
                proof = decoded[(row["name"], camera)]
                path = root / row["videos"][camera]
                assert proof["path"] == row["videos"][camera] and proof["outcome"] == row["outcome"]
                assert proof["passed"] and proof["first_decoded"] and proof["last_decoded"]
                assert path.stat().st_size == proof["bytes"] and sha256(path) == proof["sha256"]
                filename = f"{row['name']}_{camera}.mp4"
                view = {
                    "camera": camera,
                    "asset_name": filename,
                    "url": base + filename,
                    **{
                        key: proof[key] for key in ("sha256", "bytes", "frames", "fps", "width", "height", "duration_s")
                    },
                    "source": row["videos"][camera],
                    "first_last_decode_verified": True,
                    "full_decode_verified": False,
                }
                case["views"].append(view)
                uploads.append({
                    "name": filename,
                    "path": str(path),
                    "sha256": proof["sha256"],
                    "bytes": proof["bytes"],
                    "outcome": row["outcome"],
                    "camera": camera,
                    "content_type": "video/mp4",
                })
            cases.append(case)
    previews = json.loads((args.preview_dir / "manifest.json").read_text())
    by_case = {case["id"]: case for case in cases}
    for preview in previews:
        case = by_case[preview["trial_id"]]
        original = next(view for view in case["views"] if view["camera"] == preview["camera"])
        assert preview["source_sha256"] == original["sha256"] and preview["outcome_value"] == case["outcome"]
        assert preview["source_report_sha256"] == case["native_summary_sha256"]
        assert preview["mp4_full_decode_passed"] is True
        for prefix in ("", "mp4_"):
            assert preview[prefix + "source_frame_indices"][0] == 0
            assert preview[prefix + "source_frame_indices"][-1] == preview["source_frame_count"] - 1
        for extension in ("gif", "mp4"):
            path = args.preview_dir / preview[extension]
            assert (
                sha256(path) == preview[extension + "_sha256"] and path.stat().st_size == preview[extension + "_bytes"]
            )
            preview[extension + "_url"] = base + path.name
            uploads.append({
                "name": path.name,
                "path": str(path.resolve()),
                "sha256": preview[extension + "_sha256"],
                "bytes": path.stat().st_size,
                "outcome": preview["outcome_value"],
                "camera": preview["camera"],
                "content_type": "video/mp4" if extension == "mp4" else "image/gif",
            })
    scores = {}
    for family in ("act", "dp"):
        final = [case for case in cases if case["family"] == family and case["phase"] == "sealed"]
        assert {case["seed"] for case in final} == set(range(33001, 33031))
        successes = sum(case["outcome"] == "success" for case in final)
        assert successes == models[family]["final_successes"]
        selection = [case for case in cases if case["family"] == family and case["phase"] == "selection"]
        assert sum(case["outcome"] == "success" for case in selection) == models[family]["selection_successes"]
        candidates = {case["candidate_updates"] for case in selection}
        assert len(candidates) == 8
        for updates in candidates:
            assert {case["seed"] for case in selection if case["candidate_updates"] == updates} == set(
                validation["selection_seeds"]
            )
        scores[family] = {"successes": successes, "attempts": len(final), "wilson95": wilson(successes, len(final))}
    payload = {
        "schema_version": 1,
        "release_tag": args.release_tag,
        "validation": validation_reference,
        "data_split": split,
        "scores": scores,
        "scope": (
            "UAQuad PressButton; all 80 selection and 30 sealed first attempts per model, corrected reset rerender=1."
        ),
        "media_validation": (
            "Original videos: existing first/last decode receipts, bytes rehashed at assembly. Representative previews:"
            " fully decoded by exporter. Public delivery is checked after upload."
        ),
        "preview_selection": (
            "DP seed indices first/middle/last (33001,33015,33030); ACT first seed (33001). Three synchronized cameras"
            " per trial; previews do not define the score denominator."
        ),
        "sources": sources,
        "cases": cases,
        "previews": previews,
    }
    assert len({item["name"] for item in uploads}) == len(uploads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for path in (manifest_path, validation_path):
        uploads.append({
            "name": path.name,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "content_type": "application/json",
        })
    args.upload_plan.parent.mkdir(parents=True, exist_ok=True)
    args.upload_plan.write_text(
        json.dumps({"repository": "superboySB/ambench", "tag": args.release_tag, "assets": uploads}, indent=2) + "\n"
    )
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", r"\u003c")
    (args.output_dir / "index.html").write_text(template.replace("/* VERIFIED_RECORDINGS */ null", embedded))
    print(
        f"Verified {len(cases)} cases, {sum(len(case['views']) for case in cases)} original videos,"
        f" {len(previews)} previews"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
