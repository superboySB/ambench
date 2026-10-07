# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Small provenance checks for merged research matrix reports."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from merge_matrix_results import merge_reports


class MergeMatrixResultsTests(unittest.TestCase):
    """Check that a later passing probe keeps its failed history."""

    def test_retry_preserves_timeout_and_checks_child_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            task_id = "PressButton-Am-EE-Abs-PID-Direct-v0"
            reports = []
            for index, (parent_status, child_status, steps) in enumerate(
                (("timeout", "failed", 0), ("passed", "passed", 8))
            ):
                run_dir = root / f"run-{index}"
                run_dir.mkdir()
                log_path = run_dir / "probe.log"
                log_path.write_text("probe log\n")
                record_path = run_dir / "probe.json"
                record_path.write_text(
                    json.dumps({"task_id": task_id, "status": child_status, "steps_completed": steps})
                )
                report_path = run_dir / "results.json"
                report_path.write_text(
                    json.dumps({
                        "registry_count": 106,
                        "completed_probe_count": 1,
                        "generated_at_utc": f"2026-09-30T00:00:0{index}+00:00",
                        "results": [{
                            "task_id": task_id,
                            "family": "PressButton",
                            "robot": "EE",
                            "status": parent_status,
                            "steps_completed": steps,
                            "exit_code": 0 if index else -2,
                            "result_path": str(record_path),
                            "log_path": str(log_path),
                            "error": "initial timeout" if index == 0 else "",
                        }],
                    })
                )
                reports.append(report_path)

            args = argparse.Namespace(reports=reports, min_steps=8)
            merged = merge_reports(args)
            self.assertEqual(merged["passed_count"], 1)
            self.assertEqual(merged["results"][0]["selected_source_report"], str(reports[1]))
            self.assertEqual([attempt["status"] for attempt in merged["results"][0]["attempts"]], ["timeout", "passed"])

            retry_record = root / "run-1" / "probe.json"
            retry_record.write_text(json.dumps({"task_id": task_id, "status": "passed", "steps_completed": 7}))
            with self.assertRaisesRegex(ValueError, "steps_completed differs"):
                merge_reports(args)


if __name__ == "__main__":
    unittest.main()
