# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Process-isolation checks for the research environment matrix runner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import verify_matrix
from verify_matrix import run_child


class RunChildTests(unittest.TestCase):
    """Ensure a timed-out Isaac-style process cannot leave a GPU child running."""

    def test_timeout_kills_descendant_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            child_pid_file = root / "child.pid"
            script = (
                "import subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                "open(sys.argv[1], 'w').write(str(child.pid)); "
                "time.sleep(30)"
            )
            _, timed_out, duration_s = run_child(
                [sys.executable, "-c", script, str(child_pid_file)],
                log_path=root / "run.log",
                timeout_s=0.5,
            )

            self.assertTrue(timed_out)
            self.assertLess(duration_s, 10)
            self.assertTrue(child_pid_file.is_file())
            process_stat = Path("/proc") / child_pid_file.read_text().strip() / "stat"
            if process_stat.exists():
                self.assertEqual(process_stat.read_text().split()[2], "Z")

    def test_matrix_continues_after_probe_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "matrix"
            task_ids = ["NDT-Am-EE-Abs-PID-Direct-v0", "PressButton-Am-EE-Abs-PID-Direct-v0"]
            registry = [{"task_id": task_id, "family": task_id.split("-Am-")[0], "robot": "EE"} for task_id in task_ids]

            def fake_child(command, *, log_path, timeout_s):
                if "--list-json" in command:
                    path = Path(command[command.index("--list-json") + 1])
                    path.write_text(json.dumps(registry))
                    return 0, False, 0.1
                task_id = command[command.index("--task") + 1]
                self.assertEqual(command[command.index("--seed") + 1], "42")
                path = Path(command[command.index("--result-json") + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                passed = task_id.startswith("PressButton")
                path.write_text(
                    json.dumps({"task_id": task_id, "status": "passed" if passed else "failed", "steps_completed": 8})
                )
                return (0 if passed else 1), False, 0.2

            argv = [
                "verify_matrix.py",
                "--all",
                "--output-dir",
                str(output_dir),
                "--expect-count",
                "2",
                "--expect-families",
                "2",
            ]
            with patch.object(sys, "argv", argv), patch.object(verify_matrix, "run_child", side_effect=fake_child):
                exit_code = verify_matrix.main()

            report = json.loads((output_dir / "results.json").read_text())
            self.assertEqual(exit_code, 1)
            self.assertEqual(report["completed_probe_count"], 2)
            self.assertEqual(report["passed_count"], 1)
            self.assertEqual(report["failed_count"], 1)
            self.assertEqual(report["probe_config"]["seed"], 42)
            self.assertTrue(all(row["seed"] == 42 for row in report["results"]))


if __name__ == "__main__":
    unittest.main()
