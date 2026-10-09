# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Record bounded UAQuad rollouts with separately restarted remote CPU policies.

Run this standard-library entrypoint on the host after the scripted recording
batch finishes. Isaac and model dependencies remain in their existing Docker
containers. Establish the SSH tunnel with tools/research/tunnel_policy.sh first.
Every attempt has its own output directory; task failures are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("scene_camera", "base_camera", "ee_camera")
TASK_ID = "PressButton-Am-UAQuad-Abs-PID-Direct-v0"
SSH_OPTIONS = ["-p", "22", "-o", "BatchMode=yes", "-o", "PasswordAuthentication=no"]
SSH_TARGET = "tencent-86"
SERVER_IDENTITY_CODE = """
import json
import os
from pathlib import Path
pid = int(Path('/data/outputs/policy-server.pid').read_text())
os.kill(pid, 0)
root = Path('/proc') / str(pid)
argv = root.joinpath('cmdline').read_bytes().decode().rstrip('\\0').split('\\0')
environment = dict(item.split('=', 1) for item in root.joinpath('environ').read_bytes().decode().split('\\0') if '=' in item)
sockets = set()
for descriptor in root.joinpath('fd').iterdir():
    try:
        target = os.readlink(descriptor)
    except FileNotFoundError:
        continue
    if target.startswith('socket:['):
        sockets.add(target[8:-1])
ports = set()
for table in ('tcp', 'tcp6'):
    for line in Path('/proc/net', table).read_text().splitlines()[1:]:
        fields = line.split()
        if fields[3] == '0A' and fields[9] in sockets:
            ports.add(int(fields[1].rsplit(':', 1)[1], 16))
print(json.dumps({
    'pid': pid,
    'start_time_ticks': root.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19],
    'argv': argv,
    'listening_ports': sorted(ports),
    'environment': {key: environment.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED')},
}))
"""


def run_logged(command: list[str], path: Path, timeout_s: int) -> int:
    """Keep exact argument lists and full output for one bounded stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".argv.json").write_text(json.dumps(command, indent=2) + "\n")
    with path.open("w") as log:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    return result.returncode


def copy_server_log(target: str, family: str, path: Path) -> None:
    """Copy this project's policy server log without reading SSH credentials."""
    remote = (
        "if docker info >/dev/null 2>&1; then d=docker; else d='sudo -n docker'; fi; "
        f"$d exec ambench-policy-research cat /data/outputs/policy-server-{family}.log"
    )
    if run_logged(["ssh", *SSH_OPTIONS, target, remote], path, 60):
        raise RuntimeError("Could not save the remote policy server log")


def read_server_identity(spec: dict, path: Path) -> dict:
    """Verify which live CPU process actually owns the inference listener."""
    remote = (
        "if docker info >/dev/null 2>&1; then d=docker; else d='sudo -n docker'; fi; "
        "$d exec ambench-policy-research /opt/venvs/act/bin/python -c "
        + shlex.quote(SERVER_IDENTITY_CODE)
    )
    result = subprocess.run(
        ["ssh", *SSH_OPTIONS, SSH_TARGET, remote], capture_output=True, text=True, timeout=60, check=True
    )
    identity = json.loads(result.stdout)
    path.write_text(json.dumps(identity, indent=2) + "\n")
    arguments = identity["argv"]
    if identity["environment"] != {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": "42"}:
        raise ValueError("Policy process does not have the expected CPU environment")
    port = 8000 if spec["family"] == "pi" else 8001
    if port not in identity["listening_ports"]:
        raise ValueError("The expected policy process does not own the inference port")
    if spec["family"] == "pi":
        required = [f"--policy.config={spec['openpi_config']}", f"--policy.dir={spec['checkpoint']}"]
        if any(argument not in arguments for argument in required):
            raise ValueError("OpenPI process loaded a different configuration or checkpoint")
    else:
        for option, value in (("--policy", spec["family"]), ("--checkpoint", spec["checkpoint"]), ("--device", "cpu")):
            if option not in arguments or arguments[arguments.index(option) + 1] != value:
                raise ValueError("Policy process arguments differ from the specification")
    return identity


def validate_rollout(path: Path, spec: dict, task_id: str = TASK_ID) -> dict:
    """Require a completed real rollout and all three recordings."""
    report_path = path / "eval_summary.json"
    payload = json.loads(report_path.read_text())
    metadata = payload["metadata"]
    if payload["status"] != "completed" or payload["error"] is not None:
        raise ValueError("Evaluator did not complete")
    if metadata["Task"] != task_id or metadata["Seed"] != spec["seed"]:
        raise ValueError("Runtime task or seed differs from its specification")
    if metadata["Policy ID"] != spec["name"] or sorted(metadata["Video cameras"]) != sorted(CAMERAS):
        raise ValueError("Runtime policy identifier or cameras differ")
    if metadata["Action semantics"] != "ee_absolute" or metadata["Disturbance"]:
        raise ValueError("Unexpected action semantics or disturbance condition")
    if metadata["Episode length override (s)"] != spec["episode_length_s"]:
        raise ValueError("Runtime episode length differs")
    if len(payload["rollouts"]) != 1 or payload["summary"]["num_rollouts"] != 1:
        raise ValueError("Expected exactly one rollout per attempt")
    rollout = payload["rollouts"][0]
    if rollout["termination_reason"] not in {"success", "timeout"}:
        raise ValueError("Unexpected termination reason")
    if rollout["success"] != (rollout["termination_reason"] == "success"):
        raise ValueError("Task success and termination reason disagree")
    if not 0 < rollout["executed_steps"] <= round(spec["episode_length_s"] * 120):
        raise ValueError("Invalid executed step count")
    if spec["family"] in {"act", "dp"}:
        if metadata["Server metadata"]["checkpoint"] != spec["checkpoint"]:
            raise ValueError("Server loaded a different checkpoint")
    videos = {}
    for camera in CAMERAS:
        matches = list((path / "videos").glob(f"*-{camera}-env0-eps0.mp4"))
        if len(matches) != 1 or matches[0].stat().st_size == 0:
            raise ValueError(f"Missing or ambiguous camera recording: {camera}")
        videos[camera] = str(matches[0].relative_to(REPO_ROOT))
    tracking = path / "tracking" / "tracking.jsonl"
    if not tracking.is_file() or not (path / "env_cfg.yaml").is_file():
        raise ValueError("Missing tracking or runtime environment snapshot")
    return {
        "eval_summary": str(report_path.relative_to(REPO_ROOT)),
        "eval_summary_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "tracking_sha256": hashlib.sha256(tracking.read_bytes()).hexdigest(),
        "outcome": rollout["termination_reason"],
        "steps": rollout["executed_steps"],
        "tracking": rollout.get("tracking"),
        "videos": videos,
    }


def verify_simulation_gpu(sim_container: str, path: Path, minimum_free_vram_mib: int | None) -> dict | None:
    """Reject simulation overlap and optionally admit other workloads by free VRAM."""
    gpu = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    (path / "gpu-processes-before.txt").write_text(gpu.stdout)
    if minimum_free_vram_mib is None and gpu.stdout.strip():
        raise RuntimeError("Local GPU 0 already has a compute process; finish it before recording")
    if minimum_free_vram_mib is not None:
        container_processes = subprocess.run(
            ["docker", "top", sim_container, "-eo", "pid,ppid,args"],
            capture_output=True,
            text=True,
            check=True,
        )
        (path / "sim-processes-before.txt").write_text(container_processes.stdout)
        owned_pids = {line.split()[0] for line in container_processes.stdout.splitlines()[1:] if line.strip()}
        gpu_processes = list(csv.reader(gpu.stdout.splitlines()))
        if any(fields[0].strip() in owned_pids for fields in gpu_processes):
            raise RuntimeError("The simulation container still owns an active GPU process")
        memory = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        fields = next(csv.reader(memory.stdout.splitlines()))
        baseline = {
            "gpu_index": int(fields[0]),
            "name": fields[1].strip(),
            "total_mib": int(fields[2]),
            "used_mib": int(fields[3]),
            "free_mib": int(fields[4]),
            "required_free_mib": minimum_free_vram_mib,
            "other_compute_processes": [
                {"pid": int(item[0]), "process_name": item[1].strip()} for item in gpu_processes
            ],
            "sim_owned_compute_processes": [],
        }
        (path / "gpu-baseline-before.json").write_text(json.dumps(baseline, indent=2) + "\n")
        if baseline["free_mib"] < minimum_free_vram_mib:
            raise RuntimeError(
                f"GPU 0 has {baseline['free_mib']} MiB free; this one-environment recording requires "
                f"at least {minimum_free_vram_mib} MiB before startup"
            )
    return baseline if minimum_free_vram_mib is not None else None


def main(
    *,
    task_id: str = TASK_ID,
    asset_subdir: str = "uaquad",
    minimum_free_vram_mib: int | None = None,
    eval_timeout_s: int = 900,
) -> int:
    """Run selected trials serially, restarting seeded inference for each seed."""
    parser = argparse.ArgumentParser(description=__doc__.replace("UAQuad", task_id.split("-")[2]))
    parser.add_argument(
        "--specs-json", type=Path, default=REPO_ROOT / "usage_assets" / asset_subdir / "policy_specs.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-container", default="ambench-sim-research")
    parser.add_argument("--name", action="append", default=[])
    args = parser.parse_args()
    specs = json.loads(args.specs_json.read_text())
    names = [spec["name"] for spec in specs]
    if not names or len(set(names)) != len(names) or set(args.name) - set(names):
        parser.error("Specifications need unique names and selected names must exist")
    if any(not re.fullmatch(r"[a-z0-9_-]+", name) for name in names):
        parser.error("Unsafe trial name")
    selected = [spec for spec in specs if not args.name or spec["name"] in args.name]
    for spec in selected:
        if spec["family"] not in {"act", "dp", "pi"} or spec["inference_seed"] != 42:
            parser.error("Supported families are ACT/DP/PI with the server's fixed seed 42")
        if spec["inference_device"] != "cpu" or spec["task_id"] != task_id:
            parser.error(f"This experiment requires CPU inference and {task_id}")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(REPO_ROOT / "outputs"):
        parser.error("Raw artifacts must stay inside ignored outputs/")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Use a new empty --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "results": [], "status": "running"}
    for index, spec in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {spec['name']}", flush=True)
        path = output_dir / spec["name"]
        path.mkdir()
        start = time.monotonic()
        row = {"name": spec["name"], "spec": spec, "status": "failed"}
        family = spec["family"]
        try:
            active_batch = subprocess.run(
                [
                    "docker",
                    "exec",
                    args.sim_container,
                    "pgrep",
                    "-f",
                    (
                        "record_usage_variants[.]py|record_demos_scripted[.]py|verify_matrix[.]py|verify_scripted[.]py|"
                        "ambench_learn[.]policies[.](act|dp|pi)[.]eval"
                    ),
                ],
                capture_output=True,
                text=True,
            )
            if active_batch.returncode not in {0, 1}:
                raise RuntimeError("Could not inspect the simulation container's active batches")
            if active_batch.returncode == 0:
                raise RuntimeError("A simulation recording or verification batch is still running")
            baseline = verify_simulation_gpu(args.sim_container, path, minimum_free_vram_mib)
            if baseline is not None:
                row["gpu_baseline"] = baseline
            start_command = ["bash", "tools/research/start_policy.sh", "--cpu", family]
            if family == "pi":
                start_command.append(spec["openpi_config"])
            start_command.append(spec["checkpoint"])
            if run_logged(start_command, path / "server-start.log", 90):
                raise RuntimeError("Remote policy server failed to start")
            if run_logged(["bash", "tools/research/wait_policy.sh", family, "600"], path / "server-ready.log", 630):
                raise RuntimeError("Remote policy server never became ready")
            identity_before = read_server_identity(spec, path / "server-identity-before.json")
            container_output = "/workspace/ambench/" + str(path.relative_to(REPO_ROOT))
            command = [
                "docker",
                "exec",
                args.sim_container,
                "timeout",
                "--kill-after=20s",
                f"{eval_timeout_s}s",
                "python",
                "-m",
                f"ambench_learn.policies.{family}.eval",
                "--task",
                task_id,
                "--num-envs",
                "1",
                "--num-rollouts",
                "1",
                "--seed",
                str(spec["seed"]),
                "--episode-length-s",
                str(spec["episode_length_s"]),
                "--policy-id",
                spec["name"],
                "--save-video",
                "--video-camera-names",
                *CAMERAS,
                "--output-dir",
                container_output,
                "--progress-every",
                "240",
                "--headless",
                "--device",
                "cuda:0",
            ]
            if family in {"act", "dp"}:
                command.extend(["--remote-url", "http://127.0.0.1:8001"])
            else:
                command.extend(["--host", "127.0.0.1", "--port", "8000", "--prompt", "press the button"])
            if family in {"act", "pi"}:
                command.extend(["--policy-target-hz", "20", "--n-action-steps", "8"])
            row["eval_exit_code"] = run_logged(command, path / "eval.log", eval_timeout_s + 60)
            identity_after = read_server_identity(spec, path / "server-identity-after.json")
            if (identity_before["pid"], identity_before["start_time_ticks"]) != (
                identity_after["pid"],
                identity_after["start_time_ticks"],
            ):
                raise ValueError("Inference process changed during the rollout")
            row["server_identity"] = identity_before
            copy_server_log(SSH_TARGET, family, path / "server-inference.log")
            server_log = (path / "server-inference.log").read_text()
            if "REMOTE_INFERENCE_DEVICE=cpu" not in server_log or "REMOTE_INFERENCE_SEED=42" not in server_log:
                raise ValueError("Remote CPU device and seed were not recorded")
            if row["eval_exit_code"] != 0:
                raise RuntimeError("Evaluator failed or exceeded its wall time limit")
            row.update(validate_rollout(path, spec, task_id))
            row["status"] = "completed"
            print(f"  completed: {row['outcome']}, {row['steps']} steps", flush=True)
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
            row["error"] = str(error)
            print(f"  failed: {error}", flush=True)
        row["wall_duration_s"] = time.monotonic() - start
        report["results"].append(row)
        (output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if row["status"] != "completed":
            break  # Inspect an infrastructure failure before launching another Isaac process.
    report["status"] = (
        "completed"
        if len(report["results"]) == len(selected) and all(row["status"] == "completed" for row in report["results"])
        else "failed"
    )
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
