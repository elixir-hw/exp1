#!/usr/bin/env python3
"""Launch two independent Experiment 1 shards and merge them after both finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from run_experiment1 import (
    ROOT, complete_rollout, ensure_manifests, load_json, merge_rollout_results,
    prepare_text_cache, resolve_config_paths, selected_tasks, summarize, validate,
    write_json,
)

DEFAULT_REMOTE = "chw@10.11.141.54"
DEFAULT_REMOTE_ROOT = Path("/home/chw/code/packages/wm-function/exp1")


def checked(command: list[str], **kwargs) -> None:
    print("[distributed]", shlex.join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def remote_command(host: str, command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(command)],
        check=check, capture_output=True, text=True,
    )


def split_at(num_rollouts: int, local_gpus: int, remote_gpus: int) -> int:
    if num_rollouts < 2:
        raise ValueError("Distributed evaluation needs at least two rollouts per state")
    return min(num_rollouts - 1, max(1, round(num_rollouts * local_gpus / (local_gpus + remote_gpus))))


def status_path(run_dir: Path, role: str) -> Path:
    return run_dir / f"worker_{role}_status.json"


def worker(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "run_experiment1.py"), "run",
        "--config", str(args.config.resolve()), "--run-dir", str(run_dir),
        "--rollout-start", str(args.rollout_start), "--rollout-end", str(args.rollout_end),
        "--parallel-gpus", args.gpus, "--skip-text-cache",
    ]
    rc = 1
    try:
        rc = subprocess.run(command, check=False).returncode
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        write_json(status_path(run_dir, args.role), {
            "role": args.role, "exit_code": rc,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "rollout_start": args.rollout_start, "rollout_end": args.rollout_end,
        })
    if rc:
        raise RuntimeError(f"{args.role} worker exited with code {rc}")


def remote_status(host: str, remote_run_dir: Path) -> dict | None:
    path = remote_run_dir / "worker_remote_status.json"
    result = remote_command(host, ["cat", str(path)], check=False)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None


def wait_for_workers(args: argparse.Namespace) -> None:
    local_path = status_path(args.run_dir, "local")
    while True:
        local = load_json(local_path) if local_path.is_file() else None
        remote = remote_status(args.remote_host, args.remote_run_dir)
        if local and remote:
            if local["exit_code"] or remote["exit_code"]:
                raise RuntimeError(f"Worker failure: local={local['exit_code']} remote={remote['exit_code']}")
            return
        if not local and subprocess.run(
            ["tmux", "has-session", "-t", args.local_session],
            capture_output=True, check=False,
        ).returncode:
            if not local_path.is_file():
                raise RuntimeError("Local worker tmux exited without a completion marker")
        if not remote:
            alive = remote_command(
                args.remote_host, ["tmux", "has-session", "-t", args.remote_session], check=False,
            )
            if alive.returncode not in (0, 255) and remote_status(args.remote_host, args.remote_run_dir) is None:
                raise RuntimeError("Remote worker tmux exited without a completion marker")
        print(f"[distributed] waiting: local={bool(local)} remote={bool(remote)}", flush=True)
        time.sleep(30)


def import_and_merge(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    plan = load_json(run_dir / "distributed_plan.json")
    remote_copy = run_dir / "remote_import" / "openwam"
    remote_copy.mkdir(parents=True, exist_ok=True)
    checked([
        "rsync", "-a", "--no-owner", "--no-group",
        f"{args.remote_host}:{args.remote_run_dir}/openwam/", str(remote_copy) + "/",
    ])
    tasks = plan["tasks"]
    methods = plan["methods"]
    states = plan["states_per_task"]
    rollouts = plan["rollouts_per_state"]
    cut = plan["local_rollout_end"]
    for task in tasks:
        for state_id in range(states):
            manifest = load_json(Path(plan["local_manifest_root"]) / task / plan["mode"] / "manifest.json")
            for method in methods:
                for rollout_id in range(rollouts):
                    relative = Path(method) / "raw" / task / f"state_{state_id:03d}" / f"rollout_{rollout_id:03d}"
                    dst = run_dir / "openwam" / relative
                    if rollout_id >= cut:
                        src = remote_copy / relative
                        if not complete_rollout(src / "result.json", method, task, state_id, rollout_id):
                            raise ValueError(f"Missing remote result: {src}")
                        if dst.exists():
                            if load_json(dst / "result.json") != load_json(src / "result.json"):
                                raise ValueError(f"Conflicting rollout result: {dst}")
                        else:
                            # The runtime contains links to the full RoboTwin assets.
                            # Keep it in remote_import; summaries need only rollout files.
                            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("runtime"), symlinks=True)
                    if not complete_rollout(dst / "result.json", method, task, state_id, rollout_id):
                        raise ValueError(f"Missing local result: {dst}")
                    if load_json(dst / "result.json").get("manifest_hash") != manifest.get("manifest_hash"):
                        raise ValueError(f"Manifest mismatch: {dst}")
    merge_rollout_results(run_dir, tasks, methods, states, rollouts)
    summarize(run_dir)


def coordinate(args: argparse.Namespace) -> None:
    rc = 1
    try:
        wait_for_workers(args)
        import_and_merge(args)
        rc = 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        write_json(args.run_dir / "coordinator_status.json", {
            "exit_code": rc, "finished_at": datetime.now(timezone.utc).isoformat(),
        })


def tmux_start(session: str, command: list[str], log_path: Path, *, host: str | None = None) -> None:
    shell_command = shlex.join(command) + " > " + shlex.quote(str(log_path)) + " 2>&1"
    tmux_command = ["tmux", "new-session", "-d", "-s", session, "bash", "-lc", shell_command]
    if host:
        checked(["ssh", "-o", "BatchMode=yes", host, shlex.join(tmux_command)])
    else:
        checked(tmux_command)


def launch(args: argparse.Namespace) -> None:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, underscores and hyphens")
    config_path = args.config.resolve()
    config = resolve_config_paths(load_json(config_path), config_path)
    validate(config)
    tasks = selected_tasks(config)
    methods = list(config["models"]["openwam"])
    states = int(config["protocol"]["states_per_task"])
    rollouts = int(config["protocol"]["rollouts_per_state"])
    local_gpus = [int(item) for item in args.local_gpus.split(",")]
    remote_gpus = [int(item) for item in args.remote_gpus.split(",")]
    cut = split_at(rollouts, len(local_gpus), len(remote_gpus))
    run_dir = ROOT / "runs" / "distributed" / run_id
    remote_run_dir = args.remote_root / "runs" / "distributed" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    exists = remote_command(args.remote_host, ["test", "-e", str(remote_run_dir)], check=False)
    if exists.returncode == 0:
        raise FileExistsError(f"Remote run directory already exists: {remote_run_dir}")
    run_dir.mkdir(parents=True)
    manifest_root = Path(config["paths"]["state_manifest_root"])
    print("[distributed] preparing manifests", flush=True)
    ensure_manifests(config, run_dir, tasks=tasks, states_per_task=states, dry_run=False, force=False)
    print("[distributed] preparing text cache", flush=True)
    prepare_text_cache(config, run_dir, tasks=tasks, methods=methods, dry_run=False)
    print("[distributed] copying code and manifests to remote", flush=True)
    checked(["rsync", "-a", "--no-owner", "--no-group", "--exclude=.git/", "--exclude=runs/",
             "--exclude=manifests/", "--exclude=__pycache__/", str(ROOT) + "/",
             f"{args.remote_host}:{args.remote_root}/"])
    remote_manifest_root = args.remote_root / "manifests" / manifest_root.name
    checked(["ssh", "-o", "BatchMode=yes", args.remote_host,
             shlex.join(["mkdir", "-p", str(remote_manifest_root), str(remote_run_dir)])])
    checked(["rsync", "-a", "--no-owner", "--no-group", str(manifest_root) + "/",
             f"{args.remote_host}:{remote_manifest_root}/"])
    manifest_hashes = {}
    for task in tasks:
        relative = Path(task) / config["protocol"]["mode"] / "manifest.json"
        local_digest = hashlib.sha256((manifest_root / relative).read_bytes()).hexdigest()
        result = remote_command(args.remote_host, ["sha256sum", str(remote_manifest_root / relative)])
        if result.stdout.split()[0] != local_digest:
            raise ValueError(f"Manifest transfer mismatch: {task}")
        manifest_hashes[task] = local_digest
    remote_config = args.remote_root / config_path.name
    result = remote_command(args.remote_host, ["/home/chw/miniconda3/bin/python", str(args.remote_root / "run_experiment1.py"),
                                               "validate", "--config", str(remote_config),
                                               "--parallel-gpus", args.remote_gpus])
    print(result.stdout, end="", flush=True)
    plan = {
        "run_id": run_id, "tasks": tasks, "methods": methods,
        "states_per_task": states, "rollouts_per_state": rollouts,
        "local_rollout_end": cut, "remote_rollout_start": cut,
        "local_gpus": local_gpus, "remote_gpus": remote_gpus,
        "mode": config["protocol"]["mode"], "local_manifest_root": str(manifest_root),
        "remote_manifest_root": str(remote_manifest_root),
        "manifest_file_hashes": manifest_hashes,
        "remote_host": args.remote_host, "remote_run_dir": str(remote_run_dir),
    }
    write_json(run_dir / "distributed_plan.json", plan)
    checked(["rsync", "-a", str(run_dir / "distributed_plan.json"),
             f"{args.remote_host}:{remote_run_dir}/distributed_plan.json"])
    stem = "exp1_" + run_id
    worker_script = ROOT / "run_experiment1_distributed.py"
    remote_script = args.remote_root / worker_script.name
    local_command = [sys.executable, str(worker_script), "worker", "--role", "local",
                     "--config", str(config_path), "--run-dir", str(run_dir),
                     "--rollout-start", "0", "--rollout-end", str(cut),
                     "--gpus", args.local_gpus]
    remote_command_args = ["/home/chw/miniconda3/bin/python", str(remote_script), "worker", "--role", "remote",
                           "--config", str(remote_config), "--run-dir", str(remote_run_dir),
                           "--rollout-start", str(cut), "--rollout-end", str(rollouts),
                           "--gpus", args.remote_gpus]
    tmux_start(stem + "_remote", remote_command_args, remote_run_dir / "worker.log", host=args.remote_host)
    tmux_start(stem + "_local", local_command, run_dir / "worker.log")
    coordinator_command = [sys.executable, str(worker_script), "coordinate",
                           "--run-dir", str(run_dir), "--remote-run-dir", str(remote_run_dir),
                           "--remote-host", args.remote_host,
                           "--local-session", stem + "_local", "--remote-session", stem + "_remote"]
    tmux_start(stem + "_merge", coordinator_command, run_dir / "merge.log")
    print(f"[distributed] launched; local={run_dir} remote={remote_run_dir}", flush=True)
    print(f"[distributed] tmux sessions: {stem}_local, {stem}_remote, {stem}_merge", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--config", type=Path, required=True)
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument("--remote-host", default=DEFAULT_REMOTE)
    launch_parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    launch_parser.add_argument("--local-gpus", default="0,1,2,3")
    launch_parser.add_argument("--remote-gpus", default="0,1,2")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--role", choices=("local", "remote"), required=True)
    worker_parser.add_argument("--config", type=Path, required=True)
    worker_parser.add_argument("--run-dir", type=Path, required=True)
    worker_parser.add_argument("--rollout-start", type=int, required=True)
    worker_parser.add_argument("--rollout-end", type=int, required=True)
    worker_parser.add_argument("--gpus", required=True)
    coordinator_parser = sub.add_parser("coordinate")
    coordinator_parser.add_argument("--run-dir", type=Path, required=True)
    coordinator_parser.add_argument("--remote-run-dir", type=Path, required=True)
    coordinator_parser.add_argument("--remote-host", required=True)
    coordinator_parser.add_argument("--local-session", required=True)
    coordinator_parser.add_argument("--remote-session", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "worker":
        worker(args)
    else:
        coordinate(args)


if __name__ == "__main__":
    main()
