#!/usr/bin/env python3
"""Run Experiment 1: fixed-state OpenWAM success probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from wm_eval.runtime import load_json, load_tasks, python_command, run_logged


ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "tasks"
RUNS_DIR = ROOT / "runs" / "experiment1"
PROBE = ROOT / "wm_eval" / "robotwin_state_probe.py"
TEXT_CACHE_PROBE = ROOT / "wm_eval" / "ensure_text_cache.py"


def resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base = config_path.parent

    def resolve(value: str) -> str:
        if value == "REPLACE_ME":
            return value
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        return str(path if path.is_absolute() else (base / path).resolve())

    for key, value in config.get("paths", {}).items():
        if value:
            config["paths"][key] = resolve(value)
    for model in config.get("models", {}).get("openwam", {}).values():
        if model.get("checkpoint_dir"):
            model["checkpoint_dir"] = resolve(model["checkpoint_dir"])
    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selected_tasks(config: dict[str, Any], task_limit: int | None = None) -> list[str]:
    tasks = load_tasks(Path(config["paths"]["task_file"]))
    return tasks[:task_limit] if task_limit else tasks


def validate(config: dict[str, Any]) -> None:
    errors: list[str] = []
    tasks = selected_tasks(config)
    seen = set(load_tasks(TASK_DIR / "seen_40.txt"))
    if not tasks:
        errors.append("paths.task_file must contain at least one task")
    duplicates = sorted({task for task in tasks if tasks.count(task) > 1})
    if duplicates:
        errors.append(f"duplicate tasks: {duplicates}")
    not_seen = sorted(set(tasks) - seen)
    if not_seen:
        errors.append(f"Experiment 1 tasks must be from seen_40.txt, got non-seen tasks: {not_seen}")

    openwam_repo = Path(config["paths"]["openwam_repo"])
    robotwin_repo = Path(config["paths"]["robotwin_repo"])
    robotwin_python = Path(config["paths"]["robotwin_python"])
    required_files = [
        openwam_repo / "scripts" / "deploy.py",
        openwam_repo / "benchmarks" / "robotwin" / "eval_policy_wrapper.py",
        openwam_repo / "benchmarks" / "robotwin" / "policy_config.yml",
        openwam_repo / "benchmarks" / "robotwin" / "openwam2robotwin_interface.py",
        robotwin_repo / "script" / "eval_policy.py",
        PROBE,
    ]
    task_config_dir = Path(config["paths"].get("robotwin_task_config_dir") or robotwin_repo / "task_config")
    assets_dir = Path(config["paths"].get("robotwin_assets_dir") or robotwin_repo / "assets")
    required_files.extend(
        [
            task_config_dir / f"{config['protocol']['mode']}.yml",
            task_config_dir / "_camera_config.yml",
            task_config_dir / "_embodiment_config.yml",
            assets_dir,
        ]
    )
    for path in required_files:
        if not path.exists():
            errors.append(f"required file does not exist: {path}")
    if not robotwin_python.exists():
        errors.append(f"paths.robotwin_python does not exist: {robotwin_python}")

    checkpoint_name = config["protocol"]["checkpoint_name"]
    for method, model in config["models"]["openwam"].items():
        ckpt_dir = Path(model["checkpoint_dir"])
        for path in [
            ckpt_dir,
            ckpt_dir / "config.yaml",
            ckpt_dir / "normalization_stats.npy",
            ckpt_dir / checkpoint_name,
        ]:
            if not path.exists():
                errors.append(f"models.openwam.{method} is missing: {path}")

    model_gpu = int(config["hardware"]["openwam_model_gpu"])
    sim_gpu = int(config["hardware"]["openwam_sim_gpu"])
    cache_gpu = int(config["hardware"].get("text_cache_gpu", 2))
    if any(gpu not in range(4) for gpu in (model_gpu, sim_gpu, cache_gpu)):
        errors.append(
            f"Experiment 1 smoke should use only GPU 0-3, got model={model_gpu}, sim={sim_gpu}, cache={cache_gpu}"
        )
    parallel_gpus = config["hardware"].get("parallel_gpus", [])
    if parallel_gpus and (not isinstance(parallel_gpus, list) or any(int(gpu) not in range(4) for gpu in parallel_gpus)):
        errors.append(f"hardware.parallel_gpus must use only GPU 0-3, got {parallel_gpus}")
    if int(config["hardware"].get("workers_per_gpu", 1)) < 1:
        errors.append("hardware.workers_per_gpu must be positive")
    if config["hardware"].get("persistent_server"):
        if not parallel_gpus or not config["hardware"].get("parallel_by_rollout"):
            errors.append("persistent_server requires parallel_gpus and parallel_by_rollout")
        if int(config["hardware"].get("workers_per_gpu", 1)) != 1:
            errors.append("persistent_server requires workers_per_gpu=1 to isolate RESET sessions")
        if len(set(parallel_gpus)) != len(parallel_gpus):
            errors.append("persistent_server requires distinct parallel_gpus")
        if int(config["hardware"].get("persistent_servers_per_gpu", 1)) < 1:
            errors.append("persistent_servers_per_gpu must be positive")
        clients = int(config["hardware"].get("persistent_clients_per_gpu", 1))
        if clients < 1:
            errors.append("persistent_clients_per_gpu must be positive")
        if clients > 1 and not config["hardware"].get("shared_server_sessions", False):
            errors.append("multiple persistent clients require shared_server_sessions=true")
        if clients > 1 and int(config["hardware"].get("persistent_servers_per_gpu", 1)) != 1:
            errors.append("shared clients require one persistent model server per GPU")

    if errors:
        raise ValueError("Invalid Experiment 1 configuration:\n- " + "\n- ".join(errors))


def git_revision(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    return {
        "path": str(repo),
        "commit": git("rev-parse", "HEAD") or None,
        "modified_tracked_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
    }


def make_env(config: dict[str, Any], *, sim_gpu: bool) -> dict[str, str]:
    env = os.environ.copy()
    openwam_repo = Path(config["paths"]["openwam_repo"])
    robotwin_repo = Path(config["paths"]["robotwin_repo"])
    env["PYTHONUNBUFFERED"] = "1"
    env["ROBOTWIN_PATH"] = str(robotwin_repo)
    env.setdefault("ROBOTWIN_ENABLE_PLANNER_FALLBACK", "1")
    env.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))
    if config["paths"].get("robotwin_task_config_dir"):
        env["ROBOTWIN_TASK_CONFIG_DIR"] = str(config["paths"]["robotwin_task_config_dir"])
    if config["paths"].get("robotwin_assets_dir"):
        env["ROBOTWIN_ASSETS_DIR"] = str(config["paths"]["robotwin_assets_dir"])
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(openwam_repo),
            str(openwam_repo / "benchmarks" / "robotwin"),
            str(openwam_repo / "benchmarks"),
            str(robotwin_repo),
            env.get("PYTHONPATH", ""),
        ]
    )
    if sim_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(config["hardware"]["openwam_sim_gpu"])
        robotwin_bin = str(Path(config["paths"]["robotwin_python"]).parent)
        env["PATH"] = robotwin_bin + os.pathsep + env.get("PATH", "")
        egl_json = (
            Path(config["paths"]["robotwin_python"]).parent.parent
            / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
        )
        if egl_json.is_file() and not env.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl_json)
    return env


def probe_command(
    config: dict[str, Any],
    *,
    operation: str,
    task: str,
    result_file: Path,
    progress_file: Path,
    method: str | None = None,
    checkpoint_dir: str | None = None,
    state_manifest: Path | None = None,
    state_id: int | None = None,
    num_states: int | None = None,
    num_rollouts: int | None = None,
    rollout_id_start: int | None = None,
    max_action_steps: int | None = None,
) -> list[str]:
    openwam_repo = Path(config["paths"]["openwam_repo"])
    protocol = config["protocol"]
    command = [
        config["paths"]["robotwin_python"],
        str(PROBE),
        "--operation",
        operation,
        "--openwam-wrapper",
        str(openwam_repo / "benchmarks" / "robotwin" / "eval_policy_wrapper.py"),
        "--policy-config",
        str(openwam_repo / "benchmarks" / "robotwin" / "policy_config.yml"),
        "--task",
        task,
        "--mode",
        str(protocol["mode"]),
        "--host",
        str(config["hardware"]["openwam_host"]),
        "--port",
        str(config["hardware"]["openwam_port"]),
        "--seed",
        str(protocol["seed"]),
        "--instruction-type",
        str(protocol["instruction_type"]),
        "--ckpt-setting",
        method or "openwam",
        "--result-file",
        str(result_file),
        "--progress-file",
        str(progress_file),
        "--skip-get-obs-within-replan",
        str(bool(protocol.get("skip_get_obs_within_replan", False))).lower(),
    ]
    if num_states is not None:
        command.extend(["--num-states", str(num_states)])
        command.extend(
            [
                "--max-manifest-attempts",
                str(config["protocol"].get("max_manifest_attempts", max(1000, num_states * 50))),
                "--max-consecutive-infra-errors",
                str(config["protocol"].get("max_consecutive_infra_errors", 3)),
            ]
        )
    if method is not None:
        command.extend(["--method", method])
    if checkpoint_dir is not None:
        command.extend(["--checkpoint-dir", checkpoint_dir])
    if state_manifest is not None:
        command.extend(["--state-manifest", str(state_manifest)])
    if state_id is not None:
        command.extend(["--state-id", str(state_id)])
    if num_rollouts is not None:
        command.extend(["--num-rollouts", str(num_rollouts)])
    if rollout_id_start is not None:
        command.extend(["--rollout-id-start", str(rollout_id_start)])
    if max_action_steps is not None:
        command.extend(["--max-action-steps", str(max_action_steps)])
    return command


def ensure_manifests(
    config: dict[str, Any],
    run_root: Path,
    *,
    tasks: list[str],
    states_per_task: int,
    dry_run: bool,
    force: bool,
) -> Path:
    manifest_root = Path(config["paths"]["state_manifest_root"]).resolve()
    env = make_env(config, sim_gpu=True)
    openwam_repo = Path(config["paths"]["openwam_repo"])

    for task in tasks:
        task_dir = manifest_root / task / config["protocol"]["mode"]
        manifest_path = task_dir / "manifest.json"
        if not force and manifest_path.is_file():
            manifest = load_json(manifest_path)
            if (
                manifest.get("task") == task
                and manifest.get("mode") == config["protocol"]["mode"]
                and len(manifest.get("states", [])) >= states_per_task
            ):
                print(f"[experiment1] reuse state manifest: {task}", flush=True)
                continue

        command_env = env.copy()
        command_env["ROBOTWIN_RUNTIME_ROOT"] = str(task_dir / "runtime")
        command = probe_command(
            config,
            operation="build_state_manifest",
            task=task,
            result_file=manifest_path,
            progress_file=task_dir / "progress.json",
            num_states=states_per_task,
        )
        run_logged(
            command,
            cwd=openwam_repo,
            env=command_env,
            log_path=run_root / "manifests" / f"{task}.log",
            dry_run=dry_run,
            quiet=True,
        )
    return manifest_root


def wait_for_port(host: str, port: int, process: subprocess.Popen[str], timeout: int = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OpenWAM server exited early with code {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"OpenWAM server did not listen on {host}:{port} within {timeout}s")


def require_free_port(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=2):
            raise RuntimeError(f"OpenWAM port {host}:{port} is already in use; refusing to connect to an unknown model")
    except OSError:
        pass


def choose_available_port(host: str, base_port: int, job_index: int) -> int:
    for offset in range(20):
        candidate = base_port + job_index + 1 + offset * 1000
        if candidate > 65535:
            break
        try:
            with socket.socket() as probe:
                probe.bind((host, candidate))
            return candidate
        except OSError:
            continue
    raise RuntimeError(f"No free port found for job {job_index} starting from {base_port}")


def start_server(
    config: dict[str, Any], method: str, model: dict[str, Any], run_root: Path,
    dry_run: bool, server_dir: Path | None = None,
):
    openwam_repo = Path(config["paths"]["openwam_repo"])
    server_dir = server_dir or run_root / "openwam" / method
    server_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_dir / "server.log"
    command = [
        *python_command(config, "openwam"),
        "scripts/deploy.py",
        "--ckpt-dir",
        model["checkpoint_dir"],
        "--ckpt-name",
        config["protocol"]["checkpoint_name"],
        "--device",
        "cuda:0",
        "--host",
        str(config["hardware"]["openwam_host"]),
        "--port",
        str(config["hardware"]["openwam_port"]),
        *model.get("deploy_args", []),
    ]
    if config["hardware"].get("shared_server_sessions", False):
        command.append("--session-isolation")
    if dry_run:
        log_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        return None, None

    require_free_port(str(config["hardware"]["openwam_host"]), int(config["hardware"]["openwam_port"]))
    env = make_env(config, sim_gpu=False)
    env["CUDA_VISIBLE_DEVICES"] = str(config["hardware"]["openwam_model_gpu"])
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(openwam_repo),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_port(str(config["hardware"]["openwam_host"]), int(config["hardware"]["openwam_port"]), process)
    except Exception:
        stop_server(process, log)
        raise
    return process, log


def stop_server(process, log) -> None:
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if log is not None:
        log.close()


def prepare_text_cache(
    config: dict[str, Any], run_root: Path, *, tasks: list[str], methods: list[str], dry_run: bool,
) -> None:
    if dry_run:
        return
    manifest_root = Path(config["paths"]["state_manifest_root"]).resolve()
    cache_env = make_env(config, sim_gpu=False)
    cache_env["CUDA_VISIBLE_DEVICES"] = str(config["hardware"].get("text_cache_gpu", 2))
    cache_command = [
        *python_command(config, "openwam"),
        str(TEXT_CACHE_PROBE),
        *[part for method in methods for part in ("--checkpoint-dir", config["models"]["openwam"][method]["checkpoint_dir"])],
        *[part for task in tasks for part in ("--manifest", str(manifest_root / task / config["protocol"]["mode"] / "manifest.json"))],
    ]
    run_logged(
        cache_command,
        cwd=Path(config["paths"]["openwam_repo"]),
        env=cache_env,
        log_path=run_root / "text_cache.log",
        dry_run=False,
        quiet=True,
    )


def run_methods(
    config: dict[str, Any],
    run_root: Path,
    *,
    tasks: list[str],
    methods: list[str],
    states_per_task: int,
    rollouts_per_state: int,
    max_action_steps: int | None,
    dry_run: bool,
) -> None:
    manifest_root = Path(config["paths"]["state_manifest_root"]).resolve()
    openwam_repo = Path(config["paths"]["openwam_repo"])
    client_env = make_env(config, sim_gpu=True)

    for method in methods:
        model = config["models"]["openwam"][method]
        process, log = start_server(config, method, model, run_root, dry_run)
        try:
            for task in tasks:
                manifest_path = manifest_root / task / config["protocol"]["mode"] / "manifest.json"
                if dry_run:
                    state_count = states_per_task
                else:
                    state_count = min(states_per_task, len(load_json(manifest_path)["states"]))
                for state_id in range(state_count):
                    raw_dir = run_root / "openwam" / method / "raw" / task / f"state_{state_id:03d}"
                    command_env = client_env.copy()
                    command_env["ROBOTWIN_RUNTIME_ROOT"] = str(raw_dir / "runtime")
                    command = probe_command(
                        config,
                        operation="run_state_rollouts",
                        task=task,
                        result_file=raw_dir / "result.json",
                        progress_file=raw_dir / "progress.json",
                        method=method,
                        checkpoint_dir=model["checkpoint_dir"],
                        state_manifest=manifest_path,
                        state_id=state_id,
                        num_rollouts=rollouts_per_state,
                        max_action_steps=max_action_steps,
                    )
                    run_logged(
                        command,
                        cwd=openwam_repo,
                        env=command_env,
                        log_path=raw_dir / "driver.log",
                        dry_run=dry_run,
                        quiet=True,
                    )
        finally:
            stop_server(process, log)


def run_parallel_methods(
    config: dict[str, Any],
    run_root: Path,
    *,
    tasks: list[str],
    methods: list[str],
    states_per_task: int,
    rollouts_per_state: int,
    max_action_steps: int | None,
    dry_run: bool,
    only_jobs: set[tuple[str, str, int, int | None]] | None = None,
) -> None:
    if config["hardware"].get("persistent_server"):
        run_persistent_methods(
            config, run_root, tasks=tasks, methods=methods,
            states_per_task=states_per_task, rollouts_per_state=rollouts_per_state,
            max_action_steps=max_action_steps, dry_run=dry_run, only_jobs=only_jobs,
        )
        return
    gpus = [int(gpu) for gpu in config["hardware"]["parallel_gpus"]]
    workers_per_gpu = int(config["hardware"].get("workers_per_gpu", 2))
    semaphores = {gpu: threading.Semaphore(workers_per_gpu) for gpu in gpus}
    manifest_root = Path(config["paths"]["state_manifest_root"]).resolve()
    openwam_repo = Path(config["paths"]["openwam_repo"])
    split_rollouts = bool(config["hardware"].get("parallel_by_rollout", False))
    all_jobs = [
        (method, task, state_id, rollout_id)
        for task in tasks
        for state_id in range(states_per_task)
        for method in methods
        for rollout_id in (range(rollouts_per_state) if split_rollouts else (None,))
    ]

    def complete_result(method: str, task: str, state_id: int, rollout_id: int | None) -> bool:
        raw_dir = run_root / "openwam" / method / "raw" / task / f"state_{state_id:03d}"
        worker_dir = raw_dir / f"rollout_{rollout_id:03d}" if rollout_id is not None else raw_dir
        result_path = worker_dir / "result.json"
        if not result_path.is_file():
            return False
        result = load_json(result_path)
        expected_n = 1 if rollout_id is not None else rollouts_per_state
        return (
            result.get("task") == task
            and result.get("method") == method
            and result.get("state_id") == state_id
            and result.get("num_rollouts") == expected_n
            and result.get("valid_rollouts") == expected_n
            and result.get("infra_failures") == 0
            and (rollout_id is None or result["rollouts"][0]["rollout_id"] == rollout_id)
        )

    jobs = [
        (index, method, task, state_id, rollout_id)
        for index, (method, task, state_id, rollout_id) in enumerate(all_jobs)
        if only_jobs is None or (method, task, state_id, rollout_id) in only_jobs
        if not complete_result(method, task, state_id, rollout_id)
    ]
    print(f"[experiment1] pending parallel jobs: {len(jobs)}/{len(all_jobs)}", flush=True)

    def run_job(index: int, method: str, task: str, state_id: int, rollout_id: int | None) -> None:
        gpu = gpus[index % len(gpus)]
        port = choose_available_port(
            str(config["hardware"]["openwam_host"]),
            int(config["hardware"]["openwam_port"]),
            index,
        )
        worker_config = deepcopy(config)
        worker_config["hardware"].update(
            openwam_model_gpu=gpu, openwam_sim_gpu=gpu, openwam_port=port
        )
        model = worker_config["models"]["openwam"][method]
        raw_dir = run_root / "openwam" / method / "raw" / task / f"state_{state_id:03d}"
        worker_dir = raw_dir / f"rollout_{rollout_id:03d}" if rollout_id is not None else raw_dir
        manifest_path = manifest_root / task / config["protocol"]["mode"] / "manifest.json"
        label = f"{task}/{method}/state_{state_id:03d}"
        if rollout_id is not None:
            label += f"/rollout_{rollout_id:03d}"
        with semaphores[gpu]:
            print(f"[experiment1] start {label} gpu={gpu} port={port}", flush=True)
            process, log = start_server(worker_config, method, model, run_root, dry_run, server_dir=worker_dir)
            try:
                client_env = make_env(worker_config, sim_gpu=True)
                client_env["ROBOTWIN_RUNTIME_ROOT"] = str(worker_dir / "runtime")
                command = probe_command(
                    worker_config,
                    operation="run_state_rollouts",
                    task=task,
                    result_file=worker_dir / "result.json",
                    progress_file=worker_dir / "progress.json",
                    method=method,
                    checkpoint_dir=model["checkpoint_dir"],
                    state_manifest=manifest_path,
                    state_id=state_id,
                    num_rollouts=1 if rollout_id is not None else rollouts_per_state,
                    rollout_id_start=rollout_id,
                    max_action_steps=max_action_steps,
                )
                run_logged(
                    command,
                    cwd=openwam_repo,
                    env=client_env,
                    log_path=worker_dir / "driver.log",
                    dry_run=dry_run,
                    quiet=True,
                )
            finally:
                stop_server(process, log)
        print(f"[experiment1] done {label}", flush=True)

    errors = []
    with ExitStack() as stack:
        executors = {
            gpu: stack.enter_context(ThreadPoolExecutor(max_workers=workers_per_gpu))
            for gpu in gpus
        }
        futures = {
            executors[gpus[index % len(gpus)]].submit(
                run_job, index, method, task, state_id, rollout_id
            ): (method, task, state_id, rollout_id)
            for index, method, task, state_id, rollout_id in jobs
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append((futures[future], exc))
                print(f"[experiment1] failed {futures[future]}: {exc}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} parallel evaluation job(s) failed; inspect run logs")
    if only_jobs is not None:
        return
    if split_rollouts and not dry_run:
        merge_rollout_results(run_root, tasks, methods, states_per_task, rollouts_per_state)


def complete_rollout(path: Path, method: str, task: str, state_id: int, rollout_id: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = load_json(path)
        return (
            result.get("task") == task
            and result.get("method") == method
            and result.get("state_id") == state_id
            and result.get("num_rollouts") == 1
            and result.get("valid_rollouts") == 1
            and result.get("infra_failures") == 0
            and len(result.get("rollouts", [])) == 1
            and result["rollouts"][0]["rollout_id"] == rollout_id
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def merge_rollout_results(
    run_root: Path, tasks: list[str], methods: list[str], states_per_task: int, rollouts_per_state: int,
) -> None:
    identity_keys = ("task", "mode", "state_id", "accepted_seed", "instruction", "method", "checkpoint_dir", "manifest_hash")
    for task in tasks:
        for state_id in range(states_per_task):
            for method in methods:
                raw_dir = run_root / "openwam" / method / "raw" / task / f"state_{state_id:03d}"
                paths = [raw_dir / f"rollout_{i:03d}" / "result.json" for i in range(rollouts_per_state)]
                if not all(complete_rollout(path, method, task, state_id, i) for i, path in enumerate(paths)):
                    raise ValueError(f"Missing or failed rollout fragment: {raw_dir}")
                fragments = [load_json(path) for path in paths]
                if any(any(item[key] != fragments[0][key] for key in identity_keys) for item in fragments[1:]):
                    raise ValueError(f"Rollout fragments do not share a fixed state: {raw_dir}")
                rollouts = [fragment["rollouts"][0] for fragment in fragments]
                if len({item["policy_seed"] for item in rollouts}) != rollouts_per_state:
                    raise ValueError(f"Duplicate policy seeds: {raw_dir}")
                payload = {key: fragments[0][key] for key in identity_keys}
                payload.update(
                    num_rollouts=rollouts_per_state,
                    successes=sum(bool(item["success"]) for item in rollouts),
                    valid_rollouts=rollouts_per_state,
                    infra_failures=0,
                    rollouts=rollouts,
                )
                write_json(raw_dir / "result.json", payload)


def run_persistent_methods(
    config: dict[str, Any], run_root: Path, *, tasks: list[str], methods: list[str],
    states_per_task: int, rollouts_per_state: int, max_action_steps: int | None,
    dry_run: bool, only_jobs: set[tuple[str, str, int, int | None]] | None = None,
) -> None:
    """Keep one server per GPU and method; give it exclusive rollout sessions."""
    gpus = [int(gpu) for gpu in config["hardware"]["parallel_gpus"]]
    servers_per_gpu = int(config["hardware"].get("persistent_servers_per_gpu", 1))
    clients_per_gpu = int(config["hardware"].get("persistent_clients_per_gpu", 1))
    slots = [(gpu, slot) for gpu in gpus for slot in range(servers_per_gpu)]
    manifest_root = Path(config["paths"]["state_manifest_root"])
    openwam_repo = Path(config["paths"]["openwam_repo"])
    host = str(config["hardware"]["openwam_host"])
    base_port = int(config["hardware"]["openwam_port"])
    failures: list[tuple[str, str, int, int, str]] = []
    # Phase by method so a GPU loads each checkpoint at most once per invocation.
    for method in methods:
        jobs = [
            (task, state_id, rollout_id)
            for task in tasks
            for state_id in range(states_per_task)
            for rollout_id in range(rollouts_per_state)
            if only_jobs is None or (method, task, state_id, rollout_id) in only_jobs
        ]
        pending = [
            (index, job) for index, job in enumerate(jobs)
            if not complete_rollout(
                run_root / "openwam" / method / "raw" / job[0]
                / f"state_{job[1]:03d}" / f"rollout_{job[2]:03d}" / "result.json",
                method, *job,
            )
        ]
        print(f"[experiment1] {method}: pending rollouts {len(pending)}/{len(jobs)}", flush=True)
        if not pending:
            continue
        assignments = {slot: [] for slot in slots}
        for index, job in pending:
            assignments[slots[index % len(slots)]].append(job)

        def gpu_worker(gpu: int, slot: int, gpu_jobs: list[tuple[str, int, int]]) -> list[tuple[str, str, int, int, str]]:
            worker_config = deepcopy(config)
            port = choose_available_port(host, base_port, gpu * servers_per_gpu + slot)
            worker_config["hardware"].update(
                openwam_model_gpu=gpu, openwam_sim_gpu=gpu, openwam_port=port,
            )
            model = worker_config["models"]["openwam"][method]
            worker_name = f"gpu_{gpu}" if servers_per_gpu == 1 else f"gpu_{gpu}_slot_{slot}"
            worker_dir = run_root / "openwam" / method / "workers" / worker_name
            worker_dir.mkdir(parents=True, exist_ok=True)
            process = log = None
            restarts = 0
            worker_errors = []
            started = time.monotonic()

            def run_rollout(task: str, state_id: int, rollout_id: int) -> tuple[str, str, int, int, str] | None:
                raw_dir = run_root / "openwam" / method / "raw" / task / f"state_{state_id:03d}"
                rollout_dir = raw_dir / f"rollout_{rollout_id:03d}"
                client_env = make_env(worker_config, sim_gpu=True)
                client_env["ROBOTWIN_RUNTIME_ROOT"] = str(rollout_dir / "runtime")
                command = probe_command(
                    worker_config, operation="run_state_rollouts", task=task,
                    result_file=rollout_dir / "result.json",
                    progress_file=rollout_dir / "progress.json", method=method,
                    checkpoint_dir=model["checkpoint_dir"],
                    state_manifest=manifest_root / task / config["protocol"]["mode"] / "manifest.json",
                    state_id=state_id, num_rollouts=1, rollout_id_start=rollout_id,
                    max_action_steps=max_action_steps,
                )
                job_start = time.monotonic()
                try:
                    run_logged(command, cwd=openwam_repo, env=client_env,
                               log_path=rollout_dir / "driver.log", dry_run=dry_run, quiet=True)
                    if not dry_run and not complete_rollout(
                        rollout_dir / "result.json", method, task, state_id, rollout_id,
                    ):
                        raise RuntimeError("driver exited without a valid rollout result")
                    return None
                except Exception as exc:
                    return (method, task, state_id, rollout_id, str(exc))
                finally:
                    write_json(rollout_dir / "timing.json", {
                        "gpu": gpu, "port": port, "driver_wall_sec": round(time.monotonic() - job_start, 3),
                    })

            try:
                for job_index, (task, state_id, rollout_id) in enumerate(gpu_jobs):
                    if process is None or (process is not None and process.poll() is not None):
                        if process is not None:
                            stop_server(process, log)
                            restarts += 1
                        ready_start = time.monotonic()
                        try:
                            server_dir = worker_dir if restarts == 0 else worker_dir / f"restart_{restarts:03d}"
                            process, log = start_server(
                                worker_config, method, model, run_root, dry_run, server_dir=server_dir,
                            )
                            write_json(server_dir / "timing.json", {
                                "gpu": gpu, "slot": slot, "method": method,
                                "ready_sec": round(time.monotonic() - ready_start, 3),
                                "restart": restarts,
                            })
                        except Exception as exc:
                            worker_errors.append((method, task, state_id, rollout_id, str(exc)))
                            process = log = None
                            continue
                    if clients_per_gpu == 1:
                        failure = run_rollout(task, state_id, rollout_id)
                        if failure:
                            worker_errors.append(failure)
                    else:
                        # The model is loaded once. Each driver opens its own WebSocket,
                        # whose action buffer and RESET state live in a separate session.
                        remaining = gpu_jobs[job_index:]
                        with ThreadPoolExecutor(max_workers=clients_per_gpu) as clients:
                            futures = {clients.submit(run_rollout, *job): job for job in remaining}
                            for future in as_completed(futures):
                                failure = future.result()
                                if failure:
                                    worker_errors.append(failure)
                        break
            finally:
                stop_server(process, log)
                write_json(worker_dir / "worker_timing.json", {
                    "gpu": gpu, "slot": slot, "method": method,
                    "jobs": len(gpu_jobs), "clients_per_gpu": clients_per_gpu,
                    "restarts": restarts,
                    "wall_sec": round(time.monotonic() - started, 3),
                })
            return worker_errors

        with ThreadPoolExecutor(max_workers=len(slots)) as executor:
            futures = {executor.submit(gpu_worker, gpu, slot, assigned): (gpu, slot)
                       for (gpu, slot), assigned in assignments.items() if assigned}
            for future in as_completed(futures):
                failures.extend(future.result())
    if failures:
        for failure in failures:
            print(f"[experiment1] failed {failure}", flush=True)
        raise RuntimeError(f"{len(failures)} rollout(s) failed; resume to retry only missing results")
    if only_jobs is None and not dry_run:
        merge_rollout_results(run_root, tasks, methods, states_per_task, rollouts_per_state)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_bootstrap_ci(values: list[float], *, seed: int = 0, samples: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        raise ValueError("Cannot bootstrap an empty paired sample")
    if len(values) == 1:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


CATEGORIES = (
    "newly_solved", "amplified_existing", "unchanged_easy",
    "unchanged_hard", "regressed", "same_other",
)


def state_category(no_wm_successes: int, wm_successes: int, n: int) -> str:
    no_wm = no_wm_successes / n if n else 0.0
    wm = wm_successes / n if n else 0.0
    if no_wm_successes == 0 and wm_successes > 0:
        return "newly_solved"
    if no_wm_successes > 0 and wm_successes > no_wm_successes:
        return "amplified_existing"
    if no_wm >= 0.875 and wm >= 0.875:
        return "unchanged_easy"
    if no_wm_successes == 0 and wm_successes == 0:
        return "unchanged_hard"
    if wm_successes < no_wm_successes:
        return "regressed"
    return "same_other"


def summarize(run_root: Path) -> None:
    result_paths = sorted(run_root.glob("openwam/*/raw/*/state_*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No Experiment 1 result.json files found under {run_root}")

    metadata_path = run_root / "metadata.json"
    action_capped = metadata_path.is_file() and load_json(metadata_path).get("max_action_steps") is not None
    if action_capped or any(
        rollout.get("truncated")
        for path in result_paths
        for rollout in load_json(path).get("rollouts", [])
    ):
        summarize_smoke(run_root, result_paths)
        return

    per_method_state: dict[tuple[str, int, str], dict[str, Any]] = {}
    per_rollout: list[dict[str, Any]] = []
    for path in result_paths:
        result = load_json(path)
        method = result["method"]
        task = result["task"]
        state_id = int(result["state_id"])
        n = int(result["num_rollouts"])
        successes = int(result["successes"])
        p = successes / n if n else 0.0
        ci_low, ci_high = wilson_interval(successes, n)
        row = {
            "task": task,
            "state_id": state_id,
            "method": method,
            "manifest_hash": result.get("manifest_hash"),
            "accepted_seed": result["accepted_seed"],
            "instruction": result["instruction"],
            "successes": successes,
            "num_rollouts": n,
            "p_success": p,
            "binomial_se": math.sqrt(p * (1 - p) / n) if n else 0.0,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "valid_rollouts": result.get("valid_rollouts", n),
            "infra_failures": result.get("infra_failures", 0),
        }
        per_method_state[(task, state_id, method)] = row
        for rollout in result.get("rollouts", []):
            per_rollout.append({"task": task, "state_id": state_id, "method": method, **rollout})

    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_state_method_rows = sorted(per_method_state.values(), key=lambda r: (r["task"], r["state_id"], r["method"]))
    with (summary_dir / "per_state_by_method.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_state_method_rows[0]))
        writer.writeheader()
        writer.writerows(per_state_method_rows)

    joined: list[dict[str, Any]] = []
    tasks = sorted({row["task"] for row in per_state_method_rows})
    for task in tasks:
        state_ids = sorted({row["state_id"] for row in per_state_method_rows if row["task"] == task})
        for state_id in state_ids:
            no_wm = per_method_state.get((task, state_id, "no_wm"))
            wm = per_method_state.get((task, state_id, "wm"))
            if no_wm is None or wm is None:
                continue
            if any(
                no_wm[key] != wm[key]
                for key in ("manifest_hash", "accepted_seed", "instruction", "num_rollouts")
            ):
                raise ValueError(f"Unpaired fixed-state records for {task} state {state_id}")
            if no_wm["infra_failures"] or wm["infra_failures"]:
                raise ValueError(f"Infrastructure failure in {task} state {state_id}; success rates invalid")
            n = int(no_wm["num_rollouts"])
            joined.append(
                {
                    "task": task,
                    "state_id": state_id,
                    "accepted_seed": no_wm["accepted_seed"],
                    "p_success_no_wm": no_wm["p_success"],
                    "p_success_wm": wm["p_success"],
                    "delta_p": wm["p_success"] - no_wm["p_success"],
                    "relative_delta": (wm["p_success"] - no_wm["p_success"]) / max(no_wm["p_success"], 1 / n),
                    "no_wm_successes": no_wm["successes"],
                    "wm_successes": wm["successes"],
                    "num_rollouts": n,
                    "state_category": state_category(int(no_wm["successes"]), int(wm["successes"]), n),
                }
            )

    if joined:
        with (summary_dir / "per_state.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(joined[0]))
            writer.writeheader()
            writer.writerows(joined)

    if per_rollout:
        fieldnames = sorted({key for row in per_rollout for key in row})
        with (summary_dir / "per_rollout.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_rollout)

    per_task: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for task in tasks:
        rows = [row for row in joined if row["task"] == task]
        if not rows:
            continue
        delta_ci_low, delta_ci_high = paired_bootstrap_ci([row["delta_p"] for row in rows])
        per_task.append(
            {
                "task": task,
                "states": len(rows),
                "mean_p_no_wm": sum(row["p_success_no_wm"] for row in rows) / len(rows),
                "mean_p_wm": sum(row["p_success_wm"] for row in rows) / len(rows),
                "mean_delta_p": sum(row["delta_p"] for row in rows) / len(rows),
                "delta_ci95_low": delta_ci_low,
                "delta_ci95_high": delta_ci_high,
                **{category: sum(row["state_category"] == category for row in rows) for category in CATEGORIES},
            }
        )
        category_rows.extend(
            {
                "task": task,
                "category": category,
                "count": count,
                "fraction": count / len(rows),
            }
            for category in CATEGORIES
            for count in [sum(row["state_category"] == category for row in rows)]
        )
    if per_task:
        with (summary_dir / "per_task.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_task[0]))
            writer.writeheader()
            writer.writerows(per_task)
        with (summary_dir / "state_categories.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(category_rows[0]))
            writer.writeheader()
            writer.writerows(category_rows)

    lines = ["# Experiment 1 Summary", ""]
    if joined:
        mean_no_wm = sum(row["p_success_no_wm"] for row in joined) / len(joined)
        mean_wm = sum(row["p_success_wm"] for row in joined) / len(joined)
        deltas = [row["delta_p"] for row in joined]
        global_ci_low, global_ci_high = paired_bootstrap_ci(deltas)
        lines.extend(
            [
                f"States: {len(joined)}",
                f"No-WM mean success probability: {100 * mean_no_wm:.2f}%",
                f"WM mean success probability: {100 * mean_wm:.2f}%",
                f"Mean delta: {100 * (mean_wm - mean_no_wm):+.2f} pp",
                f"Median delta: {100 * median(deltas):+.2f} pp",
                (
                    f"Paired bootstrap 95% CI for mean delta: [{100 * global_ci_low:+.2f}, {100 * global_ci_high:+.2f}] pp"
                    if global_ci_low is not None and global_ci_high is not None
                    else "Paired bootstrap 95% CI: insufficient states (n=1)"
                ),
                f"Successful rollouts (no_wm/wm): {sum(row['no_wm_successes'] for row in joined)}/{sum(row['wm_successes'] for row in joined)}",
                "",
                "| Category | Count |",
                "|---|---:|",
            ]
        )
        for category in sorted({row["state_category"] for row in joined}):
            lines.append(f"| {category} | {sum(1 for row in joined if row['state_category'] == category)} |")
    else:
        lines.append("No paired states found. Did both methods finish?")
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[experiment1] summary: {summary_dir / 'summary.md'}", flush=True)


def summarize_smoke(run_root: Path, result_paths: list[Path]) -> None:
    rows = []
    for path in result_paths:
        result = load_json(path)
        for rollout in result.get("rollouts", []):
            rows.append(
                {
                    "task": result["task"],
                    "method": result["method"],
                    "state_id": result["state_id"],
                    "rollout_id": rollout["rollout_id"],
                    "terminal_step": rollout["terminal_step"],
                    "success": rollout["success"],
                    "truncated": rollout.get("truncated", False),
                    "error": rollout["error"],
                }
            )
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "smoke_rollouts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tasks = {row["task"] for row in rows}
    methods = {row["method"] for row in rows}
    failures = [row for row in rows if row["error"]]
    lines = [
        "# Experiment 1 Smoke Summary",
        "",
        "Action steps were capped. Truncated rollouts do not estimate task success probability.",
        "",
        f"Tasks: {len(tasks)}",
        f"Methods: {', '.join(sorted(methods))}",
        f"Rollouts: {len(rows)}",
        f"Infrastructure failures: {len(failures)}",
        f"Truncated: {sum(bool(row['truncated']) for row in rows)}",
    ]
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[experiment1] smoke summary: {summary_dir / 'summary.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "build-manifests", "run", "summarize"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.experiment1.smoke.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--method", choices=("all", "no_wm", "wm"), default="all")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--rollouts-per-state", type=int)
    parser.add_argument("--max-action-steps", type=int)
    parser.add_argument("--rollout-start", type=int)
    parser.add_argument("--rollout-end", type=int)
    parser.add_argument("--skip-text-cache", action="store_true")
    parser.add_argument("--parallel-gpus", help="Comma-separated GPU indices for this host")
    parser.add_argument("--persistent-servers-per-gpu", type=int)
    parser.add_argument("--persistent-clients-per-gpu", type=int)
    parser.add_argument("--force-manifests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_action_steps is not None and args.max_action_steps <= 0:
        parser.error("--max-action-steps must be positive")

    if args.command == "summarize":
        if args.run_dir is None:
            parser.error("summarize requires --run-dir")
        summarize(args.run_dir.resolve())
        return

    config_path = args.config.resolve()
    config = resolve_config_paths(load_json(config_path), config_path)
    if args.parallel_gpus:
        try:
            config["hardware"]["parallel_gpus"] = [int(item) for item in args.parallel_gpus.split(",")]
        except ValueError:
            parser.error("--parallel-gpus must be comma-separated integers")
    if args.persistent_servers_per_gpu is not None:
        config["hardware"]["persistent_servers_per_gpu"] = args.persistent_servers_per_gpu
    if args.persistent_clients_per_gpu is not None:
        config["hardware"]["persistent_clients_per_gpu"] = args.persistent_clients_per_gpu
    validate(config)
    print("[experiment1] configuration is valid")
    if args.command == "validate":
        return

    run_root = args.run_dir.resolve() if args.run_dir else RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    tasks = selected_tasks(config, args.task_limit)
    states_per_task = args.state_limit or int(config["protocol"]["states_per_task"])
    rollouts_per_state = args.rollouts_per_state or int(config["protocol"]["rollouts_per_state"])
    partial = args.rollout_start is not None or args.rollout_end is not None
    if partial:
        start = 0 if args.rollout_start is None else args.rollout_start
        end = rollouts_per_state if args.rollout_end is None else args.rollout_end
        if not (0 <= start < end <= rollouts_per_state):
            parser.error("rollout range must satisfy 0 <= start < end <= rollouts_per_state")
        if not config["hardware"].get("persistent_server"):
            parser.error("rollout ranges require persistent_server=true")
    config["paths"]["state_manifest_root"] = str(Path(config["paths"]["state_manifest_root"]).resolve())
    metadata = {
            "created_at": datetime.now().astimezone().isoformat(),
            "label": config["label"],
            "smoke_only": config.get("smoke_only", False),
            "tasks": tasks,
            "states_per_task": states_per_task,
            "rollouts_per_state": rollouts_per_state,
            "max_action_steps": args.max_action_steps,
            "repositories": [
                git_revision(Path(config["paths"]["openwam_repo"])),
                git_revision(Path(config["paths"]["robotwin_repo"])),
            ],
        }
    if args.run_dir and (run_root / "metadata.json").is_file():
        previous = load_json(run_root / "metadata.json")
        if previous.get("valid_for_analysis") is False:
            raise ValueError(f"Cannot resume invalidated run: {run_root}")
        if load_json(run_root / "config.json") != config or any(
            previous.get(key) != metadata[key]
            for key in ("tasks", "states_per_task", "rollouts_per_state", "max_action_steps")
        ):
            raise ValueError(f"Resume configuration differs from original run: {run_root}")
        print(f"[experiment1] resume run directory: {run_root}", flush=True)
    else:
        write_json(run_root / "config.json", config)
        write_json(run_root / "metadata.json", metadata)

    ensure_manifests(
        config,
        run_root,
        tasks=tasks,
        states_per_task=states_per_task,
        dry_run=args.dry_run,
        force=args.force_manifests,
    )
    if args.command == "build-manifests":
        print(f"[experiment1] run directory: {run_root}")
        return

    methods = list(config["models"]["openwam"]) if args.method == "all" else [args.method]
    if not args.skip_text_cache:
        prepare_text_cache(config, run_root, tasks=tasks, methods=methods, dry_run=args.dry_run)
    runner = run_parallel_methods if config["hardware"].get("parallel_gpus") else run_methods
    runner_args = dict(
        tasks=tasks,
        methods=methods,
        states_per_task=states_per_task,
        rollouts_per_state=rollouts_per_state,
        max_action_steps=args.max_action_steps,
        dry_run=args.dry_run,
    )
    if partial:
        runner_args["only_jobs"] = {
            (method, task, state_id, rollout_id)
            for task in tasks for state_id in range(states_per_task)
            for method in methods for rollout_id in range(start, end)
        }
    runner(config, run_root, **runner_args)
    if not args.dry_run and not partial:
        summarize(run_root)
    print(f"[experiment1] run directory: {run_root}", flush=True)


if __name__ == "__main__":
    main()
