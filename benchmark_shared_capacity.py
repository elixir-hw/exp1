#!/usr/bin/env python3
"""Measure short rollout capacity for one shared OpenWAM model on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def gpu_memory_mib(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--counts", default="4,8,12")
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-memory-mib", type=int, default=72000)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    config["hardware"].update(
        parallel_gpus=[args.gpu], persistent_server=True,
        persistent_servers_per_gpu=1, shared_server_sessions=True,
    )
    result_path = args.output / "capacity_results.json"
    results = json.loads(result_path.read_text()) if result_path.is_file() else []
    for count in [int(part) for part in args.counts.split(",")]:
        config["hardware"]["persistent_clients_per_gpu"] = count
        config_path = args.output / f"config_clients_{count}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        run_dir = args.output / f"clients_{count}"
        command = [
            sys.executable, str(Path(__file__).with_name("run_experiment1.py")), "run",
            "--config", str(config_path.resolve()), "--run-dir", str(run_dir.resolve()),
            "--method", "wm", "--task-limit", "1", "--state-limit", "1",
            "--rollouts-per-state", str(count), "--parallel-gpus", str(args.gpu),
            "--max-action-steps", str(args.max_action_steps), "--skip-text-cache",
        ]
        started = time.monotonic()
        peak_mib = 0
        timed_out = False
        memory_limited = False
        with (args.output / f"clients_{count}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                peak_mib = max(peak_mib, gpu_memory_mib(args.gpu))
                if peak_mib >= args.max_memory_mib:
                    memory_limited = True
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                    break
                if time.monotonic() - started > args.timeout_sec:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                    break
                time.sleep(1)
        rows = []
        summary = run_dir / "summary/smoke_rollouts.csv"
        if summary.is_file():
            with summary.open() as stream:
                rows = list(csv.DictReader(stream))
        item = {
            "clients_per_gpu": count, "rollouts": len(rows),
            "errors": sum(bool(row["error"]) for row in rows),
            "exit_code": process.returncode, "timed_out": timed_out,
            "memory_limited": memory_limited,
            "wall_sec": round(time.monotonic() - started, 3),
            "peak_memory_mib": peak_mib,
            "max_action_steps": args.max_action_steps,
        }
        results.append(item)
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        print(item, flush=True)
        if process.returncode or len(rows) != count or item["errors"]:
            print("Stopping capacity ramp after the first failed level", flush=True)
            break


if __name__ == "__main__":
    main()
