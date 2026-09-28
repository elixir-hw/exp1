#!/usr/bin/env python3
"""Fixed-state RoboTwin probe for Experiment 1.

This script runs inside the RoboTwin Python environment. It reuses OpenWAM's
existing RoboTwin wrapper bootstrap, then adds two operations that Experiment 3
does not need:

* build_state_manifest: choose expert-solvable fixed simulator seeds.
* run_state_rollouts: repeatedly sample the policy from one fixed seed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp_path = Path(handle.name)
        try:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(path)


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_runtime_link(name: str, env_var: str) -> None:
    source = os.environ.get(env_var, "").strip()
    if not source:
        return
    src = Path(source).resolve()
    if not src.exists():
        raise FileNotFoundError(f"{env_var} does not exist: {src}")
    dst = Path.cwd() / name
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src, target_is_directory=src.is_dir())
    print(f"[state_probe] linked {dst} -> {src}")


def bootstrap_robotwin_with_links(wrapper):
    robotwin_path = os.environ.get("ROBOTWIN_PATH")
    if not robotwin_path:
        raise SystemExit("ROBOTWIN_PATH must be set")

    wrapper._warn_on_unverified_robotwin(robotwin_path)
    runtime_root = wrapper._prepare_runtime_root(robotwin_path)
    os.chdir(runtime_root)
    ensure_runtime_link("task_config", "ROBOTWIN_TASK_CONFIG_DIR")
    ensure_runtime_link("assets", "ROBOTWIN_ASSETS_DIR")
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)

    wrapper._prewarm_cuda_for_curobo()
    wrapper._patch_warp_torch_namespace()
    module = wrapper._load_robotwin_eval_module(robotwin_path)
    patch_runtime_constants(module)
    patch_sapien_urdf_loader()
    if os.environ.get("ROBOTWIN_ENABLE_PLANNER_FALLBACK", "") == "1":
        wrapper._install_robot_planner_fallbacks(robotwin_path)
        patch_mplib_plan_path_signature()
    wrapper._install_env_trace_hooks(module)
    return module


def patch_runtime_constants(module) -> None:
    import envs

    runtime_root = Path.cwd()
    root = str(runtime_root) + os.sep
    task_config = str(runtime_root / "task_config") + os.sep
    assets = str(runtime_root / "assets") + os.sep
    if hasattr(module, "parent_directory"):
        script_context = runtime_root / "_script_context"
        script_context.mkdir(exist_ok=True)
        module.parent_directory = str(script_context)
    targets = [module, envs]
    targets.extend(
        loaded for name, loaded in sys.modules.items() if name == "envs" or name.startswith("envs.")
    )
    for target in targets:
        if target is None:
            continue
        if hasattr(target, "ROOT_PATH"):
            setattr(target, "ROOT_PATH", root)
        if hasattr(target, "CONFIGS_PATH"):
            setattr(target, "CONFIGS_PATH", task_config)
        if hasattr(target, "ASSETS_PATH"):
            setattr(target, "ASSETS_PATH", assets)
        if hasattr(target, "EMBODIMENTS_PATH"):
            setattr(target, "EMBODIMENTS_PATH", assets + "embodiments/")
        if hasattr(target, "TEXTURES_PATH"):
            setattr(target, "TEXTURES_PATH", assets + "background_texture/")


def patch_sapien_urdf_loader() -> None:
    """Work around a broken SAPIEN wheel that ships one malformed open() call."""
    try:
        import sapien.wrapper as wrapper_pkg
    except Exception:
        return

    module_name = "sapien.wrapper.urdf_loader"
    if module_name in sys.modules:
        return
    source_path = Path(wrapper_pkg.__file__).resolve().parent / "urdf_loader.py"
    if not source_path.is_file():
        return
    source = source_path.read_text(encoding="utf-8")
    replacements = {
        'with open(urdf_file, "r", encoding="utf-8" as f:': 'with open(urdf_file, "r", encoding="utf-8") as f:',
        'with open(srdf_file, "r", encoding="utf-8" as f:': 'with open(srdf_file, "r", encoding="utf-8") as f:',
    }
    if not any(broken in source for broken in replacements):
        return
    for broken, fixed in replacements.items():
        source = source.replace(broken, fixed)

    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(source_path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "sapien.wrapper"
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    print(f"[state_probe] patched in-memory {module_name} from {source_path}")


def patch_mplib_plan_path_signature() -> None:
    try:
        from envs.robot.planner import MplibPlanner
    except Exception:
        return
    original = getattr(MplibPlanner, "plan_path", None)
    if original is None or getattr(original, "_experiment1_patched", False):
        return

    def plan_path_compat(self, now_qpos, target_pose, *args, **kwargs):
        kwargs.pop("constraint_pose", None)
        return original(self, now_qpos, target_pose, *args, **kwargs)

    plan_path_compat._experiment1_patched = True
    MplibPlanner.plan_path = plan_path_compat
    print("[state_probe] patched MplibPlanner.plan_path signature")


def load_policy_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(
        {
            "task_name": args.task,
            "task_config": args.mode,
            "ckpt_setting": args.ckpt_setting,
            "seed": args.seed,
            "instruction_type": args.instruction_type,
            "host": args.host,
            "port": args.port,
            "policy_name": "openwam2robotwin_interface",
            "skip_get_obs_within_replan": args.skip_get_obs_within_replan,
        }
    )
    return config


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def prepare_task(module, usr_args: dict[str, Any]) -> tuple[Any, dict[str, Any], str | None]:
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        env_args = yaml.load(handle.read(), Loader=yaml.FullLoader)

    env_args["task_name"] = task_name
    env_args["task_config"] = task_config
    env_args["ckpt_setting"] = usr_args["ckpt_setting"]
    env_args["policy_name"] = usr_args["policy_name"]
    env_args["eval_mode"] = True
    env_args["eval_video_log"] = False

    embodiment_type = env_args.get("embodiment")
    embodiment_config_path = os.path.join(module.CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle, Loader=yaml.FullLoader)

    def embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file for {name}")
        robot_path = Path(str(robot_file))
        if robot_path.is_absolute():
            return str(robot_path)
        return str((Path.cwd() / str(robot_file).lstrip("./")).resolve())

    with open(module.CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as handle:
        camera_config = yaml.load(handle, Loader=yaml.FullLoader)
    head_camera_type = env_args["camera"]["head_camera_type"]
    env_args["head_camera_h"] = camera_config[head_camera_type]["h"]
    env_args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        env_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        env_args["right_robot_file"] = embodiment_file(embodiment_type[0])
        env_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        env_args["left_robot_file"] = embodiment_file(embodiment_type[0])
        env_args["right_robot_file"] = embodiment_file(embodiment_type[1])
        env_args["embodiment_dis"] = embodiment_type[2]
        env_args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    env_args["left_embodiment_config"] = module.get_embodiment_config(env_args["left_robot_file"])
    env_args["right_embodiment_config"] = module.get_embodiment_config(env_args["right_robot_file"])
    usr_args["left_arm_dim"] = len(env_args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(env_args["right_embodiment_config"]["arm_joints_name"][1])

    video_size = module.get_eval_video_size(env_args) if env_args.get("eval_video_log") else None
    task_env = module.class_decorator(task_name)
    patch_runtime_constants(module)
    return task_env, env_args, video_size


def choose_instruction(module, task_name: str, episode_info: dict[str, Any], instruction_type: str, state_id: int) -> str:
    descriptions = module.generate_episode_descriptions(task_name, [episode_info["info"]], 1)
    candidates = descriptions[0][instruction_type]
    if not candidates:
        raise ValueError(f"No {instruction_type!r} instruction generated for {task_name}")
    return str(candidates[state_id % len(candidates)])


def safe_close(task_env) -> None:
    try:
        task_env.close_env()
    except Exception:
        pass


def build_state_manifest(module, usr_args: dict[str, Any], args: argparse.Namespace) -> None:
    task_env, env_args, _video_size = prepare_task(module, usr_args)
    start_seed = 100000 * (1 + int(args.seed))
    now_seed = start_seed
    state_id = 0
    attempts = 0
    consecutive_infra_errors = 0
    states: list[dict[str, Any]] = []

    progress = {
        "operation": "build_state_manifest",
        "task": args.task,
        "mode": args.mode,
        "requested_states": args.num_states,
        "accepted_states": 0,
        "attempts": 0,
    }
    if args.progress_file:
        write_json(args.progress_file, progress)

    while len(states) < args.num_states and attempts < args.max_manifest_attempts:
        attempts += 1
        try:
            task_env.setup_demo(now_ep_num=state_id, seed=now_seed, is_test=True, **env_args)
            episode_info = task_env.play_once()
            accepted = bool(task_env.plan_success and task_env.check_success())
            if accepted:
                consecutive_infra_errors = 0
                states.append(
                    {
                        "state_id": state_id,
                        "accepted_seed": now_seed,
                        "now_ep_num": state_id,
                        "instruction": choose_instruction(
                            module, args.task, episode_info, args.instruction_type, state_id
                        ),
                        "expert_check": {"plan_success": True, "check_success": True},
                    }
                )
                state_id += 1
        except module.UnStableError:
            pass
        except Exception as exc:
            consecutive_infra_errors += 1
            print(f"[state_probe] manifest candidate failed seed={now_seed}: {type(exc).__name__}: {exc}")
            if consecutive_infra_errors <= args.max_consecutive_infra_errors:
                traceback.print_exc()
            if consecutive_infra_errors >= args.max_consecutive_infra_errors:
                raise RuntimeError(
                    "Stopping manifest build after "
                    f"{consecutive_infra_errors} consecutive infrastructure errors. "
                    "The simulator/render stack is likely not healthy; see the first traceback above."
                ) from exc
        finally:
            safe_close(task_env)

        now_seed += 1
        progress.update({"accepted_states": len(states), "attempts": attempts, "last_seed": now_seed - 1})
        if args.progress_file:
            write_json(args.progress_file, progress)

    if len(states) < args.num_states:
        raise RuntimeError(
            f"Only found {len(states)} accepted states for {args.task} after "
            f"{attempts} attempts; requested {args.num_states}."
        )

    payload = {
        "task": args.task,
        "mode": args.mode,
        "num_states": len(states),
        "instruction_type": args.instruction_type,
        "start_seed": start_seed,
        "attempts": attempts,
        "states": states,
    }
    payload["manifest_hash"] = stable_hash(payload)
    write_json(args.result_file, payload)


def run_one_rollout(
    module,
    task_env,
    env_args: dict[str, Any],
    model,
    eval_func,
    reset_func,
    state: dict[str, Any],
    rollout_id: int,
    policy_seed: int,
    skip_get_obs_within_replan: bool,
    max_action_steps: int | None,
) -> dict[str, Any]:
    np.random.seed(policy_seed % (2**32 - 1))
    started_at = time.monotonic()
    result: dict[str, Any] = {
        "rollout_id": rollout_id,
        "policy_seed": policy_seed,
        "success": False,
        "terminal_step": None,
        "truncated": False,
        "error": None,
        "elapsed_sec": None,
        "timing": {"setup_sec": 0.0, "get_obs_sec": 0.0, "eval_sec": 0.0,
                   "teardown_sec": 0.0, "obs_count": 0, "action_steps": 0},
    }
    try:
        phase_start = time.monotonic()
        reset_func(model)
        if hasattr(model, "set_sampling_seed"):
            model.set_sampling_seed(policy_seed)
        task_env.setup_demo(
            now_ep_num=int(state["now_ep_num"]),
            seed=int(state["accepted_seed"]),
            is_test=True,
            **env_args,
        )
        task_env.set_instruction(instruction=str(state["instruction"]))
        result["timing"]["setup_sec"] = round(time.monotonic() - phase_start, 3)
        action_limit = min(task_env.step_lim, max_action_steps) if max_action_steps else task_env.step_lim
        while task_env.take_action_cnt < action_limit:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())
            phase_start = time.monotonic()
            observation = task_env.get_obs() if need_obs else None
            result["timing"]["get_obs_sec"] += time.monotonic() - phase_start
            result["timing"]["obs_count"] += int(need_obs)
            phase_start = time.monotonic()
            eval_func(task_env, model, observation)
            result["timing"]["eval_sec"] += time.monotonic() - phase_start
            if task_env.eval_success:
                result["success"] = True
                break
        result["terminal_step"] = int(getattr(task_env, "take_action_cnt", -1))
        result["timing"]["action_steps"] = result["terminal_step"]
        result["truncated"] = bool(
            max_action_steps and not result["success"] and task_env.take_action_cnt < task_env.step_lim
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        phase_start = time.monotonic()
        safe_close(task_env)
        result["timing"]["teardown_sec"] = round(time.monotonic() - phase_start, 3)
        for key in ("get_obs_sec", "eval_sec"):
            result["timing"][key] = round(result["timing"][key], 3)
        result["elapsed_sec"] = round(time.monotonic() - started_at, 3)
    return result


def run_state_rollouts(module, usr_args: dict[str, Any], args: argparse.Namespace) -> None:
    manifest = load_json(args.state_manifest)
    states = manifest["states"]
    if args.state_id < 0 or args.state_id >= len(states):
        raise IndexError(f"state_id {args.state_id} out of range for {args.state_manifest}")
    state = states[args.state_id]

    task_env, env_args, _video_size = prepare_task(module, usr_args)
    policy_name = env_args["policy_name"]
    get_model = module.eval_function_decorator(policy_name, "get_model")
    eval_func = module.eval_function_decorator(policy_name, "eval")
    reset_func = module.eval_function_decorator(policy_name, "reset_model")
    model = get_model(usr_args)

    rollouts: list[dict[str, Any]] = []
    progress = {
        "operation": "run_state_rollouts",
        "task": args.task,
        "mode": args.mode,
        "state_id": args.state_id,
        "num_rollouts": args.num_rollouts,
        "completed": 0,
        "successes": 0,
    }
    if args.progress_file:
        write_json(args.progress_file, progress)

    for rollout_id in range(args.rollout_id_start, args.rollout_id_start + args.num_rollouts):
        policy_seed = int(args.policy_seed_base + args.state_id * 1000 + rollout_id)
        row = run_one_rollout(
            module,
            task_env,
            env_args,
            model,
            eval_func,
            reset_func,
            state,
            rollout_id,
            policy_seed,
            bool(usr_args.get("skip_get_obs_within_replan", False)),
            args.max_action_steps,
        )
        rollouts.append(row)
        progress.update(
            {
                "completed": len(rollouts),
                "successes": sum(1 for item in rollouts if item["success"]),
                "last_error": row["error"],
            }
        )
        if args.progress_file:
            write_json(args.progress_file, progress)

    successes = sum(1 for item in rollouts if item["success"])
    payload = {
        "task": args.task,
        "mode": args.mode,
        "state_id": args.state_id,
        "accepted_seed": state["accepted_seed"],
        "instruction": state["instruction"],
        "method": args.method,
        "checkpoint_dir": args.checkpoint_dir,
        "manifest_hash": manifest.get("manifest_hash"),
        "num_rollouts": args.num_rollouts,
        "successes": successes,
        "valid_rollouts": sum(1 for item in rollouts if item["error"] is None),
        "infra_failures": sum(1 for item in rollouts if item["error"] is not None),
        "rollouts": rollouts,
    }
    write_json(args.result_file, payload)
    if payload["infra_failures"]:
        raise RuntimeError(
            f"{payload['infra_failures']} rollout(s) failed for {args.task} state {args.state_id}; "
            f"see {args.result_file}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("build_state_manifest", "run_state_rollouts"), required=True)
    parser.add_argument("--openwam-wrapper", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", default="demo_clean")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8848)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--ckpt-setting", default="openwam")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--num-states", type=int, default=20)
    parser.add_argument("--max-manifest-attempts", type=int, default=1000)
    parser.add_argument("--max-consecutive-infra-errors", type=int, default=3)
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--state-id", type=int, default=0)
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--rollout-id-start", type=int, default=0)
    parser.add_argument("--max-action-steps", type=int)
    parser.add_argument("--policy-seed-base", type=int, default=200000)
    parser.add_argument("--skip-get-obs-within-replan", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wrapper = load_module_from_path("openwam_robotwin_eval_policy_wrapper", args.openwam_wrapper)
    module = bootstrap_robotwin_with_links(wrapper)
    usr_args = load_policy_config(args.policy_config, args)

    if args.operation == "build_state_manifest":
        build_state_manifest(module, usr_args, args)
    elif args.operation == "run_state_rollouts":
        if args.state_manifest is None:
            raise ValueError("--state-manifest is required for run_state_rollouts")
        run_state_rollouts(module, usr_args, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
