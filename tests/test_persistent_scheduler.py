import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_experiment1 as experiment


class LiveProcess:
    def poll(self):
        return None


class PersistentSchedulerTest(unittest.TestCase):
    def test_one_server_per_gpu_and_method_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "paths": {"state_manifest_root": str(root / "manifests"), "openwam_repo": str(root),
                          "robotwin_python": "/bin/python"},
                "protocol": {"mode": "demo_clean", "checkpoint_name": "checkpoint.safetensors", "seed": 0,
                             "instruction_type": "unseen"},
                "hardware": {"parallel_gpus": [0, 1], "openwam_host": "127.0.0.1", "openwam_port": 9000,
                             "openwam_model_gpu": 0, "openwam_sim_gpu": 0},
                "models": {"openwam": {"no_wm": {"checkpoint_dir": "no_wm"},
                                        "wm": {"checkpoint_dir": "wm"}}},
            }
            starts = []
            drivers = []

            def start_server(cfg, method, model, run_root, dry_run, server_dir=None):
                starts.append((method, cfg["hardware"]["openwam_model_gpu"]))
                return LiveProcess(), None

            def run_logged(command, *, cwd, env, log_path, dry_run, quiet):
                def value(flag):
                    return command[command.index(flag) + 1]

                method = value("--method")
                task = value("--task")
                state_id = int(value("--state-id"))
                rollout_id = int(value("--rollout-id-start"))
                drivers.append((method, task, state_id, rollout_id))
                payload = {
                    "task": task, "mode": "demo_clean", "state_id": state_id,
                    "accepted_seed": 100000, "instruction": "place it", "method": method,
                    "checkpoint_dir": method, "manifest_hash": "fixed",
                    "num_rollouts": 1, "successes": 0, "valid_rollouts": 1,
                    "infra_failures": 0, "rollouts": [{"rollout_id": rollout_id, "success": False,
                                                   "error": None, "policy_seed": 200000 + rollout_id}],
                }
                experiment.write_json(Path(value("--result-file")), payload)

            with patch.object(experiment, "start_server", side_effect=start_server), \
                 patch.object(experiment, "stop_server"), \
                 patch.object(experiment, "run_logged", side_effect=run_logged), \
                 patch.object(experiment, "make_env", return_value={}):
                for _ in range(2):
                    experiment.run_persistent_methods(
                        config, root, tasks=["place_fan"], methods=["no_wm", "wm"],
                        states_per_task=1, rollouts_per_state=4,
                        max_action_steps=None, dry_run=False,
                    )

            self.assertEqual(len(starts), 4)
            self.assertEqual(len(drivers), 8)
            preserved = root / "openwam" / "wm" / "raw" / "place_fan" / "state_000" / "rollout_001" / "result.json"
            original_bytes = preserved.read_bytes()
            missing = root / "openwam" / "wm" / "raw" / "place_fan" / "state_000" / "rollout_002" / "result.json"
            missing.unlink()
            with patch.object(experiment, "start_server", side_effect=start_server), \
                 patch.object(experiment, "stop_server"), \
                 patch.object(experiment, "run_logged", side_effect=run_logged), \
                 patch.object(experiment, "make_env", return_value={}):
                experiment.run_persistent_methods(
                    config, root, tasks=["place_fan"], methods=["no_wm", "wm"],
                    states_per_task=1, rollouts_per_state=4,
                    max_action_steps=None, dry_run=False,
                )
            self.assertEqual(len(starts), 5)
            self.assertEqual(drivers[-1], ("wm", "place_fan", 0, 2))
            self.assertEqual(len(drivers), 9)
            self.assertEqual(preserved.read_bytes(), original_bytes)
            for method in ("no_wm", "wm"):
                merged = experiment.load_json(root / "openwam" / method / "raw" /
                                              "place_fan" / "state_000" / "result.json")
                self.assertEqual([row["rollout_id"] for row in merged["rollouts"]], list(range(4)))


if __name__ == "__main__":
    unittest.main()
