# Experiment 1: OpenWAM fixed-state success probability evaluation plan

## 0. Scope

This document describes how to evaluate Experiment 1 in
`wm_action_learning_experiment_plan.md` for the two trained OpenWAM models:

| Method | Training setting | Checkpoint dir |
|---|---|---|
| `no_wm` | action-only, video loss off | `/mnt/data/chw/model/openwam_train_runs/robotwin_clean40_action_only_no_wm_50k_8gpu_bs2_acc2_gbs32_20260926_055033/2026-09-26_05-51-27` |
| `wm` | full WM, video loss on | `/mnt/data/chw/model/openwam_train_runs/robotwin_clean40_wm_full_50k_8gpu_bs2_acc2_gbs32_20260926_055033/2026-09-26_05-51-23` |

Default checkpoint: `checkpoint_step_50000.safetensors` under each directory,
unless a later/manual checkpoint is explicitly selected. Both directories also
contain `config.yaml` and `normalization_stats.npy`, so the OpenWAM deploy path
should use the checkpoint directory as the model root and let the server resolve
the training config/statistics exactly as in Experiment 3.

The user will later provide the final 10 representative seen tasks. Until then,
the task list should be a placeholder file such as
`exp1/tasks/experiment1_representative_10.txt`.

## 1. Evaluation Metrics

### 1.1 Unit of measurement

The core unit is one fixed simulator initial state:

```text
task_name + task_config + accepted_seed + instruction
```

For each selected task, generate or reuse 20 fixed initial states. For each
state, run 32 stochastic rollouts with `no_wm` and 32 stochastic rollouts with
`wm`. The environment state, task config, and instruction are held fixed across
the two models; only policy sampling changes.

Formal size:

```text
10 tasks * 20 states/task * 32 rollouts/state/model * 2 models = 12,800 rollouts
```

Use `demo_clean` first, because Experiment 1 is meant to explain the core
seen-task phenomenon before adding environment randomization.

### 1.2 Primary per-state metrics

For every `(task, state_id, method)`:

| Metric | Definition | Why it matters |
|---|---|---|
| `successes` | Count of successful rollouts among 32 | Raw Bernoulli evidence |
| `num_rollouts` | Always 32 for formal runs | Denominator audit |
| `p_success` | `successes / num_rollouts` | Main state-level success probability |
| `binomial_se` | `sqrt(p * (1 - p) / 32)` | Uncertainty for error bars |
| `ci95_low`, `ci95_high` | Wilson 95% confidence interval | More stable near 0 or 1 than normal CI |
| `valid_rollouts` | Rollouts that reached evaluation rather than infra failure | Separates model failure from runner failure |
| `infra_failures` | Crashes/timeouts/env instability after accepted state selection | Quality control |

For every fixed state, join `no_wm` and `wm` rows and compute:

| Metric | Definition |
|---|---|
| `delta_p` | `p_success_wm - p_success_no_wm` |
| `relative_delta` | `(p_success_wm - p_success_no_wm) / max(p_success_no_wm, eps)` |
| `state_category` | One of the categories below |

Recommended state categories:

| Category | Rule with 32 rollouts |
|---|---|
| `newly_solved` | `no_wm_successes == 0` and `wm_successes > 0` |
| `amplified_existing` | `no_wm_successes > 0` and `wm_successes > no_wm_successes` |
| `unchanged_easy` | both `p_success >= 0.875` |
| `unchanged_hard` | both `successes == 0` |
| `regressed` | `wm_successes < no_wm_successes` |
| `same_other` | all remaining states |

The main claim-supporting number is not only the mean success rate. It is the
distribution of `(p_no_wm, p_wm)` across the same fixed states, especially how
many states are `amplified_existing` versus `newly_solved`.

### 1.3 Aggregate metrics

Report aggregates at three levels:

1. Per task:
   - mean `p_success` for `no_wm` and `wm`
   - mean `delta_p`
   - count and fraction of each `state_category`
   - paired bootstrap 95% CI over the 20 states for mean `delta_p`

2. Across all 200 states:
   - mean `p_success_no_wm`
   - mean `p_success_wm`
   - mean and median `delta_p`
   - category histogram
   - paired bootstrap 95% CI over states

3. Rollout-level sanity:
   - total successful rollouts per method
   - total infra failures per method
   - timeout count per method
   - distribution of rollout length / terminal step if available

### 1.4 Figures

Minimum figures for the paper/debug note:

1. Scatter plot: x-axis `p_success_no_wm`, y-axis `p_success_wm`, one point per
   fixed state, diagonal `y=x`.
2. Histogram of `delta_p`.
3. Stacked bar chart of `state_category` counts per task.
4. Optional paired slope plot per task: each state connects `no_wm` to `wm`.

The first scatter plot is the key Experiment 1 visualization. If the hypothesis
is right, many points should move upward from small nonzero x-values rather than
only appearing on the `x=0, y>0` edge.

## 2. Reuse From Task 3 Implementation

Existing Experiment 3 code should be reused for:

| Existing component | Reuse |
|---|---|
| `exp1/run_experiment3.py` | Pattern for config loading, path resolution, run directory creation, metadata, dry-run, summary generation |
| `exp1/wm_eval/runtime.py` | Shared command execution, JSON/task loading, environment setup |
| `exp1/wm_eval/adapters/openwam.py` | OpenWAM server lifecycle: launch `scripts/deploy.py`, wait for port, run RoboTwin client, terminate server |
| `exp1/config.formal.template.json` | Structure for model paths, runtime command, GPU IDs, OpenWAM/RoboTwin paths |

Experiment 1 should not be forced into the Experiment 3 result contract because
Experiment 3 writes one success rate per task/condition, while Experiment 1
needs per-state and per-rollout records. Implement a separate runner:

```text
exp1/run_experiment1.py
exp1/config.experiment1.template.json
exp1/wm_eval/experiment1_runtime.py      # optional if result helpers grow large
```

The OpenWAM adapter can either be extended with a dedicated method such as
`run_state_probe(...)`, or a small `OpenWAMExperiment1Adapter` can subclass/use
the existing server startup logic. Prefer extracting server lifecycle into a
small shared helper if the code duplication becomes noticeable.

## 3. Required Model/Evaluation Code Changes

The trained OpenWAM model weights do not need to change. The required changes
are evaluation-interface changes around the model server and RoboTwin driver.

### 3.1 OpenWAM model server

Target files:

```text
/home/chw/code/packages/OpenWAM/OpenWAM/openwam/deploy/server.py
/home/chw/code/packages/OpenWAM/OpenWAM/openwam/deploy/policy.py
/home/chw/code/packages/OpenWAM/OpenWAM/scripts/deploy.py
```

Required logic:

1. Keep using `scripts/deploy.py --ckpt-dir <checkpoint_dir>`.
2. Confirm that `--ckpt-dir` can load the 50k checkpoint inside the model
   directory. If the deploy code expects a single checkpoint file rather than
   the directory, add a deploy arg such as:

   ```text
   --checkpoint checkpoint_step_50000.safetensors
   ```

   or resolve `checkpoint_step_50000.safetensors` by default in the runner.
3. Ensure WebSocket `reset` resets all receding-horizon/action-buffer state
   before every rollout. Experiment 1 relies on 32 independent policy samples
   from the same simulator state.
4. If OpenWAM has an explicit sampling seed/generator path, expose an optional
   per-rollout seed in the request or reset payload:

   ```json
   {"type": "reset", "seed": 12345}
   ```

   If no such hook exists, keep the model stochastic and record rollout index;
   do not make the rollout deterministic accidentally by reusing the same
   random seed for all 32 samples.
5. Do not change training-time loss, architecture, normalization, or action
   decoding. Any change here would contaminate the comparison.

Expected conclusion: model code changes should be minimal or zero. The only
acceptable model-side change is an evaluation-only reset/seed control for
clean repeated sampling.

### 3.2 OpenWAM RoboTwin wrapper/client

Target file:

```text
/home/chw/code/packages/OpenWAM/OpenWAM/benchmarks/robotwin/eval_policy_wrapper.py
```

Task 3 already uses this wrapper as the RoboTwin entrypoint. For Experiment 1,
add an operation dedicated to fixed-state repeated rollout, for example:

```text
python benchmarks/robotwin/eval_policy_wrapper.py labtasker \
  --operation run_state_rollouts \
  --task <task_name> \
  --mode demo_clean \
  --policy-config benchmarks/robotwin/policy_config.yml \
  --host 127.0.0.1 \
  --port 8848 \
  --state-manifest <manifest.json> \
  --state-id <state_id> \
  --num-rollouts 32 \
  --result-file <result.json> \
  --progress-file <progress.json> \
  --instruction-type unseen
```

Required logic:

1. Bootstrap RoboTwin exactly as the current wrapper does:
   - `ROBOTWIN_PATH`
   - writable `ROBOTWIN_RUNTIME_ROOT`
   - CUDA/Curobo prewarm
   - `warp.torch` compatibility patch
   - planner fallback if enabled
   - trace hooks around `setup_demo` and `play_once`
2. Load the selected fixed state from `state-manifest`.
3. For each rollout:
   - call server `reset`
   - call `TASK_ENV.setup_demo(now_ep_num=<state_id>, seed=<accepted_seed>, is_test=True, **args)`
   - set exactly the instruction stored in the manifest
   - run policy until `TASK_ENV.eval_success` or `TASK_ENV.step_lim`
   - record success, terminal step, exception/timeout if any
   - close env after the rollout
4. Do not rerun expert planning checks inside the 32 stochastic rollouts. Expert
   checks are used only while building the state manifest to select stable,
   solvable simulator seeds.
5. Save a result JSON with full per-rollout records rather than only a text
   success rate.

Recommended result schema:

```json
{
  "task": "place_object_basket",
  "mode": "demo_clean",
  "state_id": 3,
  "accepted_seed": 100021,
  "instruction": "place the object into the basket",
  "method": "wm",
  "checkpoint_dir": "...",
  "num_rollouts": 32,
  "successes": 11,
  "rollouts": [
    {
      "rollout_id": 0,
      "policy_seed": 203000,
      "success": true,
      "terminal_step": 117,
      "error": null
    }
  ]
}
```

### 3.3 RoboTwin state manifest builder

Task 3 uses manifests to keep evaluation episodes shared. Experiment 1 needs a
smaller but stricter manifest: 20 accepted initial states per task, each with a
fixed instruction.

Implement this either inside the OpenWAM wrapper operation
`build_state_manifest`, or as a small helper called by
`exp1/run_experiment1.py`.

Recommended manifest schema:

```json
{
  "task": "place_object_basket",
  "mode": "demo_clean",
  "num_states": 20,
  "instruction_type": "unseen",
  "manifest_hash": "...",
  "states": [
    {
      "state_id": 0,
      "accepted_seed": 100000,
      "now_ep_num": 0,
      "instruction": "place the object into the basket",
      "expert_check": {
        "plan_success": true,
        "check_success": true
      }
    }
  ]
}
```

State selection logic:

1. Start from `st_seed = 100000 * (1 + seed)` to match RoboTwin eval policy.
2. Iterate seeds until 20 expert-solvable seeds are found.
3. For each candidate seed, run the expert check once:
   - `TASK_ENV.setup_demo(..., seed=now_seed, is_test=True, **args)`
   - `episode_info = TASK_ENV.play_once()`
   - accept only if `TASK_ENV.plan_success and TASK_ENV.check_success()`
4. Generate the instruction once from `episode_info["info"]` using
   `generate_episode_descriptions(...)`.
5. Store the exact instruction string. During model rollouts, reuse it instead
   of sampling a new instruction.

This avoids a common confound: if each of the 32 rollouts samples a different
instruction, the measured variance is no longer only policy sampling variance.

### 3.4 Experiment 1 runner in this repository

Target files to add/change:

```text
exp1/run_experiment1.py
exp1/config.experiment1.template.json
exp1/tasks/experiment1_representative_10.txt
exp1/wm_eval/adapters/openwam.py        # small extension or shared server helper
```

Runner behavior:

1. Validate:
   - 10 task names are present and unique.
   - Tasks are a subset of `exp1/tasks/seen_40.txt`.
   - Both checkpoint dirs exist.
   - Both contain `config.yaml`, `normalization_stats.npy`, and
     `checkpoint_step_50000.safetensors`.
   - OpenWAM/RoboTwin paths point to the runtime layout expected by the wrapper.
2. Build/reuse state manifests:
   - output root: `exp1/manifests/experiment1_20_states/<task>/demo_clean/manifest.json`
   - skip rebuild if manifest exists and has at least 20 states.
3. For each method:
   - start one OpenWAM server with the selected checkpoint dir.
   - loop over tasks and fixed states.
   - call wrapper `run_state_rollouts` with `--num-rollouts 32`.
   - stop server after the method finishes.
4. Summarize:
   - write normalized per-state and per-rollout CSV/JSON.
   - compute categories and aggregate deltas.
   - write a Markdown summary and figure-ready CSVs.

Suggested output layout:

```text
exp1/runs/experiment1/<run-id>/
  config.json
  metadata.json
  manifests/
  openwam/no_wm/server.log
  openwam/no_wm/raw/<task>/state_<id>/result.json
  openwam/wm/server.log
  openwam/wm/raw/<task>/state_<id>/result.json
  summary/per_rollout.csv
  summary/per_state.csv
  summary/per_task.csv
  summary/state_categories.csv
  summary/summary.md
```

### 3.5 Configuration template

`config.experiment1.template.json` should mirror Experiment 3 but remove
FastWAM and randomized/unseen splits:

```json
{
  "label": "experiment1-openwam-fixed-state",
  "paths": {
    "openwam_repo": "/home/chw/code/packages/OpenWAM/OpenWAM",
    "robotwin_repo": "REPLACE_ME",
    "robotwin_python": "REPLACE_ME",
    "state_manifest_root": "manifests/experiment1_20_states",
    "task_file": "tasks/experiment1_representative_10.txt"
  },
  "protocol": {
    "mode": "demo_clean",
    "states_per_task": 20,
    "rollouts_per_state": 32,
    "seed": 0,
    "instruction_type": "unseen",
    "checkpoint_name": "checkpoint_step_50000.safetensors"
  },
  "hardware": {
    "openwam_model_gpu": 6,
    "openwam_sim_gpu": 0,
    "openwam_host": "127.0.0.1",
    "openwam_port": 8848
  },
  "models": {
    "openwam": {
      "no_wm": {
        "checkpoint_dir": "/mnt/data/chw/model/openwam_train_runs/robotwin_clean40_action_only_no_wm_50k_8gpu_bs2_acc2_gbs32_20260926_055033/2026-09-26_05-51-27",
        "deploy_args": []
      },
      "wm": {
        "checkpoint_dir": "/mnt/data/chw/model/openwam_train_runs/robotwin_clean40_wm_full_50k_8gpu_bs2_acc2_gbs32_20260926_055033/2026-09-26_05-51-23",
        "deploy_args": []
      }
    }
  }
}
```

## 4. Execution Plan After Task List Is Provided

1. Fill `exp1/tasks/experiment1_representative_10.txt` with the selected tasks.
2. Add `config.experiment1.json` from the template and resolve local Python/GPU
   paths.
3. Implement wrapper operations:
   - `build_state_manifest`
   - `run_state_rollouts`
4. Implement `run_experiment1.py`:
   - `validate`
   - `build-manifests`
   - `run`
   - `summarize`
5. Smoke test one task, one state, two rollouts:

   ```text
   python exp1/run_experiment1.py run --config exp1/config.experiment1.json \
     --task-limit 1 --state-limit 1 --rollouts-per-state 2 --dry-run
   ```

6. Run a real smoke test on one task/state with both models.
7. Run the formal 10-task evaluation.
8. Inspect `summary/per_state.csv` and the scatter plot input before writing
   claims.

## 5. Interpretation Rules

The Experiment 1 conclusion should be framed from paired fixed-state evidence:

- Strong support for the intended hypothesis:
  - `amplified_existing` is common.
  - `newly_solved` exists but is not the dominant category.
  - paired mean `delta_p` is positive with bootstrap CI mostly above 0.

- Weak or mixed support:
  - improvement comes mostly from `newly_solved`.
  - many states regress.
  - large gains occur only in a small number of tasks.

- Not enough evidence:
  - many infra failures.
  - accepted states differ between models.
  - instructions are not fixed per state.
  - rollouts accidentally become deterministic and all 32 samples are identical.

## 6. Open Questions / User Inputs Needed

1. The final 10 representative seen tasks.
2. Whether to save rollout videos. Default recommendation: disable videos for
   the formal run, enable only for a small debug subset.
3. Whether to use only `checkpoint_step_50000.safetensors` or compare multiple
   checkpoints later. Default for Experiment 1: only 50k.

## 7. Implementation and Smoke Protocol (2026-09-28)

The first implementation is in `exp1/run_experiment1.py` and
`exp1/wm_eval/robotwin_state_probe.py`. The temporary task file is
`exp1/tasks/experiment1_smoke_10.txt`. The local config is
`exp1/config.experiment1.smoke.json`; a shareable example is
`exp1/config.experiment1.smoke.example.json`. The final representative list
can replace the temporary task file without changing evaluator code.

The fixed-state manifests are built once from expert-solvable RoboTwin seeds.
Both checkpoints replay the same accepted seed and the same instruction for
each state. The probe uses the same OpenWAM RoboTwin client as Experiment 3 and
records raw rollout results and per-state summaries. The simulator is on GPU 1,
OpenWAM deployment is on GPU 0, and missing text embeddings are encoded on GPU
2; no GPU above index 3 is used. Each model server is started separately on
port 8848. The runner checks that the port is free before each launch.

One model-side fix was required in
`OpenWAM/openwam/model/video_backbone/wan_backbone.py`:
`preprocess_input_for_inference()` now calls `TextEmbeddingCache.load_batch()`
when the checkpoint uses cached text embeddings. Both trained checkpoints set
`load_text_encoder: false`; the previous inference path dereferenced
`self.text_encoder` unconditionally and failed on the first request. The
non-cached path still uses the text encoder. No checkpoint weights changed.

RoboTwin's generated evaluation instructions are often absent from the
training text cache. `exp1/wm_eval/ensure_text_cache.py` hashes the wrapped
instruction with the cache's original identity and encodes only missing
prompts with the original Wan T5 weights. It writes compatible embedding
records into the cache before model deployment. The text cache is shared by
both checkpoints. A cache miss during rollout is an infrastructure failure,
not a failed task attempt.

The smoke config disables `torch.compile` for deployment because its first
request compilation took several minutes. The formal config may enable it.
For the 10-task smoke run, use one fixed state, one rollout, and an action cap:

```bash
python exp1/run_experiment1.py run \
  --config exp1/config.experiment1.smoke.json \
  --state-limit 1 --rollouts-per-state 1 --max-action-steps 16
```

The cap checks environment setup, instruction handling, server communication,
model sampling, and action execution on all 10 tasks. Capped rollouts are
marked `truncated` and summarized in `summary/smoke_rollouts.csv`; they are
never used to estimate task success probability. Omit `--max-action-steps` for
full-episode evaluation. The one-state full-episode sanity run completed for
both checkpoints on `beat_block_hammer` (0/1 and 1/1 successes, respectively);
these single-trial values are only a pipeline check.

The summary code computes paired bootstrap intervals at task and pooled-state
level and checks that the two model records share a manifest, seed,
instruction, and rollout count. The planned figures are not yet generated;
`per_state.csv` and `state_categories.csv` contain their source data.
The first rollout implementation recorded a simulator-side `policy_seed`, but
OpenWAM's inference engine defaulted to seed 42 on every replan. That run is
diagnostic only. The current evaluator sets a different policy seed for each
rollout. The RoboTwin client sends `policy_seed * 10000 + request_index` in each
observation, and `openwam/deploy/policy.py` passes it to the existing engine
`seed` argument. This makes stochastic model sampling distinct across rollouts
and replans while keeping the simulator seed fixed. The five-task comparison
uses one independent model server per rollout and merges those records only
after all paired samples finish.

The user-selected five-task test is configured in
`exp1/config.experiment1.five_tasks.json` with task list
`exp1/tasks/experiment1_selected_5.txt`. It uses one accepted initial state
and four full-episode rollouts per model and task (40 rollouts total), with up
to three independent jobs on each of GPUs 0, 1, 2, and 3. These four samples
per model give only coarse probability estimates; zero or four successes do
not establish a true probability of exactly 0 or 1. The source rollout JSON
files, Wilson intervals, and paired state differences must be reported together.
