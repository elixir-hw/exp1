# Experiment 1: selected five-task parallel test

## Protocol and provenance

- Corrected run: `exp1/runs/experiment1/20260928_060106`.
- Models: action-only `no_wm` and full `wm`, both at `checkpoint_step_50000.safetensors`.
- Environment: RoboTwin `demo_clean`; the five tasks are in `seen_40.txt`.
- Per task: one expert-solvable fixed initial state, one fixed instruction, four full-episode rollouts per model. The two methods use the same manifest and rollout IDs.
- OpenWAM model sampling seed: `policy_seed * 10000 + request_index`, passed from the RoboTwin client to the inference engine. Each rollout uses `policy_seed = 200000 + rollout_id` for this one-state test. This hook was added after an earlier diagnostic run was found to use the engine's default seed 42 on every replan.
- Hardware: GPUs 0-3 only, at most three independent model/server plus simulator jobs per GPU. Jobs have separate WebSocket ports and policy state. Two port collisions at 8887/8888 were retried on free ports; completed rollout files were reused.
- All 40 final rollouts are valid; infrastructure failures: 0. No rollout was step-capped.

The interrupted run `20260928_052746` is marked invalid in its metadata and must not be included in analysis.

## Results

Each cell below is successful rollouts / 4. Wilson 95% intervals are shown in parentheses. The `delta` is the difference in observed success fraction for the same fixed state.

| Task | no_wm | wm | Delta | State category |
|---|---:|---:|---:|---|
| Place Fan | 0/4 (0-49%) | 4/4 (51-100%) | +100 pp | newly_solved |
| Place A2B Right | 0/4 (0-49%) | 4/4 (51-100%) | +100 pp | newly_solved |
| Put Bottles Dustbin | 0/4 (0-49%) | 2/4 (15-85%) | +50 pp | newly_solved |
| Place Mouse Pad | 0/4 (0-49%) | 0/4 (0-49%) | 0 pp | unchanged_hard |
| Place Dual Shoes | 0/4 (0-49%) | 4/4 (51-100%) | +100 pp | newly_solved |

Across the five selected states, observed success is 0/20 for `no_wm` and 14/20 for `wm`. The unweighted mean state difference is +70 percentage points. A paired bootstrap over the five observed state differences yields a descriptive 95% interval of [+30, +100] percentage points (2,000 resamples). This interval does not cover task selection uncertainty: these five tasks were deliberately chosen, and there is only one initial state per task.

## Rollout status and interpretation

- **Place Fan:** all four `wm` rollouts succeeded at steps 128-153; all four `no_wm` rollouts failed at the 400-step limit. The placement-sensitive state showed a consistent observed gain.
- **Place A2B Right:** all four `wm` rollouts succeeded at steps 146-161; all four `no_wm` rollouts failed at step 400. This is a clean fixed-state contrast, but outcome logs alone cannot locate the exact action error.
- **Put Bottles Dustbin:** `wm` succeeded at steps 650 and 652 in two samples and failed at the 1700-step limit in two others. All four `no_wm` samples failed at step 1700. This is the clearest within-state outcome variation; action/video traces are needed to identify which substage differs.
- **Place Mouse Pad:** both methods failed all four samples at the 400-step limit. This fixed state remains unsolved in the observed samples; inspect placement tolerance and action traces before assigning a cause.
- **Place Dual Shoes:** all four `wm` samples succeeded at steps 213-223; all four `no_wm` samples failed at the 600-step limit. The observed gain extends to this dual-object task.

Under the current state-category rules, four states are `newly_solved` and one is `unchanged_hard`; none is `amplified_existing`. Thus this small selected test does not show the planned "amplify an already nonzero success probability" pattern. Zero successes in four trials does **not** establish a true zero probability, and one state per task cannot describe task-wide state coverage. The later formal experiment should sample more fixed states and more policy seeds before making that mechanism claim.

## Artifacts

- `summary/summary.md`: pooled metrics and category counts.
- `summary/per_task.csv`: per-task means and category counts.
- `summary/per_state_by_method.csv`: per-method successes, Wilson intervals, and infrastructure counts.
- `summary/per_state.csv`: paired state differences and categories.
- `summary/per_rollout.csv`: rollout IDs, seeds, outcomes, terminal steps, and duration.
- `openwam/<method>/raw/<task>/state_000/rollout_<id>/`: raw result, driver log, and model server log for each rollout.
