# Experiment 1 评测代码复用与迁移手册

本文记录截至 2026-09-29 已完成验证的 Experiment 1 推理评测流程。目标是在新机器上复用同一套代码，运行 10 或 20 个 RoboTwin 任务，并保持 initial state、policy seed、结果格式和汇总逻辑一致。

本文所说的“10/20 个任务”只改变任务数量。正式实验仍默认每个任务 20 个 initial state、每个 state 每个模型 32 次 rollout，并评测 `no_wm`、`wm` 两个模型。

## 1. 已固定的实现与验证结果

### 1.1 固定版本

| 组件 | 固定版本或要求 | 用途 |
|---|---|---|
| Experiment 1 核心评测代码 | `5ce986ac0e9ba5166219cb87c6c71cec41edf50c` | manifest、调度、常驻服务、结果合并与汇总 |
| OpenWAM | 分支 `wm-function`，提交 `1493a2d4ea11f80766b2acbbf5a24f6645079c56` | 每个 WebSocket 会话隔离 policy 状态 |
| OpenWAM 运行时补丁 | `patches/openwam-experiment1-runtime.patch` | 固定 rollout 采样 seed，并在推理时使用文本 embedding cache |
| OpenWAM 补丁 SHA256 | `edcfc9fa7c68f66a18260ec399abecdf6415db0aae7f726da0636acb8c3544ba` | 迁移后完整性检查 |
| RoboTwin 评测代码 | `c37109c500be67d0dea6b36bf7337bbd26e763cd` | 仿真环境和任务执行 |
| RoboTwin 外层仓库快照 | `45a853a9c87bd190e028db911137be01c2087d54` | `env_cfg/task_config`、assets 和 XPolicyLab 子模块入口 |

OpenWAM 的固定提交已经包含共享模型的会话隔离服务，但已验证流程还需要随本仓库保存的运行时补丁。只 checkout OpenWAM 提交而不应用补丁，无法保证 policy seed 对齐；使用卸载 T5、依赖文本 cache 的 checkpoint 时还可能无法启动推理。

### 1.2 当前执行结构

每张 GPU 在一个模型阶段只加载一份 checkpoint，并启动一个 OpenWAM 服务。该服务最多接收 8 个独立 WebSocket 会话，每个会话对应一个 RoboTwin rollout：

```text
GPU
└── 一份 OpenWAM checkpoint
    ├── WebSocket session 1 ── RoboTwin rollout
    ├── WebSocket session 2 ── RoboTwin rollout
    ├── ...
    └── WebSocket session 8 ── RoboTwin rollout
```

不同 session 的 `WAMPolicy`、action buffer、RESET 状态和请求计数互相隔离；模型权重和文本 cache 共享。模型请求在服务端串行执行，但多个仿真环境的物理步进与观测获取可并行。两个模型分阶段执行，因此每张卡同一时间只加载 `no_wm` 或 `wm` 中的一份。

默认并行参数必须保持：

```json
{
  "workers_per_gpu": 1,
  "parallel_by_rollout": true,
  "persistent_server": true,
  "shared_server_sessions": true,
  "persistent_clients_per_gpu": 8
}
```

单卡 8 个完整 A2B/WM rollout 已验证：8/8 有效、0 error、峰值显存约 66.1 GB。10 或 12 个会话曾触及约 75 GB 保护线，所以默认值固定为 8。

双机 smoke 已验证：本机 4 卡加 141 机器 3 卡，共完成 64 条完整 rollout；结果键全部唯一、0 error，两个 worker 和自动合并器均以 `exit_code=0` 结束。验证结果目录是：

```text
runs/distributed/four_task_shared_smoke_20260928_1000
```

## 2. 需要迁移的内容

### 2.1 必须迁移

1. **整个 Experiment 1 工作目录**

   核心代码可以从 Git 拉取并 checkout 固定提交。由于本手册和 OpenWAM 运行时补丁是在该提交之后补充的迁移资产，在它们进入新的 Git 提交前，必须从当前源机器一并复制。至少必须包含：

   ```text
   run_experiment1.py
   run_experiment1_distributed.py
   wm_eval/
   tasks/
   patches/openwam-experiment1-runtime.patch
   config.experiment1.distributed.four_tasks.json
   ```

2. **OpenWAM 代码**

   checkout `1493a2d4`，随后应用 `openwam-experiment1-runtime.patch`。

3. **RoboTwin 代码、任务配置和 assets**

   当前配置分别依赖：

   ```text
   RoboTwin/XPolicyLab/policy/FastWAM/FastWAM/third_party/RoboTwin
   RoboTwin/env_cfg/task_config
   RoboTwin/RoboTwin/assets
   ```

   RoboTwin 包含子模块和大文件。迁移时优先同步当前已验证的完整目录快照，或者使用 Git、Git LFS 和 submodule 严格恢复上述版本。

4. **两个 checkpoint 目录**

   每个目录必须包含：

   ```text
   config.yaml
   normalization_stats.npy
   checkpoint_step_50000.safetensors
   ```

5. **两个 Python 环境**

   - OpenWAM 推理环境，例如 `/home/chw/miniconda3/envs/openwam/bin/python`
   - RoboTwin 仿真环境，例如 `/home/chw/miniconda3/envs/RoboTwin/bin/python`

6. **正式 state manifests**

   如果需要在新机器上复现实验中的同一批 initial state，必须迁移正式配置所指向的整个 `manifests/<name>/`。重新生成 manifest 只保证遵循同一生成协议，不应当视为已经复用了同一批 state。

### 2.2 不需要迁移

- 旧的 `runs/`，除非需要续跑或保留历史证据。
- `__pycache__`、`.pytest_cache`、视频和临时 runtime。
- 文本 embedding cache 可以由启动器根据 checkpoint 和 manifest 自动准备；共享存储已有 cache 时会直接复用。

## 3. 推荐目录布局

保持与当前机器相同的绝对路径最省事：

```text
/home/chw/code/packages/
├── wm-func/exp1
├── OpenWAM/OpenWAM
└── RoboTwin/
    ├── env_cfg/task_config
    ├── RoboTwin/assets
    └── XPolicyLab/policy/FastWAM/FastWAM/third_party/RoboTwin
```

模型继续放在共享 `/mnt/data` 时通常不需要修改 checkpoint 路径。如果目标机目录不同，必须同步修改配置中的以下字段：

- `paths.openwam_repo`
- `paths.robotwin_repo`
- `paths.robotwin_python`
- `paths.robotwin_task_config_dir`
- `paths.robotwin_assets_dir`
- `runtimes.openwam.command`
- 两个 `models.openwam.<method>.checkpoint_dir`

双机启动器目前还假设两台机器的 `openwam_repo` 绝对路径一致，并将远端调度 Python 固定为 `/home/chw/miniconda3/bin/python`。目标机不满足这两个条件时，应先修改 `run_experiment1_distributed.py` 中相应路径。

## 4. 在目标机恢复代码

### 4.1 Experiment 1

优先从当前已验证机器同步完整工作目录，再用 `git rev-parse HEAD` 确认核心代码版本：

```bash
mkdir -p /home/chw/code/packages/wm-func
rsync -a --exclude runs/ --exclude manifests/ \
  SOURCE:/home/chw/code/packages/wm-func/exp1/ \
  /home/chw/code/packages/wm-func/exp1/
git -C /home/chw/code/packages/wm-func/exp1 rev-parse HEAD
```

输出应为 `5ce986ac0e9ba5166219cb87c6c71cec41edf50c`，并确认以下两个新增迁移资产存在：

```bash
test -f /home/chw/code/packages/wm-func/exp1/note/experiment1_reuse_and_migration_runbook.md
test -f /home/chw/code/packages/wm-func/exp1/patches/openwam-experiment1-runtime.patch
```

如果只能从 GitHub 拉取核心提交，还需从源机器单独复制上述手册和补丁：

```bash
mkdir -p /home/chw/code/packages/wm-func
cd /home/chw/code/packages/wm-func
git clone https://github.com/elixir-hw/exp1.git
cd exp1
git checkout 5ce986ac0e9ba5166219cb87c6c71cec41edf50c

rsync SOURCE:/home/chw/code/packages/wm-func/exp1/patches/openwam-experiment1-runtime.patch patches/
rsync SOURCE:/home/chw/code/packages/wm-func/exp1/note/experiment1_reuse_and_migration_runbook.md note/
```

### 4.2 OpenWAM

```bash
mkdir -p /home/chw/code/packages
cd /home/chw/code/packages
git lfs install
git clone https://github.com/2228618603/OpenWAM.git
cd OpenWAM
git checkout 1493a2d4ea11f80766b2acbbf5a24f6645079c56

sha256sum /home/chw/code/packages/wm-func/exp1/patches/openwam-experiment1-runtime.patch
git apply --check /home/chw/code/packages/wm-func/exp1/patches/openwam-experiment1-runtime.patch
git apply /home/chw/code/packages/wm-func/exp1/patches/openwam-experiment1-runtime.patch
```

`sha256sum` 应输出第 1.1 节记录的值。应用补丁后，下面三项应能搜索到：

```bash
rg -n 'set_sampling_seed|conditions\["seed"\]|Using cached text embeddings' \
  OpenWAM/benchmarks/robotwin/openwam2robotwin_interface.py \
  OpenWAM/openwam/deploy/policy.py \
  OpenWAM/openwam/model/video_backbone/wan_backbone.py
```

### 4.3 RoboTwin

如果源机器仍可访问，建议使用 rsync 复制已验证快照，包括链接、子模块内容和 assets：

```bash
rsync -a --info=progress2 SOURCE:/home/chw/code/packages/RoboTwin/ \
  /home/chw/code/packages/RoboTwin/
```

迁移后至少检查：

```bash
test -f /home/chw/code/packages/RoboTwin/env_cfg/task_config/demo_clean.yml
test -d /home/chw/code/packages/RoboTwin/RoboTwin/assets
test -f /home/chw/code/packages/RoboTwin/XPolicyLab/policy/FastWAM/FastWAM/third_party/RoboTwin/script/eval_policy.py
git -C /home/chw/code/packages/RoboTwin/XPolicyLab/policy/FastWAM/FastWAM rev-parse HEAD
```

最后一条应输出 `c37109c500be67d0dea6b36bf7337bbd26e763cd`。

## 5. 准备 10 或 20 个任务

### 5.1 前 10 个任务

新建 `tasks/experiment1_selected_10.txt`：

```text
place_fan
place_a2b_right
put_bottles_dustbin
place_mouse_pad
place_dual_shoes
pick_diverse_bottles
rotate_qrcode
open_laptop
beat_block_hammer
stack_blocks_three
```

不要使用现有的 `tasks/experiment1_smoke_10.txt` 作为正式任务清单；它是早期临时 smoke 列表，并不是本实验选定的前 10 个任务。

### 5.2 全部 20 个任务

新建 `tasks/experiment1_selected_20.txt`：

```text
place_fan
place_a2b_right
put_bottles_dustbin
place_mouse_pad
place_dual_shoes
pick_diverse_bottles
rotate_qrcode
open_laptop
beat_block_hammer
stack_blocks_three
place_a2b_left
place_can_basket
place_phone_stand
place_shoe
pick_dual_bottles
put_object_cabinet
place_object_basket
place_bread_skillet
scan_object
stamp_seal
```

这 20 个名称均属于 `tasks/seen_40.txt`，会通过 Experiment 1 的任务范围校验。

## 6. 创建正式配置

以已经验证的四任务配置为模板：

```bash
cd /home/chw/code/packages/wm-func/exp1
cp config.experiment1.distributed.four_tasks.json config.experiment1.selected_10.json
cp config.experiment1.distributed.four_tasks.json config.experiment1.selected_20.json
```

10 任务配置至少修改：

```json
{
  "label": "experiment1-10-task-20x32",
  "paths": {
    "state_manifest_root": "manifests/experiment1_selected_10_20_states",
    "task_file": "tasks/experiment1_selected_10.txt"
  }
}
```

20 任务配置至少修改：

```json
{
  "label": "experiment1-20-task-20x32",
  "paths": {
    "state_manifest_root": "manifests/experiment1_selected_20_20_states",
    "task_file": "tasks/experiment1_selected_20.txt"
  }
}
```

这里展示的是需要替换的字段，不是完整 JSON。其余字段从模板保留，重点确认：

```json
{
  "smoke_only": false,
  "protocol": {
    "mode": "demo_clean",
    "states_per_task": 20,
    "rollouts_per_state": 32,
    "seed": 0,
    "instruction_type": "unseen",
    "checkpoint_name": "checkpoint_step_50000.safetensors",
    "skip_get_obs_within_replan": false
  }
}
```

不要设置 `max_action_steps` 来缩短正式实验。未设置时，每个任务使用 RoboTwin `_eval_step_limit.yml` 中的原始最大 episode 步数。

## 7. 启动前检查

### 7.1 系统工具与 GPU

```bash
command -v tmux
command -v rsync
command -v ssh
nvidia-smi
```

确保目标 GPU 没有其他大显存任务。单卡 8 会话的已验证峰值约 66.1 GB，只适合显存和性能接近 80 GB H800 的卡；其他型号应重新从 1、2、4、8 会话逐级做容量测试。

当前代码的配置校验只接受 GPU 编号 0–3。单机即使有 8 张卡，也不能直接填写 4–7，除非先扩展 `run_experiment1.py` 中的 GPU 校验。

### 7.2 配置校验

以 20 任务为例：

```bash
cd /home/chw/code/packages/wm-func/exp1
python run_experiment1.py validate \
  --config config.experiment1.selected_20.json \
  --parallel-gpus 0,1,2,3
```

该命令会检查任务列表、OpenWAM 文件、RoboTwin 文件、assets、Python 路径和两个 checkpoint 的必要文件。

### 7.3 构建并固定 manifests

```bash
python run_experiment1.py build-manifests \
  --config config.experiment1.selected_20.json \
  --run-dir runs/manifest_build/selected_20 \
  --parallel-gpus 0,1,2,3
```

完成后备份：

```text
manifests/experiment1_selected_20_20_states/
```

每个任务的 `manifest.json` 应至少包含 20 个 state。双机启动器会计算每个 manifest 的 SHA256、复制到远端并逐项校验。

### 7.4 正式运行前 smoke

使用正式配置，通过命令行把规模暂时限制为 1 state、2 rollout：

```bash
tmux new-session -d -s exp1_smoke \
  "bash -lc 'cd /home/chw/code/packages/wm-func/exp1 && \
  python run_experiment1.py run \
    --config config.experiment1.selected_20.json \
    --run-dir runs/smoke/selected_20_$(date +%Y%m%d_%H%M%S) \
    --state-limit 1 --rollouts-per-state 2 \
    --parallel-gpus 0,1,2,3 > runs/selected_20_smoke_tmux.log 2>&1'"
```

Smoke 应跑完整 episode 上限，不应通过 `--max-action-steps` 截断。检查 driver error、server restart、结果条数和汇总后再启动正式实验。

## 8. 单机启动

以下示例运行 20 个任务，使用 GPU 0–3：

```bash
cd /home/chw/code/packages/wm-func/exp1
RUN_ID=selected_20_$(date +%Y%m%d_%H%M%S)
tmux new-session -d -s "$RUN_ID" \
  "bash -lc 'cd /home/chw/code/packages/wm-func/exp1 && \
  python run_experiment1.py run \
    --config config.experiment1.selected_20.json \
    --run-dir runs/experiment1/$RUN_ID \
    --parallel-gpus 0,1,2,3 > runs/${RUN_ID}.log 2>&1'"
```

10 任务只需替换配置文件。进程完全运行在 tmux 中，不依赖 Codex 会话持续存在。

查看状态：

```bash
tmux ls
tail -f runs/${RUN_ID}.log
find runs/experiment1/${RUN_ID}/openwam -name result.json | wc -l
```

## 9. 双机自动分片、运行和汇总

当前 `run_experiment1_distributed.py` 原生支持恰好两台机器。它会按 GPU 数量比例切分每个 state 的 rollout ID，两个主机各自落盘；结束后协调器把远端结果同步回主机并自动汇总。

### 9.1 SSH 前提

协调器使用 `BatchMode=yes`，必须配置免密 SSH：

```bash
ssh -o BatchMode=yes chw@REMOTE_HOST true
```

该命令失败时，启动器也会失败。密码不能由后台 tmux 交互输入。

### 9.2 启动命令

主机 4 卡、远端 3 卡的 20 任务示例：

```bash
cd /home/chw/code/packages/wm-func/exp1
python run_experiment1_distributed.py launch \
  --config config.experiment1.selected_20.json \
  --run-id selected_20_$(date +%Y%m%d_%H%M%S) \
  --remote-host chw@REMOTE_HOST \
  --remote-root /home/chw/code/packages/wm-function/exp1 \
  --local-gpus 0,1,2,3 \
  --remote-gpus 0,1,2
```

启动器依次执行：

1. 校验配置并准备主机 manifest。
2. 准备两个模型的文本 embedding cache。
3. rsync 评测代码和 manifests 到远端。
4. 校验远端 manifest 哈希和配置路径。
5. 按卡数比例切分 rollout ID。
6. 启动本地 worker、远端 worker、本地 coordinator 三个 tmux session。
7. 等待两边完成，rsync 远端原始结果，检查冲突和缺失，生成统一汇总。

32 个 rollout 在 4 卡加 3 卡布局下分为：

```text
本机：rollout_id [0, 18)
远端：rollout_id [18, 32)
```

任务、state 和两个模型在两边使用同一个 manifest；每个 rollout ID 只会由一台机器执行。

### 9.3 监控

```bash
tail -f runs/distributed/<run_id>/worker.log
tail -f runs/distributed/<run_id>/merge.log
cat runs/distributed/<run_id>/coordinator_status.json

ssh chw@REMOTE_HOST \
  'tail -f /home/chw/code/packages/wm-function/exp1/runs/distributed/<run_id>/worker.log'
```

长时间运行由 tmux 和 coordinator 管理，不依赖发起命令的终端或 Codex。

## 10. 结果目录和验收

双机最终结果以主机目录为准：

```text
runs/distributed/<run_id>/
├── config.json
├── metadata.json
├── distributed_plan.json
├── worker_local_status.json
├── coordinator_status.json
├── worker.log
├── merge.log
├── openwam/
│   ├── no_wm/raw/<task>/state_NNN/rollout_NNN/result.json
│   └── wm/raw/<task>/state_NNN/rollout_NNN/result.json
├── remote_import/openwam/
└── summary/
    ├── per_rollout.csv
    ├── per_state_by_method.csv
    ├── per_state.csv
    ├── per_task.csv
    ├── state_categories.csv
    └── summary.md
```

正式结果应满足：

- `coordinator_status.json` 的 `exit_code` 为 0。
- 本地和远端 worker 的 `exit_code` 均为 0。
- `per_rollout.csv` 行数等于任务数 × 20 × 32 × 2。
- 10 任务应有 12,800 行；20 任务应有 25,600 行。
- `(method, task, state_id, rollout_id)` 没有重复。
- `error` 列为空。
- 每个结果的 `manifest_hash` 与该任务的固定 manifest 一致。

可在主机运行快速检查：

```bash
python - <<'PY' runs/distributed/<run_id>/summary/per_rollout.csv 25600
import csv, sys
path, expected = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
keys = [(r['method'], r['task'], r['state_id'], r['rollout_id']) for r in rows]
errors = [r for r in rows if r['error'].strip()]
assert len(rows) == expected, (len(rows), expected)
assert len(keys) == len(set(keys)), 'duplicate rollout keys'
assert not errors, f'{len(errors)} rollout errors'
print({'rows': len(rows), 'unique': len(set(keys)), 'errors': len(errors)})
PY
```

10 任务时把最后的 `25600` 改成 `12800`。

## 11. 续跑与故障恢复

### 11.1 单机

保留原 `--run-dir`，使用完全相同的配置和规模参数再次执行。调度器会逐条检查 `result.json`，跳过已完成且有效的 rollout，只运行缺失或失败项：

```bash
python run_experiment1.py run \
  --config config.experiment1.selected_20.json \
  --run-dir runs/experiment1/<原 run_id> \
  --parallel-gpus 0,1,2,3
```

配置内容、任务数、state 数和 rollout 数不能与首次运行不同。

### 11.2 双机

不要重新调用 `launch` 使用同一个 run ID，因为启动器会拒绝覆盖已有目录。保留两边 run 目录和主机的 `distributed_plan.json`，按原 rollout 范围重新执行失败一侧的 `worker`。worker 内部同样只补缺失结果。两边成功后重新启动 `coordinate` 即可同步和汇总。

原始范围、GPU 和远端目录都记录在：

```text
runs/distributed/<run_id>/distributed_plan.json
```

如果只是汇总步骤失败而两个 worker 已成功，不需要重跑推理；修复 rsync、磁盘或网络问题后重新运行 coordinator。

## 12. 当前边界

- 双机启动器只支持一个主机加一个远端；三台及以上机器需要扩展分片与合并逻辑，不能同时对同一个 run 直接运行多个双机 launcher。
- 配置校验当前只允许 GPU 0–3。
- 双机要求免密 SSH、`tmux`、`rsync`，并要求两边 checkpoint、OpenWAM、RoboTwin 和 Python 路径与配置一致。
- `persistent_clients_per_gpu=8` 是 80 GB H800 上的已验证值；换 GPU 型号或同卡存在其他进程时必须重新做容量 smoke。
- 不应把 16 步容量测试结果计入正式成功率，也不应使用 `--max-action-steps` 截断正式轨迹。
- `runs/` 和 `manifests/` 默认被 Git 忽略。代码迁移不能自动带走正式 initial states，必须单独保存和传输 manifests。

## 13. 最短执行清单

1. checkout Experiment 1 `5ce986a`。
2. checkout OpenWAM `1493a2d4` 并应用运行时补丁。
3. 恢复 RoboTwin/XPolicyLab `c37109c5`、任务配置和 assets。
4. 检查两个 Python 环境与两个 checkpoint。
5. 写入选定的 10/20 任务文件并生成对应正式配置。
6. `validate`。
7. 构建、备份并在多机间校验 manifests。
8. 先跑 1 state × 2 rollout 的完整 episode smoke。
9. 用 tmux 启动单机 runner，或用双机 launcher 自动启动三个 tmux session。
10. 以 status、行数、唯一键、空 error 和 manifest hash 完成最终验收。
