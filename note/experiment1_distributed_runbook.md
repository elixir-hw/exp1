# Experiment 1 双机四任务运行

本机使用 GPU 0–3，`geekplus-h800-141` 使用 GPU 0–2。141 的 GPU 3
已由其他作业占用。两台机器各自把逐条结果保存在自己的 `runs/distributed/<run_id>`，
本机另开一个 tmux 协调进程，在两边完成后复制远端结果并自动生成汇总。

## 分片规则

每个任务使用同一份 `demo_clean` state manifest、`unseen` 指令，
两个模型按相同 `(task, state_id, rollout_id)` 配对。一个 state 的 rollout ID
按 GPU 数量比例分给两台机器：smoke 的 2 次为本机 `[0,1)`、141 `[1,2)`；
正式的 32 次为本机 `[0,18)`、141 `[18,32)`。每个 rollout ID 仅在一台机器
运行，两边的每 GPU worker 都串行使用一个常驻模型服务。两个模型依次运行，
每张参与的卡一次只加载当前模型，显存占用约 20 GB 属于预期。每条轨迹仍完整执行到
成功或 RoboTwin 原有步数上限；不保存视频，也不开启取图跳过优化。

## 配置与启动

- `config.experiment1.distributed.smoke.json`：四任务、1 state、每模型每 state 2 条。
- `config.experiment1.distributed.four_tasks.json`：四任务、20 state、每模型每 state 32 条。

从本机 `exp1` 目录运行一次：

```bash
python run_experiment1_distributed.py launch \
  --config config.experiment1.distributed.smoke.json \
  --run-id four_task_smoke_20260928_0901
```

启动器会在本机检查/准备 manifest 和文本缓存，将代码及 manifest 复制到 141，
验证远端配置，然后启动本机 worker、远端 worker、本机合并器三个 tmux session。
141 的代码路径为 `/home/chw/code/packages/wm-function/exp1`。OpenWAM 代码在
141 的 `/home/chw/code/packages/OpenWAM/OpenWAM`，评测 Python 环境在
`/home/chw/miniconda3/envs/openwam`；两个 checkpoint 从共享 `/mnt/data` 读取。

查看进度：

```bash
tail -f runs/distributed/<run_id>/worker.log
tail -f runs/distributed/<run_id>/merge.log
cat runs/distributed/<run_id>/coordinator_status.json
ssh chw@10.11.141.54 'tail -f /home/chw/code/packages/wm-function/exp1/runs/distributed/<run_id>/worker.log'
```

本机的 `remote_import/openwam` 保存远端传回的原始结果和日志，合并后的规范结果
位于本机 `openwam/<method>/raw/<task>/state_NNN/result.json`；
`summary/` 下自动生成 `per_rollout.csv`、`per_state_by_method.csv`、
`per_state.csv`、`per_task.csv`、`state_categories.csv` 和 `summary.md`。
`coordinator_status.json` 的 `exit_code=0` 表示两边成功完成且合并通过。
远端的原始 rollout 目录留在 `remote_import/openwam`；规范结果目录仅复制
结果 JSON 和日志，跳过仿真 `runtime`，避免展开指向大体积资产的软链接。

## 并行对照与 smoke 验收

`four_task_smoke_20260928_0901` 在两边各完成 8 条 rollout，合并后
`per_rollout.csv` 共 16 条，键均唯一、无错误，`coordinator_status.json`
为 `exit_code=0`。本机四张卡、141 三张卡均参与了分配；141 的 GPU 0
分到两条轨迹。这个 smoke 每模型每机器只有四条轨迹，不足以测量正式评测的持续吞吐。

同卡双服务 Fan/WM 对照中，单卡顺序处理两条的 worker 墙钟时间为 281.8 秒，
两个服务并发处理各一条为 277.4 秒，约快 1.6%。两条并发轨迹的观测获取
各约 101 秒，顺序基线各约 45 秒，资源争用明显。正式配置因此保持
`persistent_servers_per_gpu=1`；显存余量本身不能预测吞吐提升。

正式启动前，应先完成 20 state manifest 的构建和配对检查。正式配置使用独立的
`manifests/experiment1_distributed_4_20_states`，避免覆盖早期单 state 记录。
smoke 验收完成前不启动正式 5,120 条评测。
