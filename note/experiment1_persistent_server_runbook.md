# Experiment 1 常驻服务评测

`config.experiment1.persistent.five_tasks.json` 是五任务加速测试配置。
`hardware.persistent_server=true` 使用每 GPU 一个 OpenWAM server，按模型分相位执行。
同一 server 上的 rollout 串行运行，每条 rollout 仍在新 RoboTwin driver 进程中执行。
每次开始都通过原有 `reset_model` 清空服务端 episode，policy seed 规则和原版一致。
`workers_per_gpu` 必须为 1，以避免多个 driver 同时重置同一 server。

运行两任务的完整轨迹测试：

```bash
python run_experiment1.py run \
  --config config.experiment1.persistent.five_tasks.json \
  --task-limit 2 --state-limit 1 --rollouts-per-state 2
```

使用原 `--run-dir` 和相同参数可续跑；调度器逐条核验
`raw/<task>/state_NNN/rollout_NNN/result.json`，只重跑缺失或失败的条目。
每条结果和进度使用原子替换写入。服务启动时间保存在
`openwam/<method>/workers/gpu_N/timing.json`，driver 墙钟时间保存在各 rollout
的 `timing.json`，rollout 内部计时保存在 `result.json` 的 `timing` 字段。

旧配置 `config.experiment1.five_tasks.json` 继续使用逐 rollout 启动服务的路径，
可用于固定 manifest 和 seed 的对照。两种配置都保持完整任务步数上限，
`skip_get_obs_within_replan=false`，checkpoint 的 `--compile-enabled false`。
正式运行前必须替换正式任务清单并按 `config.experiment1.template.json`
填写路径，正式规模是每任务 10 state、每 state 每模型 128 rollout。

## 已执行的两任务 smoke

- 新 run：`runs/experiment1/20260928_080832`；命令如上，8 条完整 rollout，基础设施错误 0。
- 旧对照：`runs/experiment1/20260928_060106` 中同任务、同 state、同模型、同 rollout ID 的 8 条。
- 配对轨迹 `elapsed_sec` 合计：旧 3220.7 秒，新 1418.2 秒；比值 2.27。
  这是轨迹延迟的比较，旧 run 使用每 GPU 3 个 worker，新 run 使用每 GPU 1 个 worker；
  不应单独解释为模型前向的加速倍数。
- 8/8 条的成功标记一致，5/8 条的终止步数一致。三条 wm 成功轨迹的
  终止步数相差 1–7 步，需要在正式评测前判断是否是运行时非确定性。
- 新 run 的服务 ready 时间每实例约 88–102 秒；单轨迹 driver 额外墙钟开销
  在完成的 no_wm 四条中约 10–16 秒。按本次计时，批量复用 driver 的优先级低于
  保持服务常驻和后续取图协议优化。

## 单 GPU 连续 rollout 检查

`config.experiment1.persistent.one_gpu.json` 仅使用 GPU 0。运行：

```bash
python run_experiment1.py run \
  --config config.experiment1.persistent.one_gpu.json \
  --method wm --task-limit 1 --state-limit 1 --rollouts-per-state 2
```

`runs/experiment1/20260928_082238` 中只生成一个 server 日志，
`worker_timing.json` 记录 `jobs=2, restarts=0`，ready 约 58 秒，
两条 `place_fan` 轨迹均成功，seed 为 200000、200001，终止步数均为 146。
同 seed 的旧 run 终止步数为 153、152；新路径两次独立执行的
`place_fan / wm / rollout_000` 都是 146。这提示旧/新并发条件下存在
运行时轨迹差异，尚无逐步动作日志，不能归因于模型或仿真数值变化。

用旧的逐 rollout 调度器单独重跑 `place_fan / wm / rollout_000`
（`runs/experiment1/20260928_082858`）也得到成功、146 步。
因此现有证据没有显示常驻服务本身改变了这条轨迹；早先 153 步的记录
是在每 GPU 最多 3 个 worker 的高并发运行中得到的。仍需动作级记录
才能确认差异产生的具体位置。
