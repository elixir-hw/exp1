# 实验 1 推理与评测加速实施规划（FastWAM 对照）

> 本文是给下一轮独立上下文使用的实施手册。本文创建时**只做分析和规划，未修改评测代码、模型代码或既有结果**。执行前先核查代码和运行环境是否已经变化，不要机械覆盖用户后续修改。

## 0. 要解决的问题与不可改变的实验语义

实验 1 比较两个 OpenWAM 50k checkpoint 在 RoboTwin 同一固定初始 state 上的成功概率：

| 方法 | 训练设置 | checkpoint 目录 |
|---|---|---|
| `no_wm` | action-only，不启 video loss | `/mnt/data/chw/model/openwam_train_runs/robotwin_clean40_action_only_no_wm_50k_8gpu_bs2_acc2_gbs32_20260926_055033/2026-09-26_05-51-27` |
| `wm` | 完整 WM，启 video loss | `/mnt/data/chw/model/openwam_train_runs/robotwin_clean40_wm_full_50k_8gpu_bs2_acc2_gbs32_20260926_055033/2026-09-26_05-51-23` |

两者评测文件名均为 `checkpoint_step_50000.safetensors`。正式实验以用户最新口径为准：**每任务 10 个固定 state，每 state 对每模型 128 条随机轨迹**，即每任务 `10 * 128 * 2 = 2,560` 条；若最终选 10 个任务，共 25,600 条。任务清单仍以用户之后提供的正式名单为准，不要把五任务测试名单当作最终名单。保持 `demo_clean`、`instruction_type=unseen`、相同 manifest/state/instruction 和可审计的不同 policy seed；两模型在同一 state、同一 rollout ID 上配对。成功定义和正常任务步数上限不变，不能靠提前截断失败轨迹来声称同一成功概率。

已完成五任务诊断运行：`exp1/runs/experiment1/20260928_060106`，每任务 1 state、每模型 4 条，共 40 条，基础设施错误 0；结果解读在 `exp1/note/experiment1_selected_5_parallel_result.md`。40 条 rollout 的 `elapsed_sec` 合计约 9.34 小时，四卡并行整轮约 65 分钟。失败轨迹通常跑满 400/600 步，`put_bottles_dustbin` 跑满 1700 步。**这里没有失败重试**，主要是长轨迹和重复启动。一个典型 server 日志从 `06:01:16` 开始加载到 `06:04:53` 监听端口，约 217 秒；这只是该日志的观测值，不能当成所有作业的平均值。

重要的配置债务：`exp1/config.experiment1.template.json` 和 `exp1/note/experiment1_openwam_state_success_eval_plan.md` 目前仍写 `20 states * 32 rollouts`；正式运行前需要另行更正并验算规模。本加速方案不以旧模板数字为准，也不修改既有五任务记录。

## 1. 必须阅读的代码位置

### 本项目：`/home/chw/code/packages/wm-func`

- `exp1/run_experiment1.py`：配置验证、manifest 准备、server 启停、四卡 job 调度、汇总。重点看 `start_server`、`run_parallel_methods`、`summarize` 和 CLI。当前 `parallel_by_rollout=true` 时，每条 rollout 是一个 job，独占启动一个模型 server 和 RoboTwin driver。
- `exp1/wm_eval/robotwin_state_probe.py`：`prepare_task`、`run_one_rollout`、`run_state_rollouts`。每条轨迹重设模型/seed、`setup_demo`、循环 `get_obs -> eval_func`，结束 `safe_close(task_env)`；当前批量路径虽能在一进程中循环多条，但环境生命周期必须重新验证。
- `exp1/config.experiment1.five_tasks.json`：现有五任务配置，`workers_per_gpu=3`、`parallel_by_rollout=true`、`skip_get_obs_within_replan=false`，两模型的 `deploy_args` 含 `--compile-enabled false`。
- `exp1/runs/experiment1/20260928_060106/summary/per_rollout.csv` 与各 rollout 的 `server.log`、`driver.log`：性能基线。不要修改或覆盖这个 run 目录。
- `exp1/wm_eval/ensure_text_cache.py`：文本 embedding 预备逻辑；保持在主运行前执行。

### FastWAM：`/home/chw/code/packages/FastWAM`

- `configs/sim_robotwin.yaml`：评测默认 `replan_steps: 24`、`skip_get_obs_within_replan: true`、`num_inference_steps: 10`、`MULTIRUN.max_tasks_per_gpu: 2`、`instruction_type: unseen`。
- `experiments/robotwin/run_robotwin_manager.py`：GPU slot 调度，一次启动一个 task/phase 进程；它不是逐 rollout 启动进程。借鉴其有界并发与 task 粒度，但不要照搬输出协议。
- `experiments/robotwin/eval_robotwin_single.py`：把配置传给 RoboTwin eval 进程；一个任务进程连续运行多个 episode。
- `experiments/robotwin/fastwam_policy/deploy_policy.py`：`pending_actions` 队列、`_fill_action_queue`、`should_request_observation`、`step`；只在队列空时重新取图/推理，其他动作直接执行。还提供 `timing_enabled` 的 `infer_s`/`sim_s` 统计。
- `third_party/RoboTwin/script/eval_policy.py`：`skip_get_obs_within_replan` 的实际调用位置。不要误以为仅改 YAML 就能使任意策略跳过取图。

### OpenWAM 与 RoboTwin：独立代码库

- `/home/chw/code/packages/OpenWAM/OpenWAM/benchmarks/robotwin/openwam2robotwin_interface.py`：`ModelClient.step` 当前每一步将三路 RGB 编码并经 WebSocket 发给 server；模块级 `eval` 要求非空 observation。已有 `set_sampling_seed`，seed 由 `policy_seed * 10000 + request_index` 生成。
- `/home/chw/code/packages/OpenWAM/OpenWAM/openwam/deploy/executors/sync_executor.py`：已有 action chunk 缓冲；`inference_horizon=null` 时消耗完整 chunk。**不用重做模型 action 队列**，应复用/暴露其剩余动作状态。
- `/home/chw/code/packages/OpenWAM/OpenWAM/openwam/deploy/server.py`：OBS/RESET/PING WebSocket 协议；当前 OBS 必经图片校验、预处理，即使下一步动作已在缓冲区也会处理图像。
- `/home/chw/code/packages/OpenWAM/OpenWAM/openwam/deploy/policy.py` 与 `openwam/deploy/engine.py`：推理调度；已 `torch.no_grad()`，默认 10 去噪步。
- `/home/chw/code/packages/OpenWAM/OpenWAM/configs/deploy.yaml`：已启用文本 prompt cache、DiT velocity cache、关闭视频解码；`torch.compile` 默认开，但五任务配置显式关。当前训练配置 `dataloader.num_frames=33`，默认每次生成 32 个动作（需在运行时再次核实实际执行 horizon）。
- `/home/chw/code/packages/RoboTwin/XPolicyLab/policy/FastWAM/FastWAM/third_party/RoboTwin/envs/_base_task.py`：`get_obs()` 会更新相机图片；`take_action()` 自身仍调用 `_update_render()`。因此跳过 `get_obs()` 是减少相机图片生成/读回/编码，而**不是完全跳过 SAPIEN 渲染或物理仿真**。

## 2. 加速机会与优先级

| 优先级 | 机会 | 当前情况 | 对结果的风险 |
|---|---|---|---|
| P0 | 量化每阶段耗时 | 只有整条 `elapsed_sec`；缺少启动、取图、模型生成、仿真分解 | 计时本身应低开销、无语义变化 |
| P1 | 模型 server 常驻，多个 rollout 共用 | 每个 rollout 重新加载 checkpoint；正式规模会放大固定开销 | reset 隔离、seed、故障恢复、GPU 内存 |
| P2 | action chunk 中间不调用 `get_obs`、不传图 | OpenWAM server 有缓冲，但 client 不知道何时需要新观测 | 协议边界、replan 时机、视频帧稀疏 |
| P3 | 批量复用 RoboTwin driver | 每条 rollout 新进程；现有 `run_state_rollouts` 已有循环框架 | 环境 close/setup 生命周期和状态污染 |
| P4 | `torch.compile`、改变去噪步数/执行 horizon、async | 尚未做受控 A/B；可能改采样轨迹或成功率 | 属于新的评测设置，须另报指标 |

FastWAM 的 `sigma_shift=5.0` 是采样参数，不是通用加速开关。`instruction_type=seen` 可能改变分数，不能用于加速且不可与已有 unseen 结果混算。当前已有 bf16、文本 embedding cache、DiT cache、`decode_video=false`，不要重复实现这些能力。提高 GPU 并发也不是无条件更快：服务加载争抢 I/O/显存、SAPIEN 和推理抢同一 GPU，必须以总吞吐及峰值显存判断。

## 3. 分阶段实施与验收

### Stage 0：冻结协议、基线与规模（先于代码改动）

1. 检查 git 工作树和运行中进程；记录 OpenWAM、RoboTwin、wm-func 版本/未提交修改，不覆盖用户工作。
2. 从五任务现有结果统计按任务/模型的 `terminal_step`、`elapsed_sec`、成功率、长尾；以新 run 目录跑一小组基线（建议 Fan、Bottles、Mouse Pad，各 1 state、每模型至少 2 条，固定原 manifest/seed），留存计时和结果。基线不需要重新跑完整 40 条。
3. 把正式规模更新进独立正式配置与旧规划文档，**不要**改五任务历史配置和 run 结果。确定最终任务清单后才构造正式 manifest。确认正式运行依旧仅使用 GPU 0–3。

**检查点 G0**：写出一张明确的运行协议表（checkpoint、manifest hash、instruction、mode、seed 规则、rollout 数、step limit、是否保存视频）；基线结果和运行命令可复现，正式规模验算为每任务 2,560 条。没有最终任务清单不阻碍加速 smoke，但不能启动正式实验。

### Stage 1：低开销计时与瓶颈定位

1. 在 `run_experiment1.py` 对 server 启动/模型 ready、driver 启动、job 结束加 wall-clock 计时，输出结构化 JSON/CSV；单调时钟用于耗时。
2. 在 `robotwin_state_probe.py` 分别累计 `setup_demo`、`get_obs`、`eval_func`、`take_action`（如需后两者分开，在 OpenWAM callback 里计时）和 teardown。每条 rollout 保存 `obs_count`、`model_generate_count`、`action_steps`、总耗时；避免每步打印。CUDA 计时需显式同步或用 CUDA event，不能把异步提交时间误称 GPU 执行时间。
3. server 侧统计 OBS 预处理、buffer hit、真正 `engine.generate` 的次数及耗时；记录每次产生/执行的 chunk 长度。客户端/服务端统计口径要对齐。
4. 保持现有结果字段与 `summarize` 兼容；新性能字段是附加信息，历史 run 仍可汇总。

**检查点 G1**：同一固定 seed 的 2–4 条测试在成功标记、terminal step、动作序列（能记录时）上与未插桩基线一致；分项时间总和与 wall-clock 差额可解释；无新增 infra failure。基于真实数据决定 P1/P2 的收益预期，不宣称尚未测得的加速倍数。

### Stage 2：常驻模型服务，消除逐轨迹 checkpoint 加载

1. 在 `run_experiment1.py` 把 `run_parallel_methods` 的 job 粒度从单 rollout 改成**可配置的 rollout shard**，例如同一 `(method, task, state)` 的连续 8/16 个 rollout ID。不要删除旧模式，保留开关便于 A/B、回退。
2. 增加每 GPU/每 method 的常驻 worker 生命周期。先采用保守方案：每 GPU 一个模型 server，串行处理该 server 上的 shard；四卡同时处理四个 shard。若想同 GPU 多 server，须先测显存与总吞吐。每次切换模型加载一次，不要在每条 rollout 后加载；可按 method 分相位或排队，保证两个模型资源设置一致。
3. 服务端已有 RESET；每条 rollout 调用 reset 清空 action buffer、request 计数和 episode 状态；client 仍使用原 policy seed。不能让前一条轨迹的缓冲动作、prompt 或随机数状态进入下一条。
4. 设计故障边界：服务崩溃时只重启受影响 worker/shard；已成功落盘的 rollout 不重跑。不要在服务处理请求期间让另一个 rollout 调用全局 RESET；若共享 server，仅允许串行会话，除非另外设计 per-session policy 状态。
5. 保持原 `raw/<task>/state_xxx/rollout_xxx/result.json` 等结果 contract，逐条原子写入、进度逐条更新；聚合时根据 rollout ID 查漏，不能仅看 shard 级 `result.json`。运行目录 metadata 记录 worker/shard 参数、版本和异常重启次数。

**检查点 G2**：至少 2 个任务、2 个 state、每模型 4 条的 smoke 中，每 GPU 同时加载的模型数符合配置；server ready 次数从逐轨迹下降为每 worker/模型一次；无 seed 重复、无漏号/重号、无跨模型结果串线。人为杀掉一个 worker 后可续跑缺失轨迹，已完成结果字节不变；`summarize` 与旧结构兼容；同设置成功/步数不因调度顺序漂移。记录 `rollouts/hour`、p50/p95 `elapsed_sec`、峰值显存，比较旧模式。

### Stage 3：OpenWAM action chunk 内免取图（参考 FastWAM）

1. 先以**同步 executor**实现，保持原 full-OBS 协议可用。让 `SyncInferenceExecutor`/`WAMPolicy` 只读地暴露 `buffer_remaining` 或 `needs_observation`，不要让外部直接篡改 deque。
2. server 的 OBS 响应附带“本动作返回后剩余可消费动作数”；新增轻量 `NEXT_ACTION`（名称可调整）请求，仅在 server buffer 非空时有效，不调用图像校验/预处理，也不调用 `engine.generate`。若 buffer 空则返回明确协议错误，绝不悄悄复用旧图像重规划。RESET 必须使剩余数归零。async executor 若不能准确给出同等语义，先禁用此优化并提示，不要假装支持。
3. `ModelClient.should_request_observation()` 根据上次响应的剩余数返回布尔值。`step` 支持无 observation：仅在剩余数大于 0 时请求 `NEXT_ACTION`；需新规划时必须带新 RGB 和 proprio。模块级 `eval(TASK_ENV, model, observation)` 在无 observation 分支直接取缓存动作并 `take_action`，不要读取图像/关节观测。保留 action 20D 到 16D 转换、二值动作投影和任务成功检查。
4. `robotwin_state_probe.py` 现有 `skip_get_obs_within_replan` 条件才能安全打开；默认仍关，待验收后在新配置中打开。服务端 seed 仍按**每步请求计数**前进，与旧协议对齐；真正生成 chunk 的请求应使用相同 seed。若原 `_step` 的语义改变，必须显式保留旧计数规则。
5. 不改变 `inference_horizon`、去噪步数和评测视频开关。当前评测已 `eval_video_log=false`；若以后需要逐步视频，免取图会让保存视频稀疏，应关闭该优化重新采集。`take_action()` 自身仍 `_update_render()`，所以把节省描述为取图/读回/编码/传输，而非零渲染。

**检查点 G3**：协议单测覆盖首次 OBS、连续 NEXT_ACTION、刚好耗尽 chunk、下一次必须 OBS、RESET 后拒绝 NEXT_ACTION、错误请求和跨 rollout 隔离。端到端在相同 state/seed 上比较开关前后每步动作与 terminal step（允许记录并设置合理浮点容差），成功结果一致；`obs_count` 从每步一次降到约每 chunk 一次，`generate_count` 不变，NEXT_ACTION 不触发图像预处理。测量真实 wall-clock 提升和视频行为，不能仅凭请求次数估算。

### Stage 4：可选的 RoboTwin driver 批量复用

1. 在 Stage 2 已有常驻 server 的基础上，考虑让一个 RoboTwin driver 对一个 `(task,state,method)` shard 连续评测多条轨迹，减少 Python/CUDA/SAPIEN 导入开销。先看 Stage 1/2 数据；若 driver 启动开销小，则可以不做。
2. 当前 `run_state_rollouts` 有多 rollout 循环，但 `run_one_rollout` 每次 `finally: safe_close(task_env)`；必须验证 `setup_demo` 是否可在同一对象 close 后再次安全运行。保守实现是每条轨迹新建 task env 对象、复用已加载的 client/模型连接和 driver 进程；只在证实安全后才复用仿真对象。固定 state 的 accepted seed、`now_ep_num`、instruction 每条都重新设置。
3. 每条轨迹即时原子写盘，driver 崩溃后按 rollout ID 补跑；不能等整个 128-rollout shard 完成才写一个大文件。监测 RSS/显存增长，每个 shard 后可主动重启 driver 以控制资源。

**检查点 G4**：同 state 连续至少 8 条，核对重复初始 state 的观测/几何一致，policy seed 全部不同，结果与单条进程模式可配对；第二条及后续无环境已关闭错误、资源泄漏或随机状态污染。人为中断后仅补未完成条目；吞吐改善超过新增复杂度，否则保留 Stage 2 方案。

### Stage 5：可改变推理设置的受控实验（不混入主评测）

1. `torch.compile`：现有五任务配置显式 `--compile-enabled false`。使用**常驻服务**后再测开启，因为编译 warmup 才有机会摊销；比较冷启动、稳定吞吐、峰值显存与数值差异。若编译不稳定/吞吐下降，继续关闭。
2. 去噪步数、`inference_horizon`、async executor：这些可能改变动作与成功概率，作为新 `inference_setting` 独立运行。不能把更少步数或更长 open-loop 的成绩直接并入旧设置，也不能单独给 `wm` 使用。
3. 调整每 GPU worker 数：用实际 GPU 利用率、显存和任务吞吐选择，先保证四卡 0–3 内无 OOM/争抢。FastWAM 的 `max_tasks_per_gpu=2` 是其模型/设备环境的选择，不是本模型的安全默认值。

**检查点 G5**：每个变体都保存完整配置、模型/代码版本、耗时和固定 seed 的成功结果；明确标为“新推理设置”，两个模型在同一设置下比较。只有在成功率不退化且收益稳定时才考虑正式实验采用，并从头重跑可比结果。

## 4. 交付标准与执行顺序

推荐顺序为 `G0 -> G1 -> G2 -> G3`，然后依据计时决定是否做 G4/G5。每个阶段一个独立小改动和新 smoke run，记录命令、配置快照、吞吐、成功数、terminal step、错误数、峰值显存；前一关未过不启动下一关或正式 25,600 条。运行中若看到与计划不一致的用户修改，先读懂并合作保留，不重置工作树。

最终交付应有：可切换的旧/新路径；单测覆盖协议与恢复；固定 state/seed 的端到端一致性记录；四卡 0–3 的吞吐与资源对比；详细运行手册及正式配置。**核心验收是“同一评测语义下更高的有效 rollout 吞吐”，而不是单次模型前向看起来更快。**
