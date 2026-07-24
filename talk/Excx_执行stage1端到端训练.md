# AdaLigand Stage1 正式端到端数据准备与训练

本 ExecPlan 是持续维护的执行日志。`Progress`、`Surprises & Discoveries`、`Decision Log`、`Outcomes & Retrospective` 与 `Plan Drift / Reconciliation` 必须随执行事实更新。它用于在客户端或网络中断后，仅依靠当前工作树、本文和服务器状态恢复任务。

## Purpose / Big Picture

本任务把已经验收的 AdaLigand A–G 全量产物转化为可审计的 Stage1 正式训练数据：先生成严格质量过滤清单和必要时的资产齐全回退清单，按唯一 PDB 确定性划分四个 split，再为 train/validation 发布 center、bias、context 三类 BOX pool，最后用同一 global batch 完成 unet_c1、Find_0 CPC1→CPC2 与新版 Find_1 CPC1→CPC2 五段训练。Find_2 的实现和短训练证据保留，但用户根据伪/真实原子密度初始化的科学判断于 2026-07-21 明确撤回其正式训练目标。完成时必须保留可加载 BEST、resolved config、日志、W&B 状态、初始化谱系、停止原因和全部数据/BOX manifest。

## Upstream Documents and Scope

本文实施下列上游文档，不替代或静默改写它们：

- `../AdaLigand/文档/规划文档/Stage1训练实现计划.md`：Stage1 数据、BOX pool、训练阶段和验收的主要计划。
- `../AdaLigand/文档/讨论/BOX-level数据契约.md`：BOX-level 数据语义和发布契约。
- `../AdaLigand/文档/exec_plan/Stage1代码实现.md`：成熟 Stage1 代码实现与验证记录。
- `../AdaLigand/文档/exec_plan/A-G数据流水线实现与全量运行.md`：正式 run `adaligand_ag_20260711T154658` 的产物与验收依据。
- `../AdaLigand/grill_with_memory/07-21-01-11.md`：本次正式执行的近期需求澄清参考。
- `../AdaLigand/grill_with_memory/07-21-12-07.md`：Find_2 架构、正式 producer 身份、单 H200 训练、Job 321108 替换和 subagent 分工的逐项确认决定。
- `src/datasets/README_STAGE1.md`、`configs/dataset/`、`configs/train/`、`configs/loss/` 与实际代码：Pocket_Plus 当前 Stage1 运行契约；代码和真实产物优先于旧说明。

覆盖范围是最终 keep-list、资产检查、四个 split、train/validation 三类 BOX pool、Dataset/DataLoader smoke、共同 batch 冻结、Find_2 最小科学实现与正式 producer 枚举适配证据，以及 unet_c1/Find_0/Find_1 五段训练的提交、监控、恢复和终态验收。Find_2 不再训练，既有实现暂不回滚；Stage2/Stage3、下游算法重构及 A–G 数据重跑不在范围内。除用户在 Find_2 grill 中明确授权的最小科学改动和已有真实阻断修复外，不修改成熟 Stage1 行为。

## Progress

- [x] (2026-07-21 02:28+08:00) 接管持续目标；读取项目 `AGENTS.md` 和 ExecPlan、计划治理、Pocket_Plus、服务器交互、handoff 技能规则。
- [x] (2026-07-21 02:44+08:00) 完成 AdaLigand/Pocket_Plus 规定材料、代码入口、最新 handoff、正式 A–G 产物和服务器资源/作业恢复盘点。
- [x] (2026-07-21 03:09+08:00) 生成严格过滤清单、简单资产检查、最终采用清单和回退说明：严格 20,483 PDB/637,140 occurrence；18,293 PDB/579,688 occurrence 的 Stage1 直接资产完整；2,190 PDB 全部因 exp/sim 几何不一致被记录；最终仍高于 18k，未使用全量回退。
- [x] (2026-07-21 03:09+08:00) 按唯一 PDB 生成 train/validation/calibration/held-out 四个确定性 split：13,719/200/100/4,274 PDB，validation/calibration 选择扫描 300 个即全部满足 80³。
- [x] (2026-07-21 04:01+08:00) 为 train/validation 生成并正式发布 center、bias、context BOX pool 与 manifest/completion；Job 321111 `COMPLETED/0:0`，后续优化验收 Job 321141 `COMPLETED/0:0`。
- [x] (2026-07-21 04:35+08:00) 用正式清单和正式 BOX pool 完成三类 BOX 的 Dataset/DataLoader batch smoke。
- [x] (2026-07-21 05:18+08:00) 在实际 GPU 上完成显存试跑并冻结三个 producer 共用的 global batch、各自 micro-batch/梯度累积、卡型和卡数。
- [ ] 提交并监控 unet_c1 到第 4 次实质学习率下降或 20 epoch 硬上限，保留可加载 BEST。
- [ ] 提交并监控新版 Find_1 CPC1→CPC2，验证 Gaussian embed scatter、双 H100 运行及 CPC2 从同名 CPC1 BEST 严格 model-only 初始化，保留两段 checkpoint（Job 321540 已直接启动正式 CPC1）。
- [ ] 提交并监控 Find_0 CPC1→CPC2，验证 CPC2 从同名 CPC1 BEST 严格 model-only 初始化，保留两段 checkpoint（Job 321743 双 H200 micro8/accum3 正式 CPC1 W&B `w21l8dof` online，至少 step20）。
- [x] (2026-07-21 18:06+08:00) 按用户科学判断撤回 Find_2 正式训练目标；保留实现、五步 smoke、W&B run `30dq7bql` 和截至 global step 53 的日志，原双 H100 allocation 原地转换为 Find_1。
- [ ] 审计五段 resolved config、每 epoch 30 次 validation、停止原因、W&B、Slurm/本地日志、Job ID 与终态。
- [ ] 最终收口临时脚手架分类、风险、产物路径和 handoff；仅在全部训练完成或用户明确指示后释放 allocation。

### Milestone checkpoint — 2026-07-21 04:35+08:00

- [x] BOX pool 正式发布并完成全量生产读取验收。Job 321111 `COMPLETED/0:0`，发布 train 13,715 PDB、validation 200 PDB；优化后的验收 Job 321141 `COMPLETED/0:0`，耗时 00:01:39、16 CPU、MaxRSS 102,648 K。
- [x] `box_pool/verification.json` 状态为 `ok`：train 437,332 occurrence、5,635,991 context candidates、zero-context PDB 2、underfilled 0；validation 5,942 occurrence、84,229 context candidates、zero/underfilled 均为 0；固定 selection 恰好 29,529 条，center/bias/context 为 3,281/16,405/9,843。
- [x] 三个 producer 都已用正式 keep-list 与正式 BOX pool 完成真实 Dataset/DataLoader 读取 smoke；该 smoke 专门以 `batch_size=3` 各取一个 center、bias、context，因此 Find 输入为 `[3,56,80,80,80]` 并含原子字段，unet 输入为 `[3,1,80,80,80]`。这里的 3 不是训练 micro-batch；正式训练另行固定为 micro-batch 6。
- [x] 实际 GPU 显存试跑第一次派发在 Hydra 配置组合阶段失败：run-scoped 包装把训练配置中不存在的 `max_steps`、`limit_*` 和 `num_sanity_val_steps` 当作已有字段覆盖。三台 allocation 的命令均以 exit 1 返回并重新停在 `try_lock`；没有启动正式训练、没有释放 `after_lock`。已把四个新增字段改为 `+train.*`，同时删除正式脚本中不必要的 `train.max_steps=-1` 覆盖，经服务器 `bash -n`、非删除式同步及后续成功试跑证明恢复有效。

### Batch freeze checkpoint — 2026-07-21 05:18+08:00

- [x] 三个 producer 均以正式 Dataset/DataLoader、真实前向/反向和恰好 2 个 optimizer steps 完成显存试跑，共同 `global_batch_size=48` 已冻结。
- [x] Find_1：Job 321108，2×NVIDIA H200 NVL，micro-batch 6，gradient accumulation 4；两卡峰值 67.811%/68.773%，GPU utilization 均到 100%，低于 Find 85% 门限。
- [x] Find_0：Job 321106，1×NVIDIA H100 PCIe，micro-batch 4，gradient accumulation 12；峰值 78.998%，GPU utilization 100%，低于 Find 85% 门限。
- [x] unet_c1：Job 321107，1×NVIDIA H100 PCIe，micro-batch 6，gradient accumulation 8；峰值 80.006%，GPU utilization 100%，低于 unet 95% 门限。
- [x] 真实试跑故障已恢复且证据保留：Hydra 新字段语法失败；`expandable_segments` 在当前节点/torch 上触发 allocator internal assert；Find CPC1 的 2-step 调试使自动 warmup 舍入为 0；双 H200 Find_1 的成熟配置关闭 unused-parameter detection 后在第 2 个 DDP step 报未用参数；较大 micro-batch 的 OOM/超门限结果均留在各 allocation `memory_trial/`。
- [x] 恢复均限制在 `tmp/` 运行包装：显式采用兼容的 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`；2-step trial 显式设 `warmup_steps=2`；仅双卡运行覆盖 `train.ddp_find_unused_parameters=true`。Pocket_Plus 科学源码保持 commit `de6a89f38e48553edfd440c76bf61823229d9205`，未修改模型、loss、Dataset 或默认生产入口。
- [x] 最终 formal launcher 门禁与 `batch_freeze.env` 已发布，三条正式训练链于约 05:18+08:00 在原 allocation 内启动；三个 allocation 均保留 `after_lock`。Find_0/unet_c1 已 W&B online 初始化并进入 Trainer，Find_1 正在双 rank 冷初始化窗口。

### Formal launch checkpoint — 2026-07-21 05:24+08:00

- [x] 三个 producer 的正式命令全部由无时限 sbatch allocation 承载：Find_1 Job 321108（hnode03，2×H200 NVL）、Find_0 Job 321106（hnode01，1×H100 PCIe）、unet_c1 Job 321107（hnode02，1×H100 PCIe）；Slurm 均为 `RUNNING`，`after_lock` 均保留。
- [x] 已核对首段 resolved config：共同 `global_batch_size=48`；Find_1 micro=6、DDP unused detection=true、`init_from=null`；Find_0 micro=4、DDP unused detection=false、`init_from=null`；unet_c1 micro=6、DDP unused detection=false、`init_from=null`；三者均 `offline=false`。
- [x] Find_0 CPC1 与 unet_c1 已分别写出 `wandb_status=online_initialized`，W&B run 为 `s04troim` 与 `ohhl54t1`；两者日志均确认 `val_per_epoch=25`、`val_check_interval=0.04`、`stop_after_lr_reductions=4`、effective global batch 48。
- [x] Find_1 CPC1 rank 0 在约 25 分钟、累计约 53.4 GB pool 读取后完成 lazy-module/DataModule 冷初始化，并于 05:43+08:00 写出 `wandb_status=online_initialized`；W&B run 为 `c641mktn`。当前 rank 1 正在重复建立自己的正式 DataModule 后加入 DDP，双 H200 尚未装载模型，继续高频监控到 2/2 rendezvous 与首个 optimizer step。
- [ ] 首个完整正式 epoch 后，按本地日志和 W&B 历史分别审计恰好 30 次 validation；每个阶段完成后审计停止原因、BEST 可加载性和在线 W&B 完整性。

### Find_0 memory recovery checkpoint — 2026-07-21 05:57+08:00

- [x] Find_0 CPC1 micro=4 的短试跑峰值是 78.998%，但正式长跑在变长真实样本上于 05:48+08:00 升到 79,494/81,559 MiB，即 97.468%，并持续至少约 90 秒；没有等到 OOM 才处理。原在线 W&B run `s04troim`、完整 `nvidia_smi_1s.csv`、`gpu_peak_unsafe.tsv` 与 `unsafe_memory_recovery.env` 均保留在 `formal/Find_0_CPC1/launch_1/`。
- [x] 仅对自有 Job 321106 写入精确 `kill_lock`；allocation 返回 exit 143 并恢复 `try_lock`，`after_lock` 未删除、Slurm job 未取消。由于内层训练用了 `setsid`，PID/PGID 7235 成为孤儿；核对完整命令行为本次 `Find_0 CPC1`、micro=4 后，先 TERM 再 KILL 该精确进程组，确认 H100 回到 1 MiB/0%。
- [x] run-scoped 正式包装已增加 TERM 转发，并允许通过 `FORMAL_RUN_TAG` 生成唯一恢复 run stamp；科学源码、Dataset、模型、loss 和默认生产入口未修改。改动已通过远端 `bash -n` 和非删除式安全同步。
- [x] Find_0 在同一 Job 321106 内完成 micro=2、GBS 48、accumulation 24 的扩大试跑：实际 120/160 micro-batches，`max_steps=5 reached`，exit 0，峰值 44.179%，GPU utilization 100%。`batch_freeze.env` 已更新，旧 micro=4 正式目录归档为 `formal/Find_0_CPC1_unsafe_m4_20260721T0548/`。
- [x] Find_0 正式恢复以唯一 tag `formal2_m2` 在原 Job 321106 内派发，`offline=false`；正式 pool/lazy-module 初始化完成，W&B online run `pzfktt9t` 已推进到至少 `trainer/global_step=5`。日志确认 GBS48/micro2/accum24、`val_per_epoch=25`、`val_check_interval=0.04`、CPC1 stop-after-4；正式峰值当前 48.564%，GPU utilization 100%，继续监控长期 allocator/样本峰值。

### Find_1 distributed validation recovery checkpoint — 2026-07-21 06:10+08:00

- [x] Find_1 CPC1 首次正式 run `c641mktn` 完成双 rank NCCL rendezvous 和 2/2 sanity validation batch 的前向，但在首次全局 AP 汇总时失败：TorchMetrics 的 `BinaryAveragePrecision(compute_on_cpu=true)` 试图通过默认 NCCL process group 汇总 CPU tensor，报 `RuntimeError: No backend type associated with device type cpu`。这是真实 DDP validation 阻断，不是资源、数据或 W&B 故障。
- [x] 仅在 `src/wrappers/voxel_point_stage1_metrics.py` 做最小修复：当 world size 大于 1、默认 backend 为 NCCL 且 metric 明确 `compute_on_cpu=true` 时，建立并复用一个 Gloo process group 供这些 CPU metric 精确聚合；GPU/binned metric 继续使用默认 NCCL。新增定向回归测试证明所有 CPU metric 共享该 Gloo group，远端正式 Conda 环境的相关两个测试文件共 `14 passed`。
- [x] 首次失败目录原样归档为 `formal/Find_1_CPC1_failed_nccl_cpu_metric_20260721T0549/`；只复用自有 Job 321108，未删除 `after_lock`、未取消 sbatch。修复重跑使用唯一 tag `formal2_metricfix`，在线 W&B run `mb99gwzt`，已完成 2/2 NCCL rendezvous，正在正式 pool/lazy-module 初始化并继续高频监控至 sanity validation 和首个 optimizer step。
- [x] `mb99gwzt` 再次完成 2/2 sanity forward；原 TorchMetrics traceback 已消失，证明专用 Gloo group 生效。随后 Lightning logger 对已经全局聚合完成的 CPU scalar 又用默认 NCCL 执行 `sync_dist`，在 `logger_connector/result.py` 报同类 backend 错误。最小后续修复仅把 `on_validation_epoch_end` 中该已聚合 payload 的 Lightning 二次同步关闭；这也消除重复归约，不改变任何指标定义。
- [x] 新增 logger 回归断言后，远端正式 Conda 环境定向测试更新为 `15 passed`。第二次失败目录归档为 `formal/Find_1_CPC1_failed_lightning_cpu_log_sync_20260721T0612/`，第三次复验使用唯一 tag `formal3_logsyncfix`，已在原 Job 321108 内派发；继续监控到 sanity validation 与首个 optimizer step。
- [x] 第三次 Find_1 run `nj97r8mc` 已完成双 rank 的 2/2 sanity validation，并在 W&B 推进到至少 `trainer/global_step=29`；原两处 CPU/NCCL traceback 均未复现。双 H200 长跑峰值当前为 72.962%/75.615%，GPU utilization 均到 100%，仍低于 Find 85% 门限；继续监控长期峰值。

### Stable formal training checkpoint — 2026-07-21 06:48+08:00

- [x] 三个 producer 均处于正式训练推进而非排队/初始化状态，W&B 均为 `running`：Find_1 `nj97r8mc` 至少 global step 41，Find_0 `pzfktt9t` 至少 global step 14，unet_c1 `ohhl54t1` 至少 global step 260；均仍在 epoch 0，尚未到第一次周期 validation。
- [x] 最新正式峰值/利用率：Find_1 双 H200 为 77.181%/76.060%，Find_0 单 H100 为 53.502%，unet 单 H100 为 79.091%；三者 utilization 均达到 100%。百分比只用于启动前 smoke 的经验性容量判断，不是正式长跑停止条件。
- [x] 纠正显存百分比语义：旧 `>85%`/`>87%` 停止脚本均未作为自动监控器运行，并已改成显式拒绝执行；正式训练只因 OOM、进程/数据/数值错误或用户明确授权的替换操作终止，不因运行中显存百分比本身终止。
- [x] (2026-07-21 10:16+08:00) 按用户经验完成 Find micro-batch 重测与最终冻结：Find_1 的 micro8 试跑受控结束后按用户明确决定恢复为 micro6；Find_0 micro6 后续完成 5-step smoke。两个正式 Find 最终均为 micro6，allocation 与 `after_lock` 全程保留。
- [x] (2026-07-21 10:49+08:00) Find_0 单 H100/micro6/accum8 完成 5 optimizer-step smoke：40 micro-batch、exit 0、峰值 99.226%、GPU utilization 100%、无 OOM；按用户明确选择启动正式链，不因百分比回退。
- [x] (2026-07-21 11:02+08:00) 三个正式 CPC1/unet 已统一按 GBS48、micro6、`warmup_ratio=0.005`、`val_per_epoch=25` 从头运行；Find_0 `wm5e3v72` 至少 step 20，Find_1 `2py4thx4` 至少 step 23，新 unet 正在冷初始化。
- [x] (2026-07-21 11:02+08:00) 处理 unet `setsid` 孤儿进程：旧 run `ohhl54t1` 的 PGID/SID 183780 未随 lock wrapper TERM 退出，经 run stamp/PID/PGID 精确核对后在 hnode02 内 KILL；新 warmup=0.005 PID/PGID 232266 未受影响，恢复证据写入旧归档。
- [x] (2026-07-21 11:17+08:00) 三个最终 run 均进入真实 optimizer step 且在线 W&B 正常：Find_0 `wm5e3v72` 至少 step 50，Find_1 `2py4thx4` 至少 step 47，unet `tp6k1gar` 至少 step 2；三份正式错误匹配均为 0。
- [x] (2026-07-21 11:25+08:00) 初次用三条 W&B 相邻 `warmup_lr` 增量反推调度，但漏计 `log_every_n_steps=3`，得到的 warmup=1,316、首次 validation≈526 是错误中间结论；11:40 已由真实 Trainer micro-batch 总数纠正。
- [x] (2026-07-21 11:35+08:00) 用户彻底撤销 Find_0 双 H100 切换设想：无论后续是否出现两张空闲 H100，都不监控、不重提、不替换当前 Job 321106；只有用户再次主动提出才重新评估。
- [x] (2026-07-21 11:40+08:00) 查明旧 unet `ohhl54t1` 在 global step 968 无正式 validation 属预期：epoch 共 315,903 micro-batch，`val_check_interval=0.04` 在 micro-batch 12,636、global step≈1,579/1,580 才首验；替换动作开始时约 step 918–919，旧 `setsid` 孤儿继续到 micro-batch 7,749/step 968 才终止。本轮三条 run 的正确首验关注窗口同步改为 step≈1,579/1,580。
- [x] (2026-07-21 11:46+08:00) 经用户授权创建并校验当前任务的每 3 小时 heartbeat `adaligand-stage1`，状态 `ACTIVE`；休眠前 Job/锁/stderr 均健康，W&B Find_0/Find_1/unet 分别至少 step 98/98/83。
- [x] (2026-07-21 11:55+08:00) 补强 run-scoped validation 审计工具：`tmp/stage1_wandb_validation_audit.py` 现支持 `--train-log` 与 `--run` 二选一，解决部分 W&B URL 不写入 `train.out` 时无法审计的问题；服务器 `/home/penghongen/My_Project/tmp/` 精确非删除式同步、`py_compile`、CLI 与旧 run 直接查询均验证通过。
- [x] (2026-07-21 13:04+08:00) 完成 Find_2 `grill with doc` 并获得实施授权：冻结 Gaussian/tune 数值语义、独立 CPC1→CPC2、统一 producer 名单、下游 subagent 实施/审计、两个单 H200 sbatch 探路和 Job 321108 交接契约；实现期 heartbeat `adaligand-stage1` 已暂停。
- [x] (2026-07-21 13:22+08:00) 完成 Find_1 Gaussian 与 Find_2 56D tune 核心、CPC1/CPC2/Selector 配置和统一 producer 名单下游适配；本地配置/Gaussian 数值测试 15 passed，下游独立审计 73 passed，并修复一处 CPC-v3 测试分类遗漏。
- [x] (2026-07-21 13:34+08:00) 同时提交新版 Find_1/Find_2 单 H200 Job 321372/321373；321372 立即 RUNNING 后精确 `scancel 321108`，321373 随旧 cgroup 释放也进入 RUNNING。旧 Find_1 记为用户批准的架构/资源替换，不是 crash。
- [x] (2026-07-21 15:18+08:00) 两条新链已完成 micro6/GBS48/accum8 的 5-step smoke并启动正式 CPC1；Find_1/Find_2 在线 W&B 曲线均已正常绘制。用户撤销“至少 30 分钟且各到 step 50”的休眠门槛，允许立即恢复三小时 heartbeat；旧前台 30 秒轮询已停止，不影响服务器训练。

### Find_2 implementation checkpoint — 2026-07-21 13:22+08:00

- [x] `Stage1EmbedHead` 新增默认关闭的 `use_gaussian_splatting` 与 `voxel_embed_as_tune`。Find_1 保持 49D raw residual、2D occupancy 和 density56 拼接，只把 scatter 切换为 sigma=0.7 的 3×3×3 Gaussian；Find_2 把有效 voxel 输出硬编码为 56D、关闭 voxel raw residual/occupancy，并与 density56 逐元素相加；共同 point 64D 路径不变。
- [x] 新增 `Find_2` embed/CPC1/CPC2/Selector 配置；CPC2 仍使用同名 CPC1 BEST 的 `init_from="***"` 解析语义。`tests/test_adaligand_stage1_configs.py` 已覆盖六份 Find 阶段配置、两个 Gaussian/tune 开关和 Find_2 独立 checkpoint 谱系。
- [x] 新增 `src/stage1_producers.py` 作为 `Find_0`、`Find_1`、`Find_2`、`unet_c1` 的唯一 producer 名单，并由其派生 Find 子集；artifact、Dataset、inference、CLI 与 Selector 复用该名单，三份邻近 README 已同步当前接口。
- [x] 本地 `baseline_env` 验证：配置测试 `10 passed`，Gaussian scatter 数值/边界测试 `5 passed`，相关源码 `py_compile` 通过。完整模型测试在收集阶段只因该本机环境缺少 `torch_cluster` 而不可运行，未产生代码断言失败；计划在服务器正式环境补跑。
- [x] subagent A 下游验证：核心回归 `40 passed`，Selector calibration/推理 CLI `11 passed`，`compileall` 与 `git diff --check` 通过。subagent B 独立审计复测下游 `73 passed`、配置/producer `21 passed`；唯一阻断是 `tests/test_cpc_v3_configs.py` 旧分类名单遗漏 Find_2，已改为从 `FIND_MODEL_NAMES` 派生并复验通过。
- [x] 主线在审计修复及文案同步后重跑 artifact/Dataset/inference/Selector/CPC/AdaLigand 配置组合，结果 `67 passed in 46.79s`；与服务器核心 `42 passed` 共同构成本轮发布测试证据。
- [x] 13:18+08:00 只读服务器巡检确认 Job 321106/321107/321108 均 `RUNNING`；三个 `after_lock` 存在、`try_lock`/`kill_lock` 不存在，当前三份 stderr 错误模式匹配均为 0。实现和本地测试期间没有停止旧训练。

### Single-H200 replacement checkpoint — 2026-07-21 13:46+08:00

- [x] 非删除式安全同步后，远端 `bash -n`、`py_compile` 和旧三 Job 健康检查通过。提交返回 Find_1 Job `321372`、Find_2 Job `321373`；两者均为单 H200、24 CPU、无时限 allocation，长期 H200 占用回到两张。
- [x] 321372 在 hnode03 真正 RUNNING 后按用户确认规则直接 `scancel 321108`。`sacct` 记录 321108 为 `CANCELLED by 1351/0:0`；取消原因是新版架构与资源替换。Find_0 Job 321106、unet_c1 Job 321107 全程保持 RUNNING。
- [x] 首次服务器发布测试在两条新 allocation 上均为 `41 passed, 1 failed`：旧 `MagicMock` embed-head stub 未设置新增 bool，动态未知属性被 truthy 解释为 tune。最小修复只在测试 stub 显式设置 `voxel_embed_as_tune=False`，两条重派均为 `42 passed`；失败日志已保留为 `find2_release_tests_failed_missing_mock_flag.*`。
- [x] Find_2 第一次 Dataset smoke 被 run-scoped `tmp/stage1_formal_dataset_smoke.py` 的旧三模型 CLI choices 拒绝，尚未进入 Dataset；已让该临时工具复用 `STAGE1_MODEL_NAMES`，保留 `dataset_smoke_Find_2_failed_old_cli.*` 后在原 Job 321373 重派。科学 Dataset/模型代码未因此修改。
- [x] Find_1 321388 与 Find_2 321373 的真实 5-step smoke 均 exit 0。两者 runtime 日志均确认 micro-batch 6；Find_1 loss `0.805482→0.667957`、峰值 103,682/143,771 MiB=`72.116%`，Find_2 loss `0.803148→0.666263`、峰值 99,718/143,771 MiB=`69.359%`，utilization 均为 100%，无 OOM。
- [x] (2026-07-21 14:00+08:00) 确认 321372 被分配的 H200 同时存在本任务外 PID 85729 `alphafold3/run_af_json.py --card 4`；用户明确要求只读查验、不得干预该进程，也不得再 `scancel` 现有 Job。后续保留 321372/321373 allocation 与 `after_lock`，若 smoke 因共享卡 OOM，只等命令自然返回 `try_lock` 后在原 Job 内恢复。
- [x] (2026-07-21 14:02+08:00) 用户本人手动 `scancel 321372` 并授权重新提交；新 Find_1 Job 321388 使用 `h200g4` 单 H200/24 CPU 立即 RUNNING，获配干净 UUID `GPU-4af0ba5d-b718-0a8d-01bc-670b512c2772`（1 MiB、0%、无计算进程），与 AlphaFold 所在 `GPU-05e7...9100` 不同。321388 已完成服务器发布测试 42 passed 并进入 smoke 链。
- [x] (2026-07-21 14:31+08:00) 两条 smoke 成功后自动进入正式 CPC1：Find_1 Job 321388 run stamp `job321388_Find_1_CPC1_gaussian_single_h200_m6_w1`，Find_2 Job 321373 run stamp `job321373_Find_2_CPC1_gaussian_single_h200_m6_w1`。resolved config 均为 `offline=false`、GBS48/micro6/device1/accum8、val25、warmup0.005、stop-after-4、`init_from=null`；当前在线 W&B gate 已通过，训练进程正在 DataModule 冷加载。
- [x] (2026-07-21 14:47+08:00) 两条正式 CPC1 均完成在线 W&B 初始化并进入 optimizer step：Find_1 run `wdej8848`、Find_2 run `s7j101zn`，首次查询均为 `running`、step 2、epoch 0。日志再次确认 micro6/accum8/effective GBS48。
- [x] run-scoped W&B watchdog 原先会在 1800 秒内未见成功文本时终止仍活着的训练；双并行冷加载接近该窗口，虽本次约 20 分钟即成功、未触发，但这与用户“不因 W&B 未显示终止健康任务”的要求不一致。已改为只要 train PID 活着就继续等明确成功/失败，并非删除式同步到共享仓库及 321388/321373 runtime；三处 SHA256 均为 `498eaf981bde97af238a95bc11126e5772db15dea196335c72fa77120d2fc76a`，`bash -n` 通过，当前训练未重启。

### W&B disaster recovery and heartbeat checkpoint — 2026-07-21 15:26+08:00

- [x] 四条正式 sbatch 均仍为 `RUNNING`：Find_0 Job 321106（hnode01，1×H100）、unet_c1 Job 321107（hnode02，1×H100）、Find_2 Job 321373（hnode03，1×H200）和 Find_1 Job 321388（hnode03，1×H200）。没有执行 `scancel`、`kill_lock` 或任何 AlphaFold 操作。
- [x] 新增 AdaLigand 的人类双击入口 `../AdaLigand/与服务器交互/sync_stage1_wandb_remote.bat` 及其 PowerShell 包装器。它只扫描固定正式根 `/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations` 下的 `offline-run-*`，下载独立快照后执行 `wandb sync --no-mark-synced`，不删除、改名或标记既有本地/服务器日志；默认真实同步，`-DryRun` 只预览。
- [x] 主线独立复验 PowerShell AST 解析和真实 BAT `-DryRun`：枚举 17 个离线 run，全部属于 `Find_0`、`Find_1`、`Find_2`、`unet_c1`，没有下载或上传。当前四条正式 run 已 online 成功；该入口用于未来被迫 offline 或 online 未完整同步时的灾备补传。固定正式根之外的新 run scope 不会自动纳入，届时必须显式更新包装器。
- [x] 只读吞吐审计显示当前不同窗口内 Find_0/H100 为 `54.553 s/optimizer-step`、Find_1/H200 为 `60.725 s/step`、Find_2/H200 为 `60.711 s/step`，H200 两条表观慢约 11.3%。这不能判定为 H200 硬件退化：Find_1/2 采用真正的 3×3×3 Gaussian scatter，且同驻 hnode03、合计 40 个 DataLoader worker并共享 CPU/内存/I/O；比较窗口 global step、原子数和 recycle passes 也未匹配。该结果不触发调度更改或训练停止，后续只在相同步区间继续只读观察。
- [x] 恢复既有 heartbeat `adaligand-stage1`，状态 `ACTIVE`、每 3 小时；prompt 已更新为四 Job、四 W&B run、四 producer/七段训练、W&B 不作为健康训练停止条件和 AlphaFold 禁止干预边界。最初误保留旧线程 ID，发现后已立即纠正并复核为当前线程 `019f82d6-4c5a-7ec0-9993-55fd0f4570cf`。

## Surprises & Discoveries

- Observation: 指定活 ExecPlan 在接管时存在但内容为空，且为 Pocket_Plus 工作树未跟踪文件。
  Evidence: `git status --short --branch` 显示 `?? talk/Excx_执行stage1端到端训练.md`，读取结果为空。
- Observation: 集群 QOS 没有默认值；未显式指定 `cpu96` 的 CPU sbatch 会报 `Invalid qos specification`。H200 `h200g2` 又把单作业 CPU 限制在 48 以下，原 48 CPU 双 H200 请求进入 `QOSMaxCpuPerJobLimit`。
  Evidence: 首次提交在 sbatch 阶段失败且本人队列为空；补 `--qos=cpu96` 后 CPU Job 321104 RUNNING。双 H200 Job 321105 因该 reason 保持 PENDING，取消后以 24 CPU 重提为 321108，并于 2026-07-21 02:51+08:00 在 hnode03 RUNNING。
- Observation: 严格字段和 PDB/map 聚合首次实际计数为 20,483 个唯一 PDB，高于约 18,000 的诊断信号。
  Evidence: Job 321104 日志的资产审计总任务数为 `20483`；过滤程序只在严格计数不低于 18,000 时审计严格集合。
- Observation: 严格 PDB 中有 2,190 个无法被现有 Find Dataset 消费，原因全部是 exp/sim 的 shape、voxel size 或 origin 不完全一致；没有缺失路径、ZIP/NPY 头、schema 或 occurrence identity 异常。
  Evidence: `inventory/asset_audit.jsonl` 聚合只有 `exp_sim_geometry_mismatch: 2190`。成熟 `src/datasets/stage1_dataset.py::_materialize()` 对 Find 使用相同的 `shape` 与 `np.array_equal(voxel_size/origin)` fail-fast 条件。

- Observation: 初始显存试跑没有进入 Trainer；Hydra structured config 拒绝不存在的 `train.max_steps` 覆盖并明确要求 `+train.max_steps=2`。  
  Evidence: 三个 allocation 的 trial stderr 均为 `ConfigCompositionException`，scope 中只有失败时生成的约 75-byte resolved config；各 allocation `last_exit_code=1` 且 `try_lock` 已恢复。该故障属于 `tmp/` 运行包装，不是 Pocket_Plus Stage1 科学代码 bug。
- Observation: 串行 BOX 验收 Job 321119 对 Lustre 上 13,715 个逐文件 stat 过慢；在确认替代任务可运行后精确取消，并由单次目录枚举、16-thread、仍调用生产 `_load_pdb_pool` 的 Job 321141 完成同等 schema/候选验收。  
  Evidence: 321119 最终 `CANCELLED`；321141 `COMPLETED/0:0`，`box_pool/verification.json` 为 `status=ok`。

- Observation: 当前 torch 2.4.1+cu121 在 H100/H200 节点对 `expandable_segments:True` 先报告“不支持”，进入 3D convolution/instance norm 后会触发 CUDACachingAllocator internal assert；显式仅使用 `max_split_size_mb:256` 后该断言消失并暴露真实 OOM/峰值。  
  Evidence: 三个较大 micro trial 的第一次实际 forward 均出现同一 internal assert；修正环境后 Find_0/unet 能完成 optimizer steps，较大 unet micro=12 则正常报告可解释的 CUDA OOM。
- Observation: 双 H200 Find_1 在 `ddp_find_unused_parameters=false` 时完成第一个 optimizer step，第二步由 DDP reducer 报告存在未参与 loss 的参数；开启 detection 后 micro=6 顺利完成 2 steps。  
  Evidence: `Find_1_g48_m8_d2/train.err` 的 DDP unused-parameter RuntimeError 与 global_step 1；`Find_1_g48_m6_d2` exit 0、`max_steps=2 reached`、两卡 utilization 100%。
- Observation: 单卡/短 trial 无法暴露 Find_1 正式 DDP sanity validation 中的 CPU metric/NCCL backend 冲突；正式双卡首次全局 AP 归并才提供真实 bug 证据。  
  Evidence: 首次 run `c641mktn` 的 traceback 精确落在 `ValidationMetricManager.compute_payload` → TorchMetrics `BinaryAveragePrecision.compute` → `gather_all_tensors`，输入为 CPU tensor 而默认 process group backend 为 NCCL；修复后定向服务器测试为 14 passed，正式 run `mb99gwzt` 用于端到端复验。
- Observation: 专用 Gloo group 修复后，Lightning 仍会对 manager 已同步的 CPU payload 做一次框架级 `sync_dist`；因此 DDP validation 有两个相邻但独立的 CPU/NCCL 边界。  
  Evidence: `mb99gwzt` 已越过 TorchMetrics compute，第二个 traceback 精确落在 Lightning `logger_connector/result.py` → DDP strategy `reduce`；关闭已全局聚合 payload 的重复同步后定向测试更新为 15 passed，第三次正式 run 用于最终复验。
- Observation: 当前设备登记的 `Pocket_Plus_windows` Conda 环境路径已不存在；可用的 `baseline_env` 含 torch/Hydra，但缺少 `torch_cluster`，因此完整 Stage1 模型测试在 import `src.model.sparse_refine.anchor_sampler` 时停止于收集阶段。
  Evidence: 显式解释器路径不存在；`baseline_env` 的配置与 Gaussian 纯 torch 测试分别 10/5 passed，而 `tests/model/test_online_pdb_feature.py` 收集报 `ModuleNotFoundError: torch_cluster`。服务器正式环境补测是本轮发布门禁。
- Observation: 曾把用户给出的 Find/unet 显存百分比经验值误解为正式长跑硬停止线，导致 Find_1 run `nj97r8mc` 在峰值 85.098% 时被过早受控停止；该 run 的 W&B `crashed` 是 TERM 后的外部状态，不是训练或 W&B 自身故障。
  Evidence: 用户于 2026-07-21 明确澄清 87%/95% 只描述启动前小样本 smoke 的经验安全值；正式长跑超过该百分比不能触发 `kill_lock`。旧 run、TERM 证据与归档均保留用于审计。
- Observation: `configs/train/stage1_cpc1.yaml` 自提交 `d5981423` 起把 Stage1 默认 `val_per_epoch` 写成 10，且 CPC2 通过 `stage1_cpc2.yaml` 继承该值；这与用户要求的每 epoch 25 次 validation 不一致。
  Evidence: Git blame 指向 2026-07-20 的 `d5981423`，本轮改动前该文件无工作树差异；当前正式 launcher 显式覆盖 `train.val_per_epoch=25`，故在途作业 resolved config 正确。默认值已改为 25，并增加 CPC1/CPC2/unet_c1 组合配置回归断言。
- Observation: 初次按相邻 W&B `warmup_lr` 记录推算 schedule 时漏掉 `log_every_n_steps=3`，把每条历史记录间的增量误当成每个 optimizer step 的增量，因此将 warmup 和首次 validation 都低估为真实值的三分之一。
  Evidence: 旧 unet `train.out` 显示 `Epoch 0/19 7749/315903`，即每 epoch 315,903 个 micro-batch；micro6/accum8 对应约 39,488 optimizer step/epoch。Lightning 2.2.5 以 `int(315903*0.04)=12636` 个 micro-batch 触发首验，对应 global step≈1,579/1,580。W&B 相邻记录实际跨 3 个 optimizer step，正确反推 warmup≈3,948、20 epoch 总预算≈789,600 optimizer step。
- Observation: 旧 unet `ohhl54t1` 在 global step 968 没有正式 validation，不是配置错误、验证漏跑或 W&B 上传失败，而是被 warmup 配置替换时尚未到首验触发点。
  Evidence: 旧 resolved config 明确 `val_per_epoch=25`、`check_val_every_n_epoch=1`、无 validation 截断；`train.out` 停在 micro-batch 7,749/315,903，距首验 micro-batch 12,636 尚余 4,887。W&B API 扫描 2,903 行的 max global step 为 968，validation keys/rows 都为空，本地日志也无正式 validation loop。启动时 2-batch sanity validation 已完成，但它不是正式完整 validation。`315903 = 25×12636+3`，Lightning 以 micro-batch 模除触发，所以完整 epoch 恰好执行 25 次而不会在末尾额外产生第 26 次。
- Observation: run-scoped validation 审计工具原先只从指定 `train.out` 提取 W&B URL；旧 unet 的该文件不含 URL，导致工具在真正查询 W&B 前失败。
  Evidence: 旧调用报 `train 日志中未找到 W&B run URL`。增加显式 `--run` 后，服务器用 `ohhl54t1` 查询进入 W&B history 扫描并准确报 `没有 val_loss/global/total 历史`，与独立历史核验一致。

## Decision Log

- Decision: 接受用户列出的 A–G 与 Pocket_Plus 成熟提交事实，不做无必要的穷尽复核；只在后续真实证据异常时回到这些前提。
  Rationale: 避免重复耗费时间并尊重正式 run 和成熟提交的既有验收。
  Date/Author: 2026-07-21 / Codex
- Decision: 所有一次性过滤、切分、并行、显存试跑、sbatch 与锁调度包装只放在本地 `tmp/` 或服务器 `/home/penghongen/My_Project/tmp/`；不进入科学源码和默认入口。
  Rationale: 遵守用户明确边界与项目 run-scoped scaffolding 治理。
  Date/Author: 2026-07-21 / Codex
- Decision: 远端训练和重型加载仅通过 sbatch/lock 承载；普通 SSH 只做轻量状态探测和脚本/命令编排。
  Rationale: 遵守服务器交互纪律，并确保正式训练可审计。
  Date/Author: 2026-07-21 / Codex
- Decision: 本次正式 preparation root 冻结为 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000`；不覆盖不存在的默认根，也不与历史训练混用。
  Rationale: 独立根使 inventory、split、BOX pool、manifest 和训练绑定可追溯，满足最新 grill 的显式绑定要求。
  Date/Author: 2026-07-21 / Codex
- Decision: 先取得三个 producer 的正式 allocation，再在数据发布前用 run-scoped `pre_lock` 保持不执行；Find_1 使用双 H200，Find_0/unet_c1 各使用单 H100。
  Rationale: 当前资源允许立即启动，满足 Find_1 优先和三个 sbatch 尽快 RUNNING，同时不超过 2×H200、2×H100。
  Date/Author: 2026-07-21 / Codex
- Decision: 最终 inventory 采用严格质量通过且 Stage1 五类直接资产基本可读、exp/sim 几何满足现有 Dataset 契约的 18,293 PDB，不触发全量资产回退。
  Rationale: 18,293 高于约 18k 的异常信号；排除的唯一原因会在现有 Find Dataset 真实读取时直接阻断，属于必要资产兼容性而非新质量规则。
- Decision: 对 Find_1 的 CPU AP metric 保持精确全局聚合语义，通过专用 Gloo process group 修复 NCCL 不能收集 CPU tensor 的 backend 契约冲突，不改成 rank 均值、不把未分箱预测长期留在 GPU，也不改变模型、loss、Dataset 或 checkpoint 语义。
  Rationale: 这是能够修复实际阻断且不改变科学指标定义的最小改动；服务器定向测试与正式双卡重跑共同作为验收证据。
  Date/Author: 2026-07-21 / Codex
- Decision: Find/unet 的显存百分比只用于正式启动前短 smoke 的经验性容量选择；正式训练期间即使越过该百分比也不得因此终止。当前新一轮 Find smoke 经验参考值为 90%，若用户建议的 batch 超过 90% 或失败，则报告并使用已经验证的较小 micro-batch 继续正式链路。
  Rationale: 用户明确纠正阈值语义；运行中峰值会受样本长度与 allocator 状态影响，百分比本身不是训练失败条件。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 按用户明确授权，用精确 `kill_lock` 替换 Find_0 micro2 与 Find_1 micro6 当前运行，分别测试 micro6 与 micro8；共同 global batch 仍为 48，因此 accumulation 分别为 8 与 3。
  Rationale: 用户提供了上一轮可行 batch 的经验，希望在继续消耗完整 epoch 前重新冻结更高吞吐配置。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 用户随后把两个 Find 的最终 micro-batch 都明确固定为 6；Find_0 即使 smoke 峰值超过 90% 也不因百分比回退，只有真实 OOM 才进入故障恢复。最终共同 GBS 48 下，Find_0 为单 H100/micro6/accum8，Find_1 为双 H200/micro6/accum4。
  Rationale: 90% 是启动前经验观察值，不是用户最终 batch 选择的否决条件；用户要求先按既有经验运行，真实 OOM 后再修复。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 把 `configs/train/stage1_cpc1.yaml` 的 `val_per_epoch` 默认值从 10 改为 25，并通过继承同时校正 CPC2；正式 launcher 继续保留显式 25 覆盖作为运行审计证据。
  Rationale: 用户明确要求每个 epoch 完整 validation 25 次；生产默认契约不应依赖 run-scoped 覆盖才能满足该要求。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 把 `configs/train/stage1_cpc1.yaml` 的 `scheduler.warmup_ratio` 从 0.025 统一改为 0.005，并让 unet_c1、Find_0 CPC1、Find_1 CPC1 都从头使用新值；在途 unet_c1 与 Find_1 用精确 `kill_lock` 替换，Find_0 容量 smoke 因显式固定 `warmup_steps=5` 可继续，随后正式 CPC1 使用 0.005。
  Rationale: 当 `warmup_steps=null` 时该比例决定从头训练的 warmup 步数；继续旧 optimizer 状态不能等价应用新 schedule，因此必须从头重启受影响的正式阶段。
  Date/Author: 2026-07-21 / User + Codex
- Decision: Find_0 保持当前 Job 321106 的单 H100/micro6/accum8 正式链；先前“仅在双 H100 能立即 RUNNING 时才切换”的条件建议已整体作废，后续即使资源充足也不得主动重新提交。
  Rationale: 用户明确撤销该调度分支，要求主线只监控三条既有任务并维护 ExecPlan/handoff；避免后续恢复会话把旧条件误当成仍待执行的动作。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 旧 validation 调度核验和记忆更新完成后，使用绑定当前任务的 heartbeat `adaligand-stage1` 每 3 小时恢复监控；不另建独立 cron 任务。
  Rationale: 用户明确允许此时设置 heartbeat 并休眠；绑定原任务可直接继承持久目标和活 ExecPlan，避免并行任务产生状态分叉。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 新增独立 producer Find_2；其 voxel embed 使用 56D、无 voxel raw49 residual、无附加 occupancy/centroid、3×3×3 Gaussian 后与 density56 直接相加。新版 Find_1 只把现有 49D+raw49 residual+2D occupancy 的投影核改为 Gaussian，仍与 density56 拼接。
  Rationale: 用户希望比较同维 tune 调制，并要求最小 bool 开关、硬编码 56 和不增加保护层；两种模型共享已测试的 sigma=0.7、逐原子质量守恒 Gaussian 实现。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 统一维护当前四个 Stage1 producer 名单并派生 Find 子集；下游适配由 subagent 实施、第二个 subagent 独立审计。新版 Find_1/Find_2 以两个单 H200 无时限 sbatch 同时探路，至少一个真正 `RUNNING` 后直接取消旧双 H200 Job 321108。
  Rationale: 消除重复枚举并保留现有行为；现场证据显示双 H200 Find_1 吞吐与单 H100 Find_0 相当，把两张 H200 分给两个独立实验更有效。资源交接允许极短 3–4 卡重叠，但长期回到两张 H200。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 从 14:00+08:00 起，外部 `alphafold3/run_af_json.py --card 4` 任务只允许只读查验，绝不发送信号或修改；同时不再 `scancel` 当前任何 Job。Find_1 若因同卡外部显存占用 OOM，保留单 H200 allocation 并等待命令自然返回 `try_lock` 后原地恢复。
  Rationale: 用户明确收紧当前资源操作边界；避免影响同账户下本任务之外的 AlphaFold 工作，并保留已经取得的正式 allocation。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 321372 由用户本人于约 14:02+08:00 手动取消并授权重新提交。替代 Job 321388 先停 `pre_lock`，只读确认分配到不同且干净的 H200 UUID 后才启动同一 Find_1 测试/smoke/formal 链；agent 没有取消 321372，也没有触碰 AlphaFold。
  Rationale: 规避 Slurm 外部进程占用的物理卡，同时严格遵守用户对老板任务和 Job 控制权的边界。
  Date/Author: 2026-07-21 / User + Codex
- Decision: online W&B 不可用不得终止健康训练；正式 run 的本地 `.wandb` 记录作为审计与后续补同步来源，并提供固定 Stage1 scope 的人类双击同步入口。入口只选择离线 run，使用独立快照和 `--no-mark-synced`。
  Rationale: 在线可视化是重要但非科学训练终止条件；保留原始日志并在事后补传，可兼顾训练连续性、可视化和不修改源文件的边界。
  Date/Author: 2026-07-21 / User + Codex
- Decision: 不因当前 H200/H100 表观 11.3% 吞吐差改调度；三小时 heartbeat 绑定当前线程 `019f82d6-4c5a-7ec0-9993-55fd0f4570cf` 继续监控。
  Rationale: 当前比较混入 Gaussian scatter、节点共驻和不匹配 batch 窗口，不能归因于卡型；用户已允许取消 step50/30 分钟休眠门槛。
  Date/Author: 2026-07-21 / User + Codex

## Outcomes & Retrospective

当前尚未完成。数据、split、BOX pool、真实读取 smoke、共同 GBS48 和 Find_2 实现/下游适配证据均已完成。unet_c1 Job 321107 继续正式训练；Find_2 已按用户决定在 global step 53 停止，不再属于完成标准。原 Job 321540 的双 H100 allocation 已直接启动 Find_1 CPC1，W&B run `qqmuqyxk` online 且至少到 global step 119。Find_0 Job 321743 已获得双干净 H200 并以 micro8/accum3 派发，正在 smoke 后自动进入正式链；旧单卡 Job 321718 已精确取消。后续仍需完成 unet_c1、Find_0 CPC1→CPC2、Find_1 CPC1→CPC2 共五段训练的 BEST、LR-drop 停止、30 次 validation 和 CPC1→CPC2 strict model-only 谱系验收。

## Context and Orientation

Pocket_Plus 本地仓库位于当前仓库；AdaLigand 是相邻仓库 `../AdaLigand/`。服务器项目为 `/home/penghongen/My_Project/Pocket_Plus`，临时运行目录为 `/home/penghongen/My_Project/tmp/`。正式 A–G run 名称为 `adaligand_ag_20260711T154658`。过滤以 map/PDB 为判定单位：严格条件是实际字段 `cc_contour > 0.6` 且 map resolution `< 7.0 Å`，通过的 PDB 保留全部 occurrence。若严格清单复核无实现错误但唯一 PDB 明显少于约 18,000，不增加新质量规则，而从 20,000+ 全量密度图候选中选择 Stage1 直接资产齐全的样本形成最终清单，并明确记录回退。

最终 split 以最终清单中的唯一 PDB 为原子单位：train 为 `floor(0.75 × N)`，validation 恰好 200，calibration 恰好 100，held-out 为其余全部；同一 PDB 不跨 split，并保持成熟代码的确定性划分和 80³ eligibility 约定。

BOX pool 是不预先物化大量 dense BOX 的轻量索引池；train 和 validation 都必须发布 center、bias、context 三类请求索引及 manifest/completion，并由真实 Dataset/DataLoader 证明可以组成 batch。

正式训练 producer 为 unet_c1、Find_0、Find_1、Find_2。Find 的 CPC2 必须只加载同名 CPC1 BEST 的模型权重，不恢复 optimizer/scheduler/epoch；CPC1 checkpoint 继续保留。所有阶段共享 `global_batch_size=48`，当前单卡正式链均为 `micro_batch_size=6`、`gradient_accumulation=8`。每个 epoch 的 validation 分成完整 25 次；unet_c1 和 CPC1 在第 4 次实质学习率下降后停止，CPC2 在第 1 次实质学习率下降后停止，所有阶段 `max_epochs=20`。W&B 默认开启。显存百分比只用于启动前 smoke 的容量经验判断，不是正式训练的停止条件；W&B 网页或 online 暂时不可用也不得终止健康训练。

## Plan of Work

先读取全部权威文档、当前活日志和服务器资源/作业状态，定位正式 A–G 产物、现有 Stage1 入口和可能已有的中间产物。随后在本地 `tmp/` 编写或复用最小包装，在服务器 `tmp/` 执行只针对本次 run 的过滤、资产审计与确定性 split；每一步先生成临时目录，验证计数、唯一 PDB、无 split 泄漏、来源和基本可读性，再以成熟发布语义落盘。

数据清单冻结后，使用成熟 Stage1 box pool 入口按 CPU 总活跃核数不超过 192 的约束生成 train/validation 三类 pool。验证候选数、随机种子、selection、manifest 和 completion 后，在受调度资源上运行真实 Dataset/DataLoader smoke。

之后查看实时 GPU 空闲状态，优先让 Find_1 立即获得最多两张 H200；只有一张空闲 H200 时使用单卡。其余 producer 按 H200、H100、A800 的顺序争取尽快启动，同时不超过两张 H200 和两张 H100。先在实际 allocation 内进行短 batch 显存试跑，冻结共用 global batch 和各 stage 的 micro-batch/accumulation，再启动正式链路。刚提交和刚启动时高频检查，稳定后降低频率但持续监控利用率、错误、W&B、validation、LR drop、BEST 和阶段切换。

## Concrete Steps

本轮使用非删除式 `与服务器交互/sync_code.ps1` 把本地成熟代码和 `tmp/` 脚手架同步至 `/home/penghongen/My_Project/Pocket_Plus`。本地 Python 先通过：

    python -m py_compile tmp/stage1_prepare_inventory.py tmp/stage1_freeze_split_200.py

远端提交前通过 `bash -n`、相同 `py_compile`，以及：

    PYTHONPATH=/home/penghongen/My_Project/Pocket_Plus \
      /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python \
      -c 'from src.datasets.stage1_split import freeze_stage1_splits; from src.datasets.stage1_box_pool import build_pdb_box_pool'

2026-07-21 初始作业：

    321104  adaligand_s1_inventory  cpu/cpu96  48 CPU  cnode01  RUNNING
    321108  adaligand_s1_find1      h200       2 H200, 24 CPU  hnode03  RUNNING/pre_lock
    321106  adaligand_s1_find0      h100       1 H100, 24 CPU  hnode01  RUNNING/pre_lock
    321107  adaligand_s1_unet       h100       1 H100, 24 CPU  hnode02  RUNNING/pre_lock
    321111  adaligand_s1_box_pool   cpu/cpu96  32 CPU  cnode01  RUNNING (afterok:321104)

被替换的 321105 是本任务自有的双 H200 PENDING job；因 `QOSMaxCpuPerJobLimit` 取消，未开始运行，替换为 321108。四个当前 allocation 合计申请 CPU 120，不超过 192。普通 SSH 仅用于 `squeue`、`sinfo`、文件/日志查看和轻量编排；训练命令由上述 sbatch allocation 承载且不设置时间上限。

## Validation and Acceptance

验收必须同时满足用户列出的八类完成事实：可追溯清单/split、完整 BOX 发布与真实读取 smoke、batch/显存冻结、unet_c1 BEST、两个 Find 的 CPC1→CPC2 谱系与 checkpoint、五段配置/validation/停止/W&B/日志审计、所有阶段正确终态和故障恢复、最终 ExecPlan/handoff 收口。任何单纯“已提交作业”都不算完成。

## Idempotence and Recovery

清单、split 和 BOX 生成必须使用 run-scoped 输出与 completion/manifest 语义；重跑前先识别已完成分片，避免覆盖正式已验收产物。任何运行中作业依赖的脚手架不删除。只操作本任务自行提交或明确接管的精确 Job ID；未完成全部五段且 allocation 可复用时不删除 `after_lock`、不取消 allocation。客户端中断后，以本文记录的 Job ID、路径、manifest、completion 和服务器 `squeue/sacct` 为恢复依据。

## Artifacts and Notes

- A–G 正式 run：`/storage/penghongen/AdaLigand/Ori_Data/reports/runs/adaligand_ag_20260711T154658`
- Stage G candidates：`.../stage_g_analysis/candidates.pending.jsonl`，662,078 occurrence，SHA256 `86c9cfc75d3b91357b5f42def79eb9bee5fcc4637afc7dbe35dfb4b8136a786e`
- 本轮 preparation root：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000`
- 严格清单：`.../inventory/strict_filtered_keep_list.jsonl`，20,483 PDB/637,140 occurrence
- 最终清单：`.../inventory/final_keep_list.jsonl`，18,293 PDB/579,688 occurrence，fallback=false
- 资产审计：`.../inventory/asset_audit.jsonl` 与 `summary.json`，2,190 个不兼容 PDB 全为 `exp_sim_geometry_mismatch`
- split：`.../split/{train,validation,calibration,held_out_pool}.json`；PDB 计数 13,719/200/100/4,274
- inventory/split Slurm 日志：`/home/penghongen/My_Project/tmp/adaligand_s1_inventory_321104.{out,err}`
- BOX Slurm 日志：`/home/penghongen/My_Project/tmp/adaligand_s1_box_pool_321111.{out,err}`
- BOX 正式发布：`.../box_pool/{manifest.json,validation_selection.npz,verification.json,_COMPLETE}`；train/validation 发布 PDB 为 13,715/200
- batch 冻结：`/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/batch_freeze.env`；共同 GBS 48，Find_1/Find_0/unet micro 为 6/2/6，accumulation 为 4/24/8；Find_0 扩大试跑峰值 44.179%
- Find_0 旧 micro=4 正式失败归档：`/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/321106/formal/Find_0_CPC1_unsafe_m4_20260721T0548/`
- allocation scope：`/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/{321108,321106,321107}/`
- Job index：`/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/jobs.env`
- Find_0 CPC1 W&B：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/s04troim`
- Find_0 CPC1 micro2 recovery W&B：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/pzfktt9t`
- unet_c1 W&B：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/ohhl54t1`
- Find_1 CPC1 W&B：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/c641mktn`
- Find_1 CPC1 metric-fix recovery W&B：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/mb99gwzt`
- Find_1 CPC1 successful distributed-validation recovery W&B：`https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/nj97r8mc`
- Find_1 首次失败归档：`/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/321108/formal/Find_1_CPC1_failed_nccl_cpu_metric_20260721T0549/`
- Find_1 第二次失败归档：`/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/321108/formal/Find_1_CPC1_failed_lightning_cpu_log_sync_20260721T0612/`
- 最终 batch freeze：共同 GBS 48；Find_0=1×H100/micro6/accum8（5-step smoke 99.226%、无 OOM），Find_1=2×H200/micro6/accum4，unet=1×H100/micro6/accum8。
- 当前 warmup=0.005 W&B：Find_0 `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/wm5e3v72`；Find_1 `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/2py4thx4`；unet `https://wandb.ai/pencounkdual-111/AdaLigand_Stage1/runs/tp6k1gar`。
- 旧 unet warmup=0.025 归档：`/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/321107/formal/unet_c1_replaced_warmup0025_to0005_20260721T1038/`，含 `orphan_process_recovery.env`。
- 最小科学源码修复：`src/wrappers/voxel_point_stage1_metrics.py` 与 `src/wrappers/voxel_point_stage1.py` 的 validation payload 日志同步边界；定向测试 `tests/test_voxel_point_stage1_metric_logging.py` 与 `tests/test_voxel_point_stage1_thin_coordinator.py` 在正式服务器环境 15 passed

## Interfaces and Dependencies

科学实现优先使用 Pocket_Plus 当前 `src/datasets/`、`src/train.py`、`src/wrappers/voxel_point_stage1.py` 和 Hydra 配置。服务器 Conda 环境为 `/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu`。不得在无真实 bug 证据时修改现有 Stage1 科学代码；若阻断，只做最小修复并运行相应测试。

## Plan Drift / Reconciliation

### Beneficial drift

None known.

### Neutral drift

用户在三模型正式训练进行中曾新增 Find_2 作为第四个正式 producer，随后根据伪原子和真实原子经 density box 初始化后可能丧失纯密度判别能力的科学判断，撤回 Find_2 正式训练目标，范围恢复为原五段；Find_1 继续采用 Gaussian embed scatter。该往返不改变已发布数据、split、BOX pool、Find_0 或 unet_c1 语义，Find_2 实现和 run 证据保留而不继续消耗训练资源。

### Harmful drift

曾把启动前 smoke 的显存经验值错误执行为正式长跑硬停止线，导致 Find_1 `nj97r8mc` 在 85.098% 时被过早 TERM。用户已纠正语义；旧证据保留，所有百分比停止脚本已禁用，Find_1 已恢复并进一步按用户建议测试 micro8。该问题只影响调度过程和额外耗时，没有修改 checkpoint/科学计算语义。

### Unfinished scope

数据清单、split、BOX pool、真实读取 smoke、Find_2 实现/适配/测试和停止证据已经完成；Find_0 已获得双干净 H200 并启动。尚未完成的是 unet_c1、Find_0/Find_1 CPC1→CPC2 共五段训练的终态、BEST/初始化谱系、30 次 validation、LR-drop 停止与 W&B/日志总审计，最后还需脚手架分类和 handoff 收口。

## Revision Note

2026-07-21 13:04+08:00：根据 `../AdaLigand/grill_with_memory/07-21-12-07.md` 的最终授权，把 Find_2 正式纳入当前 ExecPlan，范围从三 producer/五段训练扩为四 producer/七段训练；记录 Gaussian/tune 科学契约、统一 producer 名单、subagent 分工、两个单 H200 sbatch 交接规则和实现期暂停 heartbeat。

2026-07-21 02:28+08:00：从空文件建立初始自包含 ExecPlan，记录用户确认事实、上游关系、执行边界、完成标准和首个恢复点。

2026-07-21 02:54+08:00：补记正式 preparation root、20,483 严格 PDB 首次数、CPU inventory/split Job 321104、三个 RUNNING GPU allocation 321108/321106/321107、QOS/CPU 上限恢复和精确 lock/log 路径。

2026-07-21 03:10+08:00：记录 inventory/split 正式完成、18,293 最终 PDB、2,190 个 exp/sim 几何不兼容的唯一排除原因、四 split 精确计数，以及依赖作业 321111 开始并行生成 BOX pool。

2026-07-21 05:24+08:00：记录 BOX/真实 smoke/显存试跑完成、共同 GBS 48 冻结、三条正式训练链启动、Find_0 与 unet 在线 W&B run，以及 Find_1 双卡冷初始化的当前恢复点。
2026-07-21 06:10+08:00：记录 Find_0 micro=4 正式显存超安全线后的精确锁恢复与 micro=2 扩大试跑；记录 Find_1 首次双卡 validation 暴露 CPU metric/NCCL backend bug、最小 Gloo process-group 修复、服务器 14 项定向测试通过、失败证据归档和 W&B run `mb99gwzt` 正式重跑。

2026-07-21 10:06+08:00：按用户澄清把显存百分比改为仅用于启动前 smoke 的经验判断，禁用所有百分比正式停止脚本；记录 Find_1 `nj97r8mc` 的过早 TERM 为调度策略误解而非训练/W&B 故障，并记录用户授权替换 Find_0/Find_1 当前 run、并行试跑 micro6/micro8。

2026-07-21 10:16+08:00：记录用户最终选择两个 Find 均使用 micro6；Find_1 b8 trial 受控结束并立即以双 H200/micro6/accum4 恢复正式链，Find_0 继续完成单 H100/micro6 的 5-step smoke，成功后不受 90% 百分比影响直接启动 micro6/accum8 正式链。

2026-07-21 10:29+08:00：审计发现 `stage1_cpc1.yaml` 的默认 `val_per_epoch=10` 与正式 25 次 validation 要求漂移；确认在途作业均由 launcher 显式覆盖为 25 后，把生产默认改为 25，并为四个 Find 配置和 unet_c1 增加组合配置回归断言。相邻的 CPC1 默认 LR-drop=3 与本轮显式 4 要求暂只记录，不在本次 validation 定向修改中擅自扩大变更。

2026-07-21 10:38+08:00：按用户新决定把三个从头训练 producer 的 `warmup_ratio` 从 0.025 统一改为 0.005；精确替换在途 unet_c1 与 Find_1，Find_0 b6 容量 smoke 保持固定 5-step warmup 继续，正式启动时再使用新 ratio。旧 run、W&B、日志与归档全部保留，不释放 allocation。

2026-07-21 11:02+08:00：记录 Find_0 micro6 五步 smoke 以 99.226% 峰值、无 OOM 完成并正式启动；记录三条最终 resolved config 均为 GBS48/micro6/val25/warmup0.005；记录旧 unet setsid 孤儿 PGID 183780 的精确 KILL 恢复及新 PID 232266 保留；实时 H100 每节点最多仅一张空卡，因此按用户确认忽略双 H100 Find_0 建议，不终止当前 Find_0。

2026-07-21 11:17+08:00：记录三条最终 run 均完成在线 W&B 初始化并产生 optimizer step，启动高频监控阶段完成；后续进入稳定监控，继续交叉检查 Slurm/锁、GPU、stderr、W&B、25 次 validation、LR drop、BEST 与 CPC1→CPC2 切换。

2026-07-21 11:25+08:00：以三条 W&B 真实 `warmup_lr` 轨迹初步核验 0.005 schedule 已运行时生效；当时漏计 `log_every_n_steps=3`，得到的 warmup=1,316、首次 validation≈step526 为待纠正中间结果。

2026-07-21 11:35+08:00：记录用户彻底撤销 Find_0 双 H100 切换分支，不再因空卡变化监控或重提；记录旧 unet 在 900+ step 未显示 validation 的待核验异常，并要求本轮首验以本地日志、W&B 与 artifact 三方证据判定。三条当前正式 run 均继续健康推进，无停止或重提操作。

2026-07-21 11:40+08:00：完成旧 unet validation 核验并纠正调度推算。旧 run 的真实 epoch 规模为 315,903 micro-batch；`val_check_interval=0.04` 对应首验 micro-batch 12,636/global step≈1,579/1,580。替换于约 step 918–919 开始，`setsid` 孤儿最终停在 micro-batch 7,749/global step 968，因此零正式 validation 正常；W&B 零 validation rows 与本地日志一致。同步把本轮首验关注窗口从错误的 step≈526 改为 step≈1,579/1,580。

2026-07-21 11:46+08:00：经用户授权创建当前任务 heartbeat `adaligand-stage1`，`ACTIVE`、每 3 小时唤醒。休眠前 Job 321106/321107/321108 均 `RUNNING`，锁和正式 stderr 正常，W&B Find_0/Find_1/unet 分别推进到至少 step 98/98/83。

2026-07-21 11:55+08:00：持久目标即时续跑期间补强 `tmp/stage1_wandb_validation_audit.py`，允许通过 `--run entity/project/run_id` 直接审计，保留原 `--train-log` 入口且要求二选一。精确同步到服务器 `/home/penghongen/My_Project/tmp/stage1_wandb_validation_audit.py`；正式环境语法/CLI 通过，旧 unet 直接查询返回预期的零 validation 历史证据。

2026-07-21 12:00+08:00：确认三个当前 W&B 本地 run 目录及 `.wandb`、`wandb-summary.json`、`debug-internal.log` 持续刷新；重定向 `train.out/err` 在训练循环期间不持续写 Rich 进度。因此后续首验以在线 API、本地 W&B 文件和 validation artifact 交叉核对，不把静态 stdout 当成训练停滞。

## Live Checkpoint — 2026-07-21 03:22+08:00

- GPU allocation 321108（2×H200）、321106（1×H100）和 321107（1×H100）已经完成二次成功 preflight，均保留 `after_lock` 并停在各自 `try_lock`；首次 preflight 仅因 run-scoped shell 在 Conda activate 周围使用 `set -u` 导致 `ADDR2LINE` 未定义，已通过只修改临时命令包装恢复，未修改科学代码。
- 三个 allocation 的 CUDA、cuDNN、显卡可见性和 W&B API 登录均已通过；321108 额外完成双卡 NCCL all-reduce。证据位于各 allocation scope 的 `allocation_preflight.json`、`resolved_preflight.yaml`、`nvidia_smi_preflight.csv`，双卡另有 `nccl_preflight.json`。
- Job 321111 在 03:21+08:00 已生成约 3,600/13,919 个 PDB pool，32 worker 持续运行且 stderr 为空；总活跃 CPU 为 104，低于 192 上限。
- 已提交依赖 `afterok:321111` 的轻量启动 Job 321119。它只在 BOX 正式 `_COMPLETE` 发布后，向三个已获资源的 allocation 派发真实 center/bias/context Dataset/DataLoader smoke 与短训练显存试跑；初始候选为共同 global batch 48，Find_1 双 H200 micro=12、Find_0 单 H100 micro=8、unet_c1 单 H100 micro=16，均为试跑值而非最终冻结值。
- Lightning 2.2.5 源码核验表明 `val_per_epoch=25` 会解析为 `val_check_interval=0.04` 和 `val_check_batch=floor(B/25)`；正式 epoch 的 train batch 数大于 650 时调度恰好 25 次 validation。仍须以首个正式 epoch 的真实日志计数作为最终证据。
- 03:23+08:00 的 Slurm `sstat` 证据显示 Job 321111 在约 26 分钟内累计 `AveCPU=02:56:32`、读入约 1.48 TB（含密度/标签等源资产反复读取），同时 32-process pool 持续推进到 7,300/13,919。该阶段实际是并行 I/O 受限而非串行空转；申请 32 CPU、峰值约 19.3 GB RSS，任务总活跃 CPU 仍为 104。
- 321119 在唤醒 GPU 试跑前会先用生产 `_load_manifest_pool_paths`/`_load_pdb_pool` 逐个读取正式 manifest 声明的全部 train/validation pool，并展开 validation selection，核对 summary、80³、30 bias candidates、1:5:3 role 及 context 汇总，成功结果发布为 `box_pool/verification.json`。
- Pocket_Plus 当前提交核实为 `de6a89f38e48553edfd440c76bf61823229d9205`；科学源码未修改。Windows 本地可运行的 `test_adaligand_stage1_configs.py`、`test_stage1_dataset.py`、`test_stage1_split_pool.py` 共 24 项通过。checkpoint 单测在本机因基础 Python 缺少 `rootutils` 无法收集，服务器正式 Conda 环境的 train import、CUDA 与模型/NCCL preflight 已通过，因此不把本机环境缺包误记为科学代码 bug。
- Dataset、短训练和正式训练包装均显式设置 `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`；双 H200 的正式 DataLoader 计划每进程 10 worker（共 20），单 H100 每个 20 worker，均留出主进程/监控余量且不突破 allocation CPU 与本任务 192 核总上限。
- `scontrol show job -o` 已确认三个正式 allocation 的 `TimeLimit=UNLIMITED`：321108 为 h200/h200g2、2×H200、24 CPU；321106 和 321107 均为 h100/h100g2、1×H100、24 CPU。满足“正式训练使用 sbatch 且不设置运行时间上限”。
- 正式训练前容量门禁：`/home` 可用约 27 TB，`/storage` 可用约 135 TB，足够保留五段 checkpoint、W&B、本地日志与临时审计。一次递归 `du` 辅助统计因既有 feedback 目录过大而在 124 秒超时，已停止；它只读且不是主链依赖。
- 04:01:43+08:00 BOX pool 已原子发布 `_COMPLETE`；Job 321111 最终 `COMPLETED/0:0`，耗时 54:59、32 CPU、MaxRSS 约 21.65 GB。summary：train 请求 13,719、发布 13,715、按既有规则 short map 4、zero-context PDB 2、underfilled-context 0；validation 请求/发布均 200、zero/underfilled-context 均 0。冻结 selection 为 center 3,281、bias 16,405、context 9,843；manifest 为 train 13,715、validation 200。
- 发布文件时间/大小：`box_pool/manifest.json` 1,058,631 bytes，`validation_selection.npz` 272,010 bytes，`_COMPLETE` 0 bytes。afterok 启动 Job 321119 已于 cnode01 RUNNING，正在做逐 manifest 文件存在性与生产 schema 读取；三套 GPU allocation 继续停在 `try_lock`。

## Live Checkpoint — 2026-07-21 07:06+08:00

- 三个正式 allocation 继续为 `RUNNING` 且 `TimeLimit=UNLIMITED`：Find_1 Job 321108 使用 hnode03 的 2×H200，Find_0 Job 321106 使用 hnode01 的 1×H100，unet_c1 Job 321107 使用 hnode02 的 1×H100；三个 `after_lock` 均保留，未执行 `scancel`，也未释放 allocation。
- Find_0 首次正式 micro=4 run `s04troim` 在较长真实序列中达到 97.468% 显存，已按精确 `kill_lock` 恢复并保留全部证据。扩大 micro=2 试跑完成 120/160 microbatch、5 optimizer step、峰值 44.179% 后，冻结共同 GBS 48 下的 `micro=2, accumulation=24`；恢复 run `pzfktt9t` 已推进到 global step 47，越过旧 micro=4 的同一长跑比较区域时峰值仅 59.981%，W&B 为 `running`，无 OOM/worker-killed/traceback。
- Find_1 的首次双卡正式 validation 暴露 CPU TorchMetrics 经 NCCL 聚合错误，第二次又暴露已全局聚合的 CPU scalar 被 Lightning 重复 `sync_dist`。只对 `src/wrappers/voxel_point_stage1_metrics.py` 与 `src/wrappers/voxel_point_stage1.py` 做了两处最小同步边界修复，并新增定向回归；服务器正式 Conda 环境共 15 项相关测试通过。第三次 run `nj97r8mc` 已完整通过双 rank 2/2 sanity validation 和正式 optimizer step，原两类错误均未复现，已推进到至少 global step 68。
- Find_1 当前正式双 H200 峰值为 79.188%/80.147%，Find_0 为 59.981%，unet_c1 为 79.091%；三者均采样到 100% utilization，低于 Find 85% 与 unet 95% 门槛。两个 `>85%` 精确停止脚本仅处于已校验待命状态，尚未执行，当前没有正式 run 被门禁中断。
- unet_c1 W&B run `ohhl54t1` 已推进到至少 global step 338，Find_1 `nj97r8mc` 已到至少 global step 80，epoch 均仍为 0；Find_0、Find_1、unet_c1 均尚未到第一次完整周期内 validation。继续以 W&B 状态/history 推进、GPU 峰值/利用率、正式 stderr 错误模式三方交叉监控，直到产生可审计终态和 BEST。

## Stable Monitor Checkpoint — 2026-07-21 07:41+08:00

- 三个 W&B run 均持续为 `running`：Find_1 `nj97r8mc` 已到至少 global step 116，Find_0 `pzfktt9t` 已到至少 74，unet_c1 `ohhl54t1` 已到至少 416；三者仍在 epoch 0，尚未进入第一次周期内 validation。
- 截至本检查点的正式长跑峰值：Find_1 双 H200 为 80.218%/82.821%，Find_0 单 H100 为 63.772%，unet_c1 单 H100 为 79.091%；所有 GPU 均采样到 100% utilization，Find_1 虽接近但仍低于 85% 硬门禁，未执行停止脚本。
- 周期健康探针确认 Job 321108/321106/321107 均为 `RUNNING`，三个 `after_lock` 均存在，`try_lock`/`kill_lock` 均不存在；当前三份正式 stderr 对 traceback、OOM、NCCL、worker-killed 的匹配数均为 0，尚无任何 CPC1 `result.env`，符合阶段仍在正常训练。
- `tmp/stage1_wandb_live_probe.py` 增加只读 summary step 与 `--terse` 输出，用于区分 W&B 批量同步延迟和真实停滞；它不修改 run。`tmp/adaligand_stage1_formal_health_live.sh` 固定执行只读 Job/锁/错误/阶段结果交叉检查。

## Stable Monitor Checkpoint — 2026-07-21 11:35+08:00

- 当前最终 W&B run 都为 `running`：Find_0 `wm5e3v72` 与 Find_1 `2py4thx4` 均至少 global step 83，unet_c1 `tp6k1gar` 至少 step 56；均为 epoch 0，尚未到经真实 Trainer batch 数纠正后的 step≈1,579/1,580 首次 validation 窗口。
- Job 321106/321107/321108 均为 `RUNNING`；三个 `after_lock` 存在，`try_lock`/`kill_lock` 不存在，正式 stderr 错误匹配均为 0，尚无 stage `result.env`。
- GPU 仍有真实计算：Find_0/Find_1/unet 均采样到 100% utilization；正式峰值分别为 Find_0 99.307%、Find_1 80.844%/82.114%、unet 79.091%。这些百分比只记录，不触发正式训练停止。
- 用户观察到旧 unet run 在 global step 900+ 仍无 validation 展示；只读历史核验确认它在 micro-batch 7,749/global step 968 被替换，而首验应在 micro-batch 12,636/global step≈1,579/1,580，因此没有正式 validation 符合调度。本轮将在正确窗口同时核对训练日志的 validation loop、W&B `val_loss/global/total` 与 validation artifact。
- 11:46+08:00 休眠前复核：三条 W&B 均为 `running`，Find_0/Find_1/unet 分别至少 step 98/98/83；三个 Job/锁/stderr 仍健康。当前任务已由 heartbeat `adaligand-stage1` 每 3 小时恢复监控。
- 12:00+08:00 本地日志证据基线：三个正式 `train.out/err` 在训练循环内不持续刷新进度，但对应本地 W&B run 的 `.wandb`、summary 与 internal log 均持续更新到 11:59。首验审计将使用这些本地持久文件和 validation artifact 补强在线 W&B 证据。

2026-07-21 15:26+08:00：按用户新决定取消新 Find 各到 step50/密集监控 30 分钟的休眠门槛，停止本地前台轮询并恢复三小时 heartbeat；补充固定 Stage1 scope 的离线 W&B 一键同步入口、H200/H100 只读吞吐审计、当前四 Job/七段训练状态和正确 heartbeat 线程 ID。本次修订没有改变训练配置、Job、锁或科学产物。

## Resource Replan Checkpoint — 2026-07-21 17:32+08:00

- CPC1 生产默认与正式 launcher 已统一为 `lr=5e-5`、`patience=2`、`val_per_epoch=30`、`warmup_ratio=0.005`；CPC2 继承 LR 和 validation 次数，但按用户纠正保持 `patience=1`。CPC1/CPC2 的停止次数仍为第 4/1 次实质 LR 下降，硬上限 20 epoch。
- Find density cube 的 P/real chunk 已从 1024/2048 翻倍为 2048/4096；服务器正式环境定向测试 21 passed，三个 Find resolved config 均核对生效。unet_c1 不使用该 density cube。
- 旧 Find_0 Job 321106、Find_1 Job 321388 和 Find_2 Job 321373 均已自然 `COMPLETED/0:0`；用户保留的 unet_c1 Job 321107 继续以单 H100/micro6/accum8/GBS48 训练，进程参数显示 `lr=5e-5` 与 `val_per_epoch=30`。
- Find_2 新 Job 321540 已在 hnode01 获得 2×H100，以 micro6/accum4/GBS48 完成五步 smoke，`exit_code=0`；峰值 99.273%/99.270%，无 OOM，按用户明确规则不因百分比停止。正式 CPC1 run `30dq7bql` 已 online 初始化、错误匹配 0，并推进到至少 global step 11。resolved config 和启动日志确认 `lr=5e-5`、`val_per_epoch=30`、CPC1 第 4 次 LR 下降停止。
- hnode03 的 Slurm 空闲卡多次被外部 Python 物理占用。所有本任务 H200 候选均只停在 `pre_lock` 且未在脏卡上启动计算；321539、321555、321562、321563、321568、321569 及真实单探针 321607 均已精确取消或由调度器拒绝/取消，当前本用户名下无 H200 Job，任何外部进程均未收到信号。
- 新增低输出脚手架 `tmp/adaligand_stage1_h200_acquire_once.sh`，服务器副本为 `/home/penghongen/My_Project/tmp/adaligand_stage1_h200_acquire_once.sh`。每次最多提交一个 24-CPU H200 候选，30 秒内未运行或显存大于 2 GiB、利用率大于 5%、存在 compute process 时立即 `scancel` 精确候选；只有干净卡才以 `CLEAN_HELD` 保留在 `pre_lock`。真实 Job 321607 验证了脏卡输出 `DIRTY_RELEASED` 后立即释放。
- 后续 H200 状态机：当前无 Find_0 时仅探 1 卡，干净则启动单 H200/micro8/accum6/GBS48；单卡 Find_0 运行后才探 2 卡，只有新双卡 allocation 两张均物理干净并已派发双 H200/micro8/accum3 后，才精确 `scancel` 原单卡 Find_0。Find_1 继续等待 Find_0 释放 H200。

2026-07-21 17:32+08:00：记录 LR/patience/validation/chunk 最终配置、Find_2 双 H100 正式启动、H200 物理占用与短探针证据，并把 H200 管理收缩为单候选、24 CPU、低输出的单卡启动/双卡升级状态机；将当前验收目标从历史 25 次 validation 更新为用户最新确认的 30 次。

## Find_2 Withdrawal / Find_1 Direct Restart — 2026-07-21 18:12+08:00

- 用户根据代码和 W&B `train_loss/global/pseudo_step` 趋势，判断 Find_2 让伪原子与真实原子在 density box 初始化后都携带相似密度调制，可能削弱点云分支仅凭密度信息区分两者的能力，因此撤回 Find_2 正式训练目标。此决定不回滚 Find_2 源码或历史证据，只停止继续训练和后续 CPC2。
- Find_2 online W&B run `30dq7bql` 在 global step 53 记录最后状态：total 约 0.5788、pseudo 约 0.3025。Job 321540 收到精确 `kill_lock` 后以 exit 143 返回 `try_lock`；`after_lock` 保留，Find_2 日志、resolved config、GPU 记录和 W&B 均未删除。
- 新增 run-scoped dispatcher `tmp/adaligand_stage1_dispatch_direct_find1.sh`，服务器副本为 `/home/penghongen/My_Project/tmp/adaligand_stage1_dispatch_direct_find1.sh`。按用户授权跳过 smoke 与科学测试，直接在原 2×H100 allocation 上派发 Find_1 CPC1→CPC2。
- Find_1 正式参数已由 resolved config 核对：Job 321540、2×H100、micro6、accum4、GBS48、每进程 workers10、`lr=5e-5`、warmup0.005、CPC1 patience2、val30、LR-drop4、max20。五分钟外部审计结束时 Job/锁正常、GPU 监控文件持续更新，stderr 没有 Traceback、OOM、NCCL 或 worker-killed；尚处模型冷初始化，W&B 状态留给 heartbeat 后续确认。
- 用户要求今后再次提出临时训练改动时，agent 应先核查代码、当前证据、科学语义、资源代价和是否只是短期焦虑驱动，再明确支持、反对或建议更小实验，而不是机械执行频繁重启。该工作约定已写入 Pocket_Plus 项目 learning。

2026-07-21 18:12+08:00：按用户最终决定把训练范围从四 producer/七段收缩回三 producer/五段；记录 Find_2 主动停止证据、Job 321540 原地切换为双 H100 Find_1 的直接正式启动和五分钟外部审计结果。

## Heartbeat Correction / Find_0 H200 Capture — 2026-07-21 19:52+08:00

- 用户发现界面显示的旧 heartbeat 提示仍引用 Job 321106/321108、旧 W&B、val25 和撤销的单 H100 Find_0。磁盘中当前激活 automation 只有 `adaligand-stage1` 一个，旧文本是历史快照；实际提示已再次更新为当前五段目标，并把周期从 3 小时缩短为 15 分钟，以低输出探针抢占 H200 空档。
- Find_1 Job 321540 已完成 online W&B 初始化，run `qqmuqyxk` 至少推进到 global step 86；最新 summary total/pseudo loss 约 0.5990/0.2980，stderr 明确错误匹配 0。
- 低输出单卡 H200 探针立即捕获 Job 321718：24 CPU、GPU `GPU-38991406-...` 实测 1 MiB/0%，输出 `CLEAN_HELD`。已用既有 dispatcher 派发 Find_0 单 H200/micro8/accum6/GBS48/workers10；`after_lock` 保留，当前 Dataset smoke 已完成并开始五步 memory trial，尚无明确错误。
- heartbeat 当前每 15 分钟运行。只要单卡 Find_0 存在，就以参数 2 提交至多一个 24-CPU 双卡候选；只有两张均物理干净并输出 `CLEAN_HELD` 时，先派发双 H200/micro8/accum3，再直接 `scancel` 当前单卡 Find_0 精确 Job。若候选脏、等待或有队列则立即释放/跳过，不并行枚举。

2026-07-21 19:52+08:00：纠正 heartbeat 显示与实际状态的漂移，记录 Find_1 online/step86、Find_0 捕获干净单 H200 Job 321718，并将 H200 见缝插针频率调整为低输出的 15 分钟状态机。

## Find_0 Dual H200 Upgrade — 2026-07-21 20:10+08:00

- 周期检查确认 unet Job 321107、Find_1 Job 321540 和单 H200 Find_0 Job 321718 均 `RUNNING`，锁健康且近 30 分钟明确错误匹配均为 0。W&B 本地 summary：unet 至少 step746；Find_1 run `qqmuqyxk` 至少 step119，total/pseudo loss 约 0.5424/0.2881。
- `tmp/adaligand_stage1_h200_acquire_once.sh` 原先只按 Slurm job name 识别探针；已派发训练的 321718 仍保留 `...probe_1` 名称，导致它被误判为未处理候选。最小修复改为“名称匹配且对应 `pre_lock` 仍存在”才算探针；服务器 `bash -n` 通过，不修改训练或科学代码。
- 修复后参数 2 立即捕获双 H200 Job 321743：24 CPU，两张 GPU `GPU-4af0ba5d-...`、`GPU-2ddacb86-...` 均为 1 MiB/0%，输出 `CLEAN_HELD`。先成功派发 Find_0 devices2/micro8/accum3/GBS48/workers10，再精确 `scancel 321718`；外部任务未收到任何信号。
- Job 321743 当前 `RUNNING`、`after_lock` 存在且无其它锁，Dataset smoke 已完成，双卡五步 memory trial 已生成 resolved config 并开始运行，明确错误匹配 0。heartbeat 保持 15 分钟启动监控，但 H200 已达到 2 卡上限，不再运行任何 H200 探针；待正式 W&B/若干 step 稳定后降回 3 小时。

2026-07-21 20:10+08:00：记录 H200 探针名称/`pre_lock` 识别修复、双干净 H200 Job 321743 捕获、先派发双卡后取消单卡的完整切换，以及 heartbeat 从资源获取转为启动健康监控。

## Find_0 Dual H200 Smoke Passed — 2026-07-21 20:25+08:00

- Job 321743 双 H200/micro8/accum3/GBS48 五步 smoke 已成功 `exit_code=0`。两卡实测峰值为 97.346%/99.387%，利用率均到 100%；没有真实 OOM，因此按用户规则继续，不因百分比停止。
- 正式 `Find_0_CPC1/launch_1` 已创建并开始冷初始化；本检查点尚无 `wandb_status` 和 W&B summary。321743 仍 `RUNNING`，`after_lock` 存在、无 `try_lock/kill_lock`，明确错误匹配 0。
- Find_1 W&B run `qqmuqyxk` 至少推进到 global step149，近 30 分钟明确错误匹配 0；unet Job 321107 继续 `RUNNING` 且明确错误匹配 0。
- heartbeat 保持 15 分钟，不再探测 H200；下一轮只确认 Find_0 正式 online、resolved config 和首 optimizer step。达到 online 且推进若干 step 后再降回 3 小时。

2026-07-21 20:25+08:00：记录 Find_0 双 H200 micro8 五步 smoke 成功、正式 CPC1 冷初始化和继续 15 分钟启动监控的决定。

## Three Producers Stable — 2026-07-21 20:40+08:00

- Find_0 Job 321743 正式 CPC1 已完成 online W&B 初始化，run `w21l8dof` 至少推进到 global step20。resolved config 核对为 devices2、micro8、GBS48、workers10、`lr=5e-5`、warmup0.005、patience2、val30、LR-drop4、max20。
- 按文件 mtime 选择最新 W&B summary：unet Job 321107 run `0yy28kmj` 至少 step587；Find_1 Job 321540 run `qqmuqyxk` 至少 step173；Find_0 run `w21l8dof` 至少 step20。三条均为 epoch0 且持续更新。
- 三个精确 Job 均 `RUNNING`，`after_lock` 存在且无 `try_lock/kill_lock`；近 30 分钟 Traceback、OOM、NCCL error、worker-killed 和 Hydra execution error 匹配均为 0。
- 启动阶段验收完成，heartbeat 从每 15 分钟降回每 3 小时。当前 2×H200 已满，不再提交探针；后续转入 validation30、LR drop、BEST 和 CPC1→CPC2 谱系的长期监控。

2026-07-21 20:40+08:00：记录三个 producer 正式 W&B/optimizer step 稳定、最新 run ID 和 heartbeat 降频至 3 小时。

## First val30 Window Monitor — 2026-07-22 01:05+08:00

- 三个精确 Job 继续 `RUNNING`：unet_c1 Job 321107 已运行 22:12，Find_1 Job 321540 已运行 8:02，Find_0 Job 321743 已运行 4:56；三个 `after_lock` 均存在，`try_lock`/`pre_lock`/`kill_lock` 均不存在。
- 按正式日志根中的 run ID 和文件 mtime 选择 W&B summary：unet `0yy28kmj` 至少 step1316，Find_1 `qqmuqyxk` 至少 step653，Find_0 `w21l8dof` 至少 step485。三份当前 stderr 对 Traceback、OOM、NCCL、worker-killed、Hydra execution error 以及独立 NaN/Inf 边界匹配均为 0；尚无 `result.env` 或 BEST，符合 CPC1/unet 仍在训练。
- unet summary 在 step1316 暂停常规训练步刷新，但 `.wandb` internal stream 仍秒级更新，单 H100 GPU 监控同时显示 97%–100% utilization。结合固定 GBS48、accum8、历史真实 epoch 规模 315,903 micro-batch 和 `val_per_epoch=30`，首验触发点为 `floor(315903/30)=10530` micro-batch，即 optimizer step 约 1316；当前证据高度一致于首次完整 validation 正在执行，而不是训练停滞或 W&B crash。首个 `val_loss/global/total` 尚未提交，下一轮须确认该 validation 完成并落盘，不能提前把“正在执行”记成“已通过”。
- 新增 run-scoped 只读探针 `tmp/adaligand_stage1_heartbeat_probe.sh`：固定检查当前三个 Job、锁、严格错误模式、按 run ID/mtime 的 W&B summary 与 stream、GPU 监控、阶段结果和 checkpoint。该脚本不写服务器、不提交/取消 Job、不创建锁，也不扫描外部进程；最终脚手架收口时归类为 reusable operations tooling 或移出活动路径。
- 本轮无恢复动作、无 H200 探针、无锁写入、无训练配置变化。用户允许健康状态下把 heartbeat 从 3 小时降频为 6 小时；后续继续审计 unet 首次 val30 完成、两个 Find 到 step≈1316 的首次 validation、每 epoch 恰好 30 次、LR-drop4/1、BEST 和 CPC1→CPC2 strict model-only 谱系。

2026-07-22 01:05+08:00：记录三个正式 Job 的稳定推进、unet 在 step1316 进入首个 val30 高计算窗口的交叉证据、新只读低输出探针边界，以及健康后 heartbeat 调整为 6 小时；不改变科学配置或运行状态。

## First Validation Confirmed — 2026-07-22 07:18+08:00

- 固定三 Job 只读巡检继续健康：321107/321540/321743 均 `RUNNING`，运行时长约 1d04h/14h/11h；三个 `after_lock` 存在，`try_lock`/`pre_lock`/`kill_lock` 均不存在；严格 stderr 错误匹配均为 0。
- W&B summary 按当前正式 run ID/mtime 读取到 unet `0yy28kmj` step2375、Find_1 `qqmuqyxk` step1274、Find_0 `w21l8dof` step1067。三条 W&B internal stream 和 GPU 监控仍持续更新；Find_1/Find_0 尚处首验前，未发现阶段结果或 checkpoint。
- unet 首个正式 validation 已由 W&B API 直接核验：run `pencounkdual-111/AdaLigand_Stage1/0yy28kmj` 状态 `running`，history 中 epoch0 已有 1 条 `val_loss/global/total`，history `_step=3944`，值 `0.2606383264064789`；当前 `lr_reduction_count_max=0`，因此尚未发生实质 LR 下降，符合 CPC1/unet 第4次下降停止规则。
- 首个 validation 后正式 checkpoint 已落盘：`/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-unet_c1/unet_c1____job321107_unet_c1_lr5e5_p2_val30_m6_w1/checkpoints/TOP_epoch_00_score_0.2606.ckpt`（513,048,846 bytes）与同目录 `last.ckpt`（513,048,974 bytes）。当前只确认文件存在和 W&B 指标，不在普通 helper 会话中加载模型。
- 发现并修复 `tmp/stage1_wandb_validation_audit.py` 的只读审计缺陷：原实现把 validation 与 LR 字段放在同一次 `scan_history(keys=...)`，W&B 会按联合字段筛选，导致已有 validation 时错误报告“无 validation”。现改为 validation 与 LR 两次独立扫描；`tmp/test_stage1_wandb_validation_audit.py` 回归 `1 passed`，两份脚本 `py_compile` 通过，真实 unet 审计成功输出 `tmp/stage1_audits/unet_c1_0yy28kmj_validation.json`。
- `tmp/adaligand_stage1_heartbeat_probe.sh` 改为固定三个正式 run 根路径，单轮探针从约 64 秒降至约 17 秒，并分别显示 `TOP_*.ckpt` 与 `last.ckpt`；远端 `bash -n` 通过。该脚本仍只读、不写服务器、不创建/删除锁、不提交/取消 Job、不触碰外部任务。
- 本轮无恢复动作、无配置变化、无 H200 探针。Find_1/Find_0 接近 optimizer step约1316时继续审计首个 validation；随后按每 epoch 30 次、CPC1/unet LR-drop4、CPC2 LR-drop1、BEST 和 strict model-only 谱系推进。

2026-07-22 07:18+08:00：完成 unet 首个 val30 validation、W&B history、TOP/last checkpoint 与审计工具修复的证据闭环；保持三个正式 Job 和 6 小时 heartbeat 不变。

## Three Producers First Validation Confirmed — 2026-07-22 13:11+08:00

- 三个正式 Job 继续 `RUNNING`：unet 321107 约 1d10h、Find_1 321540 约 20h、Find_0 321743 约 17h；三个 `after_lock` 存在，无 `try_lock`/`pre_lock`/`kill_lock`，严格 stderr 错误匹配均为0，W&B stream 与 GPU 监控持续更新。
- 最新 summary：unet `0yy28kmj` step3209、`val_loss/global/total=0.2356146574`；Find_1 `qqmuqyxk` step1673、val loss `0.4114735126`；Find_0 `w21l8dof` step1529、val loss `0.4041147530`。三个阶段均仍为 epoch0，无 `result.env`，符合尚未完成 CPC1/unet 训练。
- 修复后的 W&B API 审计确认正式 validation 计数：unet epoch0 为2次，W&B history step `[3944, 7894]`；Find_1 epoch0 为1次，history step `[2191]`；Find_0 epoch0 为1次，history step `[1752]`。这里的 history step 是 W&B 行序号，不是 trainer global step；三个 run 的 `lr_reduction_count_max` 均为0。
- 三者首验后均已生成 TOP 与 last checkpoint：unet 当前 TOP `TOP_epoch_00_score_0.2356.ckpt`，Find_1 为 `TOP_epoch_00_score_0.4115.ckpt`，Find_0 为 `TOP_epoch_00_score_0.4041.ckpt`；路径均位于各自正式日志根的 `checkpoints/`。这些是训练中间 BEST/TOP，后续 improvement 可覆盖或新增更优 TOP，不能提前当作最终终态 BEST。
- 本轮没有 OOM、进程/数据/数值错误，没有恢复、调度、锁写入、H200 探针或配置变化。后续按6小时 cadence 继续累计 epoch0 的 validation 次数，完整 epoch 结束时必须恰好30次，并继续等待 LR-drop4/阶段终态与 Find CPC2 strict model-only 初始化。

2026-07-22 13:11+08:00：完成 unet/Find_1/Find_0 三 producer 首次正式 validation、W&B 计数、TOP/last checkpoint 和 LR-drop0 的交叉验收；更新 heartbeat 快照但保持6小时周期。

## Validation Accumulation Monitor — 2026-07-22 19:12+08:00

- 三个正式作业继续运行：unet_c1 Job 321107、Find_1 CPC1 Job 321540 和 Find_0 CPC1 Job 321743 均为 `RUNNING`。三个作业目录中的 `after_lock` 文件存在，`try_lock`、`pre_lock` 和 `kill_lock` 文件均不存在；三份当前 `train.err` 对 Python 异常、CUDA 显存不足、NCCL 通信错误、数据加载进程异常退出、Hydra 配置错误以及非有限数值的严格匹配数均为0。
- W&B summary 显示 unet `0yy28kmj` 已到 trainer global step 4016，最新验证总损失为 `0.2216629684`；Find_1 `qqmuqyxk` 已到 step 2279，最新验证总损失为 `0.4114735126`；Find_0 `w21l8dof` 已到 step 2081，最新验证总损失为 `0.4041147530`。三个 W&B 内部日志和 GPU 监控文件均持续刷新，证明训练进程仍在计算。
- W&B 历史审计确认第一个训练轮次中的正式验证次数：unet 为3次，对应 W&B 历史记录序号 `[3944, 7894, 11846]`；Find_1 为1次，对应序号 `[2191]`；Find_0 为1次，对应序号 `[1752]`。W&B 历史记录序号用于标识上传记录，不等于 trainer global step。两个 Find 尚未到第二次验证约需的 trainer global step，因此各1次验证符合当前调度，没有发现漏验。
- 三个模型的 `train/runtime/lr_reduction_count` 最大值均为0，尚未发生实质学习率下降。unet 的训练中最佳检查点已改善为 `TOP_epoch_00_score_0.2217.ckpt`；Find_1 和 Find_0 的训练中最佳检查点仍分别为 `TOP_epoch_00_score_0.4115.ckpt` 与 `TOP_epoch_00_score_0.4041.ckpt`。三个正式目录均保留 `last.ckpt`。
- 本轮未出现需要恢复的错误，未修改配置，未创建或删除锁，未提交 H200 探针，也未触碰外部作业。继续按6小时周期累计验证次数；第一个训练轮次结束时，每个模型必须恰好完成30次正式验证。

2026-07-22 19:12+08:00：记录三个正式作业继续健康运行、unet/Find_1/Find_0 在第一个训练轮次中的验证次数为3/1/1、三个模型学习率下降次数均为0，以及 unet 训练中最佳检查点改善到0.2217；保持6小时 heartbeat 与现有资源不变。
