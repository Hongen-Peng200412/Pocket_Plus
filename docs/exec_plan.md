# CPC_main 训练链路修复与服务器验证

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

本仓库当前没有 `PLANS.md`。本文件是本次任务的权威执行记录，目标是让 `D:\OneDrive\My_Project\Pocket_Plus\configs\experiment\CPC\CPC_main.yaml` 对应训练逻辑通过测试、debug 提交循环，并在服务器资源 `/home/penghongen/try_lock_297517` 上正式训练且稳定监测。

## Purpose / Big Picture

用户需要的是一个能正式跑起来的 CPC 主实验，而不只是本地代码差异。完成后，`+experiment=CPC/CPC_main` 应能通过项目测试和服务器 smoke/debug 流程，最终提交到 Slurm/lock 资源并进入正式训练。训练稳定后，还要额外连续一小时、每十分钟一次记录状态。

## Progress

- [x] (2026-06-16 22:12 +08:00) Used `grill-me` on the key decision branch: when old tests conflict with the refactored CPC_main path, the refactored training contract wins; do not add production compatibility shims for removed APIs.
- [x] (2026-06-16 22:18 +08:00) Reverted the temporary compatibility shims in production code and migrated affected tests to current APIs (`select_cache_view_for_class`, `evaluate_global_instance_matching`, `FeatureCombine`).
- [x] (2026-06-16 22:20 +08:00) Local fast validation passed for migrated tests: `tests/inference/test_voxel_cache.py`, `tests/inference/test_voxel_evaluator.py`, `tests/inference/test_voxel_tuning.py`, `tests/model/test_model_utils.py` (23 passed). Local atom-head tests are blocked by missing local dependency `addict`; server validation remains authoritative.
- [x] (2026-06-17 00:15 +08:00) Clarified postprocess connectivity semantics: second connected-component analysis is intentionally paused for compute savings, so policies `x_y` currently behave as `x_none`. The inactive second-pass branch remains commented in `src/inference/voxel_postprocess.py` for future re-enable.
- [x] (2026-06-17 00:17 +08:00) Server full test suite passed on resource 297517: `287 passed, 8 warnings`.
- [x] (2026-06-17 00:21 +08:00) CPC_main one-batch smoke exposed `VolumePointStage1Model.__init__` using `act_cls` before assignment. Fixed by resolving `act_cls` before density/fusion module construction.
- [x] (2026-06-17 00:36 +08:00) CPC_main one-batch smoke exposed voxel input channel mismatch: point-only `point_clipbig` first replaced `batch["atom_feat"]` with 64-dim embed output, then online `gauss27` scatter used that tensor although the intended semantics are raw 49-dim atom features. Fixed by preserving a trimmed raw atom feature copy for online scatter and by including `gauss27` occupancy/centroid auxiliary channels in lazy input-channel initialization.
- [x] (2026-06-17 00:40 +08:00) Local `python -m compileall src\model\stage1_model.py tests\model\test_online_pdb_feature.py` passed. Local targeted pytest collection is blocked by missing Windows dependency `hydra`; server environment remains authoritative.
- [x] (2026-06-17 00:39 +08:00) Server validation after online raw scatter fix passed: targeted `tests/model/test_online_pdb_feature.py tests/inference/test_voxel_postprocess.py` reported `23 passed`; full suite reported `289 passed, 8 warnings`.
- [x] (2026-06-17 00:45 +08:00) Manual CPC_main forward smoke passed through voxel/point/atom/pseudo/refine forward: `voxel_logits_ligand`, `atom_logits`, `pseudo_logits`, and `ligand_refine_logits_C` were all finite. Loss computation then failed because manual smoke bypassed Lightning Trainer and therefore lacked `trainer.estimated_stepping_batches`, which the refine-loss schedule intentionally requires.
- [x] (2026-06-17 00:48 +08:00) Added optional Trainer debug overrides in `src/train.py` (`train.max_steps`, `train.limit_train_batches`, `train.limit_val_batches`, `train.num_sanity_val_steps`) with defaults preserving normal training behavior, so smoke can run the real Trainer lifecycle for one step.
- [x] (2026-06-17 00:51 +08:00) Server Trainer smoke for `+experiment=CPC/CPC_main` passed with `max_steps=1`, `limit_train_batches=1`, `limit_val_batches=0`, `num_sanity_val_steps=0`, and `train.scheduler.warmup_steps=1`. The run completed `Trainer.fit` to `max_steps=1`, produced finite training losses, and initialized CPC_main with raw voxel channels `56`, voxel backbone input channels `110`, and density cube channels `56`.
- [x] (2026-06-17 00:53 +08:00) Server full test suite after the Trainer debug override change passed again: `289 passed, 8 warnings in 17.48s`.
- [x] (2026-06-17 00:54 +08:00) Submitted formal CPC_main training on resource 297517 using `/home/penghongen/run_cmd_297517.sh`, then removed `/home/penghongen/try_lock_297517` to let the authorized Slurm job consume the command. Formal log path: `/home/penghongen/My_Project/tmp/cpc_main_297517/logs/cpc_main_formal_train_20260617_005441.log`.
- [x] (2026-06-17 01:07-02:09 +08:00) Completed the required stability monitor window after formal training became stable. Seven samples at roughly ten-minute spacing showed job `297517` stayed `R` on `hnode01`, `try_lock_297517` stayed absent, the formal log contained no new `Traceback`/`RuntimeError`/OOM, and the W&B offline run file continued updating through `2026-06-17 02:08:25 +08:00`.

- [x] (2026-06-16 21:50 +08:00) 阅读 `CLAUDE/plans/最后重构代码_v2.md`、`configs/experiment/CPC/CPC_main.yaml`、`configs/base.yaml`、`src/train.py`、核心 model/wrapper/loss 文件与相关测试。
- [x] (2026-06-16 21:50 +08:00) 确认仓库无 `PLANS.md`，本计划直接写入 `docs/exec_plan.md`。
- [x] 修复本地静态审查发现的 pseudo geo head 潜在 bug，并补充单元测试。
- [x] 本地运行可行的快速测试，确认新测试能覆盖修复。
- [x] 使用项目安全同步入口同步到服务器，不运行 clean sync。
- [x] 使用 `/home/penghongen/try_lock_297517` 对应资源运行全量测试并修复暴露的问题。
- [x] 使用同一资源完成 `CPC_main` 的 debug-修改-提交循环，直到正式训练进入稳定状态。
- [x] 稳定后连续一小时、每十分钟一次记录训练状态。

## Surprises & Discoveries

- Observation: `torch_cluster.radius(x=real, y=pseudo)` 的返回索引方向是本轮最大风险点；计划文档里写的是 `idx_real=edge_index[0]`、`idx_pseudo=edge_index[1]`，但常见 `torch_cluster.radius` 语义是第一行对应 `y`、第二行对应 `x`。需要服务器依赖实测并用单测锁定。
  Evidence: `src/model/stage1_atom_head.py` 当前直接把 `edge_index[0]` 当 real，`edge_index[1]` 当 pseudo 使用。
- Observation: `PseudoGeometricAggregation` 当前把 `channels` 同时作为 pseudo query 通道和 real hidden 通道；如果 `atom_head_pseudo_feature_dim` 显式设置为非 `atom_head_hidden_dim`，几何头的 K/V 输入维度会不一致。
  Evidence: `edge_in_dim = self.channels + 3 + 1`，但 forward 传入的 `real_hidden` 来自 atom attention stack，通道数是 `hidden_dim`。
- Observation: `point_clipbig` 是 point-only embed head；它没有 `voxel_pdb_embed_grid`，因此 `online_pdb_feature` 仍是 voxel 分支的原子注入路径。`_run_embed_head_once` 会把 `batch["atom_feat"]` 替换成 embed 后 64 维点特征，如果 `_build_voxel_input` 直接使用该字段，CPC_main 的体素输入通道会从期望 `raw 49 + occupancy 2 + centroid 3` 漂移成 `embed 64 + occupancy 2 + centroid 3`。
  Evidence: 服务器 one-batch smoke 报 `expected ... 105 channels, but got 125 channels`; 125 对应 `56 raw voxel channels + 64 embed point channels + 5 gauss27 auxiliary channels`。
- Observation: CPC_main 的 sparse refine loss schedule depends on Lightning's `trainer.estimated_stepping_batches`. A manual `model(batch)` smoke can validate forward tensors, but it is not a valid end-to-end training smoke for scheduled losses.
  Evidence: manual smoke produced finite `ligand_refine_logits_C shape=(25000, 1)` and then failed with `ligand_sparse_refine_loss_schedule 需要 trainer.estimated_stepping_batches 为正整数` during `_compute_total_loss()`.

## Decision Log

- Decision: Do not add compatibility shims for removed pre-refactor APIs. When tests still import old names, migrate the tests to the current CPC_main training contract instead.
  Rationale: Compatibility shims hide stale contracts and increase the risk of training a path that is no longer intended to exist.
  Date/Author: 2026-06-16 / User + Codex
- Decision: Treat all advanced postprocess connectivity policies `x_y` as `x_none` for now; keep the second-pass connected-component code commented, not deleted.
  Rationale: The second pass was intentionally disabled to save compute. Tests should protect current runtime semantics rather than revive the old behavior.
  Date/Author: 2026-06-17 / User + Codex
- Decision: Keep `online_pdb_feature` semantics as raw atom feature scatter for point-only embed heads, and update tests to current `point_clipbig` behavior instead of adding production compatibility.
  Rationale: `CPC_main.yaml` and `point_clipbig.yaml` define this path as zero-parameter raw atom feature projection into the voxel branch. Using embed output there would silently change the intended training signal and input channel contract.
  Date/Author: 2026-06-17 / Codex

- Decision: 先修 pseudo geo head 的维度和 radius 索引方向，再做服务器全量验证。
  Rationale: 这是 CPC_main 新增能力中最容易在真实依赖和真实 P/real 数量下爆炸的路径；本地已有测试多用 fake zero geo，无法覆盖真实邻域索引。
  Date/Author: 2026-06-16 / Codex
- Decision: 不清理 `src/wrappers/voxel_point_stage1_old.py` 等 legacy 文件中的旧 `p_best` 痕迹。
  Rationale: 主训练入口使用 `src/wrappers/voxel_point_stage1.py`，legacy 文件不是 CPC_main 当前消费链路；清理会扩大风险。
  Date/Author: 2026-06-16 / Codex

## Outcomes & Retrospective

本次目标已完成。`CPC_main.yaml` 对应链路的主要问题分别定位并修复为：pseudo geo head 的 real/pseudo 通道与 radius 索引契约、当前 `x_y == x_none` 的连通域后处理语义、`act_cls` 初始化顺序、point-only embed head 下 online voxel scatter 必须使用原始 49 维原子特征、以及服务器短训练 smoke 需要走真实 Lightning Trainer 生命周期。旧 API 兼容没有继续保留，冲突测试已按当前语义迁移。

服务器资源 297517 上，全量测试最终通过 `289 passed, 8 warnings`；`+experiment=CPC/CPC_main` 的 1-step Trainer smoke 通过；正式训练已提交并稳定运行超过一小时监测窗口。最后一次监测时间为 `2026-06-17 02:09:08 +08:00`，Slurm 显示 job `297517` 仍为 `R`，`try_lock_297517` 未恢复，正式日志无新异常，W&B 离线 run 文件更新到 `2026-06-17 02:08:25 +08:00`。

## Context and Orientation

`configs/base.yaml` 是 Hydra 顶层入口。运行 `python src/train.py +experiment=CPC/CPC_main` 或 sbatch 模板时，Hydra 会加载 `base.yaml`，再由 `configs/experiment/CPC/CPC_main.yaml` 覆盖默认配置。`CPC_main.yaml` 选择 `model/default.yaml` 中的 wrapper 目标 `src.wrappers.voxel_point_stage1.VoxelPointStage1Wrapper`，其 `backbone` 目标是 `src.model.stage1_model.VolumePointStage1Model`。

本次 CPC 主路径包含 dense voxel ligand 分支、real atom head、pseudo ligand 前后置监督、sparse refine 候选集 C、P anchor、P 到 C 的 refine head，以及新增的 pseudo geometric aggregation head。pseudo geometric aggregation head 是后置分类增强：它只影响 `pseudo_logits`，不能修改返回给 sparse refine 的 `pseudo_feature`。

关键文件：

- `D:\OneDrive\My_Project\Pocket_Plus\configs\experiment\CPC\CPC_main.yaml`：CPC 主实验配置，打开 `enable_pseudo_geo_head: true`，使用 `pseudo_geo_max_neighbors: 48`。
- `D:\OneDrive\My_Project\Pocket_Plus\configs\model\default.yaml`：模型默认结构参数，pseudo geo 默认关闭但提供同样的默认超参。
- `D:\OneDrive\My_Project\Pocket_Plus\src\model\stage1_atom_head.py`：real/pseudo atom head 与 pseudo geo head 实现。
- `D:\OneDrive\My_Project\Pocket_Plus\src\model\stage1_model.py`：Hydra backbone 参数透传、最终 recycle、P 注入、atom head、sparse refine 调用。
- `D:\OneDrive\My_Project\Pocket_Plus\src\modules\losses.py`：`LigandSparseRefineDeltaLoss`，当前设计应是 ranking-only。
- `D:\OneDrive\My_Project\Pocket_Plus\src\wrappers\voxel_point_stage1.py` 与 `src\wrappers\voxel_point_stage1_losses.py`：Lightning wrapper 与组合损失。
- `D:\OneDrive\My_Project\Pocket_Plus\与服务器交互`：项目级服务器工具箱。只允许使用安全同步；训练和重负载测试必须走 sbatch/lock 资源。

服务器约束：

- 远端项目目录是 `/home/penghongen/My_Project/Pocket_Plus`。
- 服务器环境是 `/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu`，环境名 `Pocket_Plus_centos7_cu121_allgpu`。
- 用户明确授权使用 `/home/penghongen/try_lock_297517` 对应资源。围绕该资源的 `run_cmd_297517.sh` / lock 流程可以使用；不创建、不删除、不触碰 `kill_lock_297517`，除非用户另行明确授权。

## Plan of Work

第一阶段，本地修复 pseudo geo head。修改 `src/model/stage1_atom_head.py` 中 `PseudoGeometricAggregation`，让它接受 `pseudo_channels` 和 `real_channels`，K/V 的输入维度使用 real hidden 通道，输出仍投影到 pseudo feature 通道。同步修正 `torch_cluster.radius(x=real, y=pseudo)` 返回索引的解释，使用第一行作为 pseudo index、第二行作为 real index。补充测试覆盖 real/pseudo 数量不等时的索引方向，以及 `pseudo_feature_dim != hidden_dim` 时几何头仍可构造和 forward。

第二阶段，本地运行小测试：优先运行 `tests/model/test_stage1_atom_head.py`、`tests/test_ligand_sparse_refine_delta_loss.py`、`tests/test_ligand_sparse_refine_loss.py`，再做 `python -m compileall src` 或等价 py_compile。

第三阶段，按项目纪律安全同步到服务器。使用 `与服务器交互/run_sync.bat` 或 `sync_code.ps1`，不运行 `run_syncWithClean.bat` 与 `sync_codeWithClean.ps1`。

第四阶段，进入服务器资源循环。使用 `/home/penghongen/try_lock_297517` 运行全量测试，若失败则读取日志、回本地修复、同步、重测。通过后运行 CPC_main 的 debug smoke，确认 Hydra 配置可解析、模型可实例化、首个 batch/短训练可过。再提交正式训练。

第五阶段，正式训练稳定后，每十分钟记录一次状态，连续一小时。状态至少包含 Slurm job 状态、最近日志尾部、是否有 Python exception/OOM、最近验证或训练 step 进展。

## Concrete Steps

本地检查命令在 `D:\OneDrive\My_Project\Pocket_Plus` 执行：

    pytest tests/model/test_stage1_atom_head.py tests/test_ligand_sparse_refine_delta_loss.py tests/test_ligand_sparse_refine_loss.py
    python -m compileall src tests

服务器同步命令在本地项目根执行：

    .\与服务器交互\run_sync.bat

服务器命令通过项目 helper 或 lock 脚本执行，所有重负载任务绑定 `/home/penghongen/try_lock_297517`。

## Validation and Acceptance

本地接受标准：

- 新增 pseudo geo 单元测试失败于旧索引/旧维度实现，修复后通过。
- `stage1_atom_head`、ranking-only delta loss、sparse refine loss 相关测试通过。
- Python 编译检查通过。

服务器接受标准：

- 全量测试通过，或仅有明确与本任务无关且已记录的环境性跳过/失败。
- `+experiment=CPC/CPC_main` 配置解析、模型构造和短训练 smoke 通过。
- 正式训练 job 进入 running，日志持续推进，无 OOM、ImportError、Hydra 参数错误、shape mismatch、NaN 终止。
- 稳定后完成 6 次监测记录，覆盖连续约 60 分钟。

## Idempotence and Recovery

本地代码修改使用小步补丁，可以重复运行测试。同步只使用安全同步，不删除远端项目目录。服务器测试和训练脚本应写入项目工作目录或 `/home/penghongen/My_Project/tmp` 下的临时文件。若正式训练失败，读取日志定位后回到本地修复，再同步并重新使用同一授权资源循环。

不得运行 clean sync，不得触碰 `kill_lock_297517`，不得删除 `after_lock_297517`，不得修改服务器公共配置。

## Artifacts and Notes

- Server full test after final code changes: `289 passed, 8 warnings in 17.48s`.
- Trainer smoke command used `+experiment=CPC/CPC_main offline=true migrate_slurm_logs=false` with debug overrides `+train.max_steps=1`, `+train.limit_train_batches=1`, `+train.limit_val_batches=0`, `+train.num_sanity_val_steps=0`, and `train.scheduler.warmup_steps=1`; it reached `max_steps=1`.
- Formal training log: `/home/penghongen/My_Project/tmp/cpc_main_297517/logs/cpc_main_formal_train_20260617_005441.log`.
- Formal output dir: `/home/penghongen/My_Project/feedback_plus/logs/CPC/CPC_main____job297517`.
- Stability samples: `01:07:39`, `01:17:50`, `01:28:01`, approximately `01:38`, `01:48:21`, `01:58:28`, and `02:09:08` CST on 2026-06-17. All samples showed job `297517` running on `hnode01`, `try_lock_absent_running`, and no new formal-run exception/OOM pattern.

## Interfaces and Dependencies

`PseudoGeometricAggregation.forward` 的输入契约：

- `pseudo_feature`: `(N_pseudo, C_pseudo)`，用于 Q 和最终残差相加。
- `real_hidden`: `(N_real, C_real)`，用于 K/V 上下文。
- `real_bind_prob`: `(N_real, 1)`，始终 detach 后作为静态先验。
- `real_coord_centered_world`: `(N_real, 3)`，`pseudo_coord_centered_world`: `(N_pseudo, 3)`。
- `real_batch_index`: `(N_real,)`，`pseudo_batch_index`: `(N_pseudo,)`。
- 输出 `(N_pseudo, C_pseudo)`。

`LigandSparseRefineDeltaLoss.forward` 保持兼容参数 `base_logit` / `p_best`，但忽略它们；真实计算只使用 `refined_logit`、`base_prob`、`target`、`valid`、`batch_index` 生成 `{"rank": rank}`。

## Revision Notes

- 2026-06-16 21:50 +08:00: 创建初版计划，记录用户目标、服务器纪律、CPC_main 调用链和第一批静态发现。
- 2026-06-17 02:15 +08:00: 补充最终测试、Trainer smoke、正式训练提交和一小时监测结果，关闭本次 ExecPlan。
