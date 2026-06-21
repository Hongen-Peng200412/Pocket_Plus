# Voxel Point Stage1 Wrapper Refactor 统一勘误与最终变更

更新时间: 2026-05-31

本文只记录对既有 wrapper refactor 计划, 特别是 C:\Users\15919\OneDrive\My_Project\Pocket_Plus\CLAUDE\plans\implement\wrapper_refactor\2026-05-29-voxel-point-stage1-wrapper-refactor-v4.md 的统一勘误、最终实现口径和验证结论。不反向修改旧计划书。

## 1. 阈值与指标语义勘误

`val_uncapped/best/global/p_sampling...` 继续保留为 candidate threshold cache 的写回来源。它来自 dense 全空间 score histogram 的监督式 best-F1 校准: 先找 `p_best`, 再用 best-F1 命中数乘 `adaptive_expand_factor` 得到目标候选数, 最后从 dense histogram 反推 `p_sampling`。它不是从训练 batch 的 `candidate_outputs["candidate_p_sampling_by_class"]` 中取中位数。

`uncapped_sampling_boundary_hist.csv` 表示当前 validation 中 candidate builder 实际使用的 per-BOX sampling boundary 分布诊断。它是本地 artifact, 不写回 cache, 不能替代 `val_uncapped/best/global/p_sampling...`。不再额外保存 `p_sampling_p5/p50/p75/p95/mean` scalar, AI agent 可从 histogram CSV 自动推导这些摘要。

因此本次没有重命名运行时/cache 的公开字段, 仍保留:

* `candidate_p_sampling_by_class`
* `voxel_ligand_p_sampling_by_class`
* `val_uncapped/best/global/p_sampling...`

## 2. by_source 功能降级

本次彻底删除 Stage1 wrapper diagnostics/logging/metric 路径中的 by_source 功能与残留:

* 删除 `SourceFolderRegistry`
* 删除 `source_folder_idx` 在 validation metric 与 diagnostics 中的传播
* 删除 `by_source_folder` metric scope
* 删除 `validation_diagnostics.source_folder_breakdown/source_folder_names/source_folder_meta_key`
* `build_metric_key(...)` 只允许 `scope="global"`

保留 dataset 层原本存在的 `class_name`、`class_folder_names` 等数据语义字段, 因为它们仍属于数据集自身元信息, 不等同于 wrapper diagnostics by_source 功能。

## 3. Histogram artifact 口径

不再新增计划中提到的 `p_C_p5/p50/p75/p95/mean` scalar。最终采用两个本地 CSV histogram artifact:

* `uncapped_sampling_boundary_hist.csv`: 按 candidate class 统计 `candidate_p_sampling_by_class` 的实际 per-BOX sampling cutoff 分布。
* `capped_routed_prob_hist.csv`: 按最终 routed `candidate_class` 统计实际进入唯一候选集 C 的 `candidate_prob` 分布。

写出位置:

`validation_diagnostics/epoch_xxxxxx/histograms/*.csv`

CSV 列:

`class_name,class_pos,class_id,bin_index,bin_left,bin_right,count`

## 4. 配置显式开关

`configs/experiment` 直接管辖的每个配置都显式写出 `model.validation_diagnostics.enabled`:

* `MINI_sparse_refine000.yaml` 到 `MINI_sparse_refine006.yaml`: `true`
* `sparse_refine000.yaml`: `true`
* `emb_unet.yaml`, `unet000.yaml`, `unet_c1.yaml`, `unet_c2.yaml`: `false`

`configs/model/default.yaml` 保留 diagnostics 默认结构, 但删除 by_source 相关配置项。

## 5. Smoke 过程中发现并修复的额外训练阻断

服务器 try_lock smoke 暴露了两个计划书未覆盖、但会阻断当前配置训练的问题:

1. `cfg.model.name` 和 `cfg.model.monitor_mode` 会由 Hydra 传给 `VoxelPointStage1Wrapper.__init__`, 但 wrapper 原签名不接收这些字段, 导致模型实例化失败。已在 wrapper 签名中接收并保存为元信息。
2. `ligand_sparse_refine_loss_schedule.warmup_steps: null` 是配置允许的形式, 但 thin wrapper 原实现直接 `int(None)`。已补回按 `trainer.estimated_stepping_batches * warmup_ratio` 解析的逻辑, 并新增单测覆盖。

另一个 smoke 特有现象是: tiny split 太小时 `train.scheduler.warmup_ratio` 会四舍五入为 0, 对 `recorded_threshold` 不适合作为 smoke 设置。最终 smoke 通过显式覆盖 `train.scheduler.total_steps=100` 与 `train.scheduler.warmup_steps=100` 模拟完整训练早期 warmup; 正式 MINI 配置的数据量下, 原 `warmup_ratio=0.06` 会产生正的 warmup 步数, 第一轮 validation 可以生成 threshold cache。

## 6. 验证记录

本地验证:

* `compileall` 通过, 使用 `PYTHONPYCACHEPREFIX` 避免 OneDrive `__pycache__` 权限问题。
* 目标测试通过: `45 passed`。
* `configs/experiment/*.yaml` 直接配置 Hydra compose 全部通过。
* by_source/source_folder 残留搜索无命中。
* `git diff --check` 无空白错误, 仅有 Git 换行提示。

服务器 smoke:

* 使用已申请的 Slurm/try_lock 资源: job `289428`, `run_cmd_289428.sh`, `try_lock_289428`。
* 临时代码副本: `/home/penghongen/My_Project/tmp/codex_wrapper_smoke_20260531_001505/unpack/Pocket_Plus`
* 7 个配置均完成 1 epoch tiny smoke, 且两个 histogram artifact 均存在:
  * `MINI_sparse_refine000`
  * `MINI_sparse_refine001`
  * `MINI_sparse_refine002`
  * `MINI_sparse_refine003`
  * `MINI_sparse_refine004`
  * `MINI_sparse_refine005`
  * `MINI_sparse_refine006`

服务器 summary:

`/home/penghongen/My_Project/tmp/codex_wrapper_smoke_20260531_001505/smoke_summary.tsv`

所有 7 行均为 `status=ok`, `exit_code=0`, `uncapped_hist=ok`, `capped_hist=ok`。测试完成后已删除 `after_lock_289428` 释放 A100 节点。
