# unet_c1 pdb-centric-v2 F2-F1-basic 消融执行记录

本文记录正式消融模型 `unet_c1` pdb-centric-v2 的 F2-F1 basic 方法级消融。现有三模型原始比较继续使用 F1-F1：第一个 F1 表示在 100-PDB calibration 上最大化 PDB 等权 macro semantic F1 以选择语义概率阈值；第二个 F1 表示以 `objective_beta=1` 最大化 macro semantic F1、coverage F1@0.3 与 one-to-one F1@0.3 之和。本任务只把前一阶段改为 semantic F2，后一阶段定义不变。

## 状态快照

- 2026-09-20 20:51 +08:00：本地与服务器只读门控通过。服务器正式根中预期 F2 目标为 0 个；calibration 与 `test_0` probability 分别完整覆盖 100/100 和 179/179 个 PDB；既有 `F1_semantic.json`、`F1_semantic_scan.npz`、`F1_basic.json`、179 份 F1 逐 PDB evaluation 及两个 F1 全局汇总均存在。
- Job `378693` 的资源身份为 gnode10、A800×1、16 CPU；当前停在 `try_lock_378693`，没有活跃载荷。`after_lock_378693` 在最终结果与文档全部验收前必须保留。
- 本任务不运行 probability、centered 或任何 GPU 前向；只用既有 probability 执行 F2 semantic 拟合、F2 blobs、basic 选参、`test_0` evaluate 与 `test_1` 保序派生。
- 2026-09-20 20:56 +08:00：服务器正式门控再次确认 CPU 配置、745 个目标缺失及锁状态后，原子改写 Job `378693` 的动态命令并只删除 `try_lock_378693`。第 29 次执行冻结 release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_49db579cfe2c/Pocket_Plus`，建立 launch `/home/penghongen/Feedback/Pocket_Plus/launches/378693/Find_1_job378693_20260920T205610_a29`；`after_lock_378693` 保持存在。20:57 的启动核查显示正式命令身份正确，流程正在计算既有 F1 内容摘要，尚未写入 F2 目标。
- 2026-09-20 21:13 +08:00：第 29 次执行成功，Job 回到 `try_lock_378693`。正式入口生成预期的 745 个新文件；calibration/test_0 的 F2 blobs 为 100/179 份，`test_0` 逐 PDB evaluation 为 179 份，`test_1` 三个全局文件完整。保护复核确认 1,861 个既有文件的大小和纳秒修改时间未变，466 个既有 F1 文件的内容摘要未变。
- 2026-09-20 21:31 +08:00：独立只读门控再次通过。F2 semantic threshold 为 `0.23468017578125`；F2 basic 参数为 `score_threshold=0.40040671825408936`、`prefiltered_min_voxel=8`、`min_voxels=26`。`test_0/test_1` 汇总分别严格包含 179/149 个 PDB，后者与 `test_1.json` 顺序一致。
- 2026-09-20 21:44 +08:00：本地两份结果文档完成确定性来源门控。服务器文档来源 JSON 的 SHA-256 为 `f0673a7343bcc00572264556aa91faecfa5905fead92d59c3dd90cb28c59a4ad`；本地 `说明.md` 与 `结果.md` 的 SHA-256 分别为 `b13ef70a7cfed1c192e4339c6bee85339bac70a19a83c19b930379142361ca8e` 与 `a33c85fec80a26762563bd0406519664ba88b56a1365d56156e71c434426d37a`，与服务器确定性生成值完全一致。两份文档保持 unstaged，未上传服务器。
- 2026-09-20 22:25 +08:00：完成态记录落盘后，释放脚本再次核对 Job `378693` 正在 gnode10 使用 16 CPU、`try_lock_378693` 与 `after_lock_378693` 均存在、第 29 次执行成功且正式输出含最终 PASS；随后只删除 `after_lock_378693`。8 秒后的核查确认 Job 已不在 `squeue`，两把活动锁均不存在，allocation 日志明确记录“Job 378693 已退出并清理活动锁与动态命令”。A800 已安全释放，未触碰其他任务。

## 输入与输出

- calibration 清单：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split/pdb_split/calibration.json`，100 个 PDB。
- `test_0` 清单：`/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json`，179 个 PDB。
- `test_1` 清单：`/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_1.json`，149 个 PDB；它是 `test_0` 的保序派生子集，不是独立推理或独立统计样本。
- 正式结果根：`/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts/unet_c1`。
- 一次性任务实现：`tmp/unet_c1_pdb_centric_v2_f2_basic_20260920/`。编排只调用 `训练与运行/sh/infer/stage1_v3.sh` 与 `src.inference.cli` 的 blobs、tune 和 evaluate 官方入口；`test_1` 聚合复用已经验收的正式聚合函数。

## 冻结科学配置

- semantic alpha：2；阈值网格分母：32768。
- basic `objective_beta=1`，`prefiltered_min_voxel=8`；basic 分数严格为 `source_probability_mean`。
- `min_voxel_values=[8, 9, ..., 40]`。
- coverage 阈值为 `[0.3, 0.5, 0.6]`；top-K 为 `[3, 4, 5]`。
- `blob_workers=16`、`calibration.workers=16`，与 Job `378693` 的 CPU 数一致。
- 测试评估名固定为 `f2_blobs_basic_macro_selected`；测试使用 `artifact=blobs`，不读取 centered。

## 正式运行命令

本任务的正式运行命令只有：

```bash
exec bash "${TASK_PROJECT_ROOT}/tmp/unet_c1_pdb_centric_v2_f2_basic_20260920/run_f2_basic.sh"
```

该入口内部按顺序执行 calibration F2 blobs 拟合、F2 basic 调参、`test_0` F2 blobs、`test_0` evaluate、`test_1` 派生、既有产物保护核对和最终只读验收。它不接受 `--overwrite`。

## 门控与只读核查命令

下列命令不是正式运行命令：

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' -Command 'bash -s' -InputFile '.\tmp\unet_c1_pdb_centric_v2_f2_basic_20260920\preflight_probe.sh'
```

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' -Command 'bash -s' -InputFile '.\tmp\unet_c1_pdb_centric_v2_f2_basic_20260920\remote_gate.sh'
```

保护脚本在正式入口最前面再次要求全部 745 个预期新增文件尚不存在，并记录除这些目标外全部既有文件的大小与纳秒修改时间；同时对全部 F1 参数、blobs、状态和评估文件计算内容摘要。正式流程结束后，保护脚本要求既有文件元数据与 F1 内容摘要完全不变，并要求 745 个新增文件全部存在。

## 文档边界

结果完成后，只增量修改本地 `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\pdb_centric_v2\说明.md` 与 `结果.md`。两份文档保持 unstaged，不上传服务器、不提交 Git；三模型原始比较统一采用 F1-F1 的历史事实必须保留。

## 正式结果与验收

### 冻结参数

| 参数 | F1-F1 | F2-F1 |
| --- | ---: | ---: |
| semantic threshold | `0.3990478515625` | `0.23468017578125` |
| score threshold | `0.5232318639755249` | `0.40040671825408936` |
| prefiltered min voxel | `8` | `8` |
| min voxels | `21` | `26` |

F2-F1 在 `test_0` 的 semantic macro F1 为 `0.39998279778612406`，F1-F1 为 `0.39443718585894233`；在 `test_1` 分别为 `0.41118686152124473` 与 `0.40469921989956786`。F2-F1 在 `test_0` 的 one-to-one macro F1@0.3 为 `0.47245157959789164`，F1-F1 为 `0.4781325170565899`；在 `test_1` 分别为 `0.492032688664956` 与 `0.48686592785119787`。两种方法复用完全相同的 probability，因此 `test_0` semantic macro PRAUC 均为 `0.4016180431955654`，这符合预期。

完整的 micro/macro P、R、F1、PRAUC、三档实例阈值、top-3/top-4/top-5、固定分层与 Spearman 描述性分析均写入本地 `结果.md`。结果只支持描述本次固定样本上的差异，不报告因果关系或显著性结论。

### 新增全局文件摘要

| 文件 | SHA-256 |
| --- | --- |
| `held_out_test_0/evaluation/f2_blobs_basic_macro_selected.metrics.json` | `503cdb50ee25a2db8fff365700fba6845378161b61499f9322191e58c2180665` |
| `held_out_test_0/evaluation/f2_blobs_basic_macro_selected.jsonl` | `e964d917b75cfff5de9bc4ef0b1055a9830e2d7e09a31fc1ebc9ccc1ca0959f7` |
| `held_out_test_1/evaluation/f2_blobs_basic_macro_selected.metrics.json` | `622205b6a7d0e2e2bed50b015f0ece95259521528049ffebea83be9c87791754` |
| `held_out_test_1/evaluation/f2_blobs_basic_macro_selected.jsonl` | `d7c19167a3a96ef82718567f311d5039245d0720967dcaa68e3a76e0724d8ce1` |
| `held_out_test_1/evaluation/f2_blobs_basic_macro_selected.provenance.json` | `49c24d44e9da1547e0ffc0eeafbd9190508cae5e4a5c2b51d0dd1e8607f54024` |

既有 F1 `test_0` metrics/JSONL 摘要仍为 `4b216227a8b9d8b528afd8d610ddf48c4bafb95bfdaa71471ec647dedbaf765e` / `2f62019c6b658df626098dcc138776859bf08b7d550986acbcfe8caf5a709309`；既有 F1 `test_1` metrics/JSONL/provenance 摘要仍为 `d49177c7620ef7e5a043d626e499c0be089d9fa2a2d65d24b0a4290c7a873b88` / `7218a789db3527dfb2cde1268c70299893dffc599dae91cf02ed8ebe23717823` / `a1ce1912a79603411e10d3603143736fe408b1316226d25310cc5f30c8fde9f6`。

### 计划与实现差异

- 有益差异：无。正式流程完全沿用既有官方入口和冻结配置。
- 中性差异：服务器只读文档来源与本地文档由一次性脚本确定性生成，便于逐值核对；它们不参与正式科学计算，也未上传正式结果根。
- 有害差异：无。
- 未完成范围：无。服务器计算、保护复核、本地文档和资源释放均按本记录的最终状态闭合。
