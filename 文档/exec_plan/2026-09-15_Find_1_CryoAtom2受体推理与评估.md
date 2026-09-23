# Find_1 CryoAtom2 受体推理与评估执行记录

本文记录 `Find_1` PDB-centric-v2 模型使用 CryoAtom2 最终预测受体时的 Stage1 calibration、F1 basic 与 F2 Gaussian 选参、`test_0` 评估、`test_1` 保序派生和 calibration Gaussian `score/selected` 回填。真实受体实验见 `文档/exec_plan/2026-09-12_Find_1真实受体推理与评估.md`；两种受体条件的共同状态见 `文档/exec_plan/2026-09-15_Find_1不同受体条件推理与评估总日志.md`。

## 当前状态

- 2026-09-15：任务开始。CryoAtom2 calibration 100 项和 `test_0` 179 项最终 CIF 已由 AdaLigand 独立验收。
- Job `368455` 使用 H100×2、64 CPU、节点 `hnode01`；第 14 次执行完成正式适配，第 15 次执行完成全量只读门控，第 16 次执行完成 Stage1 calibration 与 `test_0`，第 17 次执行完成 `test_1` 保序派生，第 18 次执行完成最终只读门控。Job 已停回 `try_lock_368455`，`after_lock_368455` 始终保留。
- 受体适配、calibration 选参、calibration Gaussian score-only 回填、179-PDB `test_0` 正式评估与 149-PDB `test_1` 保序派生均已完成并通过门控。
- 16:05 状态快照：calibration probability 已完成 34/100；两个分片各占约 73.9 GiB 显存，双 H100 利用率为 99%–100%，最近完成标记持续增加。F1/F2 语义阈值、centered、调参和测试尚未开始。
- 17:36 状态快照：calibration probability 已完成 83/100；两个分片仍各占约 73.9 GiB 显存，双 H100 利用率均为 100%，最近完成标记持续增加。其余阶段按顺序等待该阶段收齐。
- 18:37 状态快照：calibration probability、F1 blobs 与 F2 blobs 均为 100/100；F2 centered 为 75/100。F1 语义阈值为 `0.7274169921875`，F1 basic 参数为 `score_threshold=0.75816810131073`、`prefiltered_min_voxel=8`、`min_voxels=20`；F2 语义阈值为 `0.527618408203125`。centered 的一个分片已经完成，另一个分片仍以约 44.6 GiB 显存和约 90% GPU 利用率处理剩余较大候选。
- 19:38 状态快照：calibration 的 probability、F1/F2 blobs、F2 centered 与 Gaussian score-only 均为 100/100。Gaussian 参数冻结为 `score_threshold=0.9171097278594971`、`prefiltered_min_voxel=8`、`min_voxels=28`、`tau_angstrom=0.75`、`lambda_positive=0.2816`、`lambda_negative=0.0`；100 份 `F2_centered.npz` 均含形状对齐的 `score/selected`。`test_0` probability 已完成 30/179。
- 20:42 状态快照：`test_0` probability 已完成 69/179；双 H100 各占约 73.9 GiB 显存，利用率均为 100%，最近完成标记持续增加。锁、进程和错误日志均无异常。
- 22:12 状态快照：`test_0` probability 已完成 122/179；双 H100 仍各占约 73.9 GiB 显存，利用率均为 100%，最近完成标记持续增加。F1/F2 测试后处理尚未开始。
- 23:43 状态快照：`test_0` probability 已完成 159/179；一个分片已经完成，另一个分片仍占约 73.9 GiB 显存并保持 100% GPU 利用率，继续处理最后 20 个较大样本。后处理尚未开始，错误日志无新增异常。
- 2026-09-16 00:00 状态快照：`test_0` probability 已完成 165/179；剩余分片仍占约 73.9 GiB 显存并保持 100% GPU 利用率，最近完成标记持续增加。日期续接只中断本地睡眠句柄，没有影响服务器任务。
- 01:01 状态快照：`test_0` probability 与 F1 blobs 均为 179/179；F1 basic 逐 PDB evaluate 已生成 52/179 份 NPZ，单个 CPU 进程持续运行。F2 blobs 与 centered 尚未开始。
- 02:02 状态快照：第 16 次执行成功结束。`test_0` 的 probability、F1/F2 blobs 与 F2 centered 均为 179/179；两套逐 PDB evaluation NPZ 均为 179/179，两套全局 metrics JSON 与 179 行 JSONL 已生成。Job 随后进入第 17 次 `test_1` 派生执行。
- 02:07 关键事件：第 17 次执行成功结束。两套 `test_1` JSONL 均为 149 行，成员和顺序与冻结清单一致；该目录只含 metrics、JSONL 与 provenance，没有重复生成重型推理产物。
- 02:17 关键事件：第 18 次最终只读门控通过。calibration 100 项、`test_0` 179 项、`test_1` 149 项、358 份 `test_0` 逐 PDB evaluation NPZ 和 100 份 calibration scored-centered 均符合契约；Job 随后停回 `try_lock_368455`，`after_lock_368455` 保留。

## 最终参数与主要指标

| 项目 | F1 blobs+basic | F2 blobs+Gaussian |
| --- | ---: | ---: |
| calibration 语义阈值 | 0.7274169921875 | 0.527618408203125 |
| `score_threshold` | 0.75816810131073 | 0.9171097278594971 |
| `min_voxels` | 20 | 28 |
| `test_1` semantic micro F1 | 0.5406947426509066 | 0.5266402130264328 |
| `test_1` semantic macro F1 | 0.49038813561501443 | 0.4800930116407475 |
| `test_1` coverage micro F1@0.3 | 0.6303394135303625 | 0.6167108107920222 |
| `test_1` coverage macro F1@0.3 | 0.5917646757614797 | 0.5800323416959737 |
| `test_1` one-to-one micro F1@0.3 | 0.621160409556314 | 0.608187134502924 |
| `test_1` one-to-one macro F1@0.3 | 0.5787682512533054 | 0.5675483403564443 |

完整 P/R/F1/PRAUC 和 top-K 指标见 AdaLigand `收口の结果/Stage1/Find_1(pdb_centric_v2)/使用cryoatom2预测的受体/`。

## 冻结范围

- checkpoint：W&B 已完成步编号 `38622` 对应的 `TOP_epoch_04_score_0.6654.ckpt`。
- calibration：100 个 PDB；用于独立选择 F1/F2 语义阈值、basic 参数和 Gaussian 参数。
- `test_0`：179 个 PDB；执行一次完整推理与两套正式评估。
- `test_1`：`test_0` 的 149-PDB 保序子集；只从逐 PDB 评估事实派生，不重复模型前向。
- 受体：每个 PDB 使用 CryoAtom2 `latest.json` 指向的最终 CIF，不使用 `_raw.cif`。

## 正式产物

```text
/storage/penghongen/AdaLigand_stage1_inference/Find_1/CryoAtom2受体/
└── artifacts/Find_1/
    ├── calibration/
    ├── tuning/
    ├── held_out_test_0/
    └── held_out_test_1/
```

正式适配数据分别写入：

- `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/Ori_Data`
- `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data`

## 正式运行命令

受体适配，Job `368455` 第 13 次执行：

```bash
exec bash "/home/penghongen/Feedback/AdaLigand/releases/AdaLigand_fe7e6d0cd64c/AdaLigand/测评数据代码/cryoatom2_受体Adapter/sh/prepare.sh"
```

这条正式命令因 Python 环境缺少 `gemmi` 在模块导入时结束，未写入正式受体资产。后续第 14 次执行使用修复后的 release 完成重跑。本节不把测试、门控或只读检查命令混入正式运行命令。

受体适配，Job `368455` 第 14 次执行：

```bash
exec bash "/home/penghongen/Feedback/AdaLigand/releases/AdaLigand_c209bedb1883/AdaLigand/测评数据代码/cryoatom2_受体Adapter/sh/prepare.sh"
```

这是环境修复后的正式重跑命令。

Stage1 calibration、`test_0` 与 calibration Gaussian score-only 回填，Job `368455` 第 16 次执行：

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/find1_cryoatom2_receptor_evaluation.sh"
```

`test_1` 保序派生，Job `368455` 第 17 次执行：

```bash
exec bash "${TASK_PROJECT_ROOT}/tmp/find1_cryoatom2_evaluation_20260915/derive_test1.sh"
```

## 门控与测试

- 服务器单元测试：`2 passed`。
- Job `368455` 第 10 次执行使用隔离门控，固定重建 calibration 清单前三项。`6bgi` 含 7,770 个受体重原子和 `262³` 密度网格，`6dqn` 含 69,380 个受体重原子和 `418³` 密度网格，`6rec` 含 53,603 个受体重原子和 `506³` 密度网格。三者的十个 `receptor_tokens` 数组均逐项相同，`sim.npy` 均逐位相同，最大绝对误差均为 `0.0`。
- 门控结果：`/storage/penghongen/tmp/find1_cryoatom2_adapter_gate_20260915/Find_1_pdb_centric_2_job368455_20260915T114442_a10/gate_result.json`。
- Job `368455` 第 11 次执行覆盖 calibration 100 项与 `test_0` 179 项共 279 个唯一真实受体。十个 token 数组全部逐项相同，279 份 `sim.npy` 全部逐位相同，全局最大绝对误差为 `0.0`。
- 全量门控结果：`/storage/penghongen/tmp/find1_cryoatom2_adapter_gate_20260915/Find_1_pdb_centric_2_job368455_20260915T120839_a11/gate_result.json`。
- Job `368455` 第 12 次执行以 CryoAtom2 `6bgi` 验证新数据根与正式 Stage1 probability 入口的连接。适配后的模拟密度为 `(1, 262, 262, 262)`，输出 probability map 为 `(262, 262, 262)`，所有数值有限；受体链接指向 `latest.json::final_cif`，不指向 `_raw.cif`。
- Job `368455` 第 15 次执行完整检查 calibration 100 项与 `test_0` 179 项。记录顺序、三件套、十字段 token、有限值、实验网格几何、最终 CIF 来源和主真值符号链接全部通过；报告位于 `/storage/penghongen/tmp/find1_cryoatom2_adapter_formal_gate_20260915/`。
- Job `368455` 第 18 次执行完成最终只读门控。报告 `/storage/penghongen/tmp/find1_cryoatom2_final_gate_20260916/result.json` 的状态为 `passed`；`test_1` 与父 `test_0` 逐 PDB 事实等价，所有 P/R、F1、PRAUC 和比例字段合法，micro/macro 聚合与 top-K 计数均可重建。

第 10 次 Job 执行的门控命令：

```bash
exec env PYTHONPATH="/storage/penghongen/tmp/find1_cryoatom2_adapter_candidate_20260915/测评数据代码/cryoatom2_受体Adapter:/home/penghongen/My_Project/AdaLigand/Data_Preprocessing/Ori_Data" /home/penghongen/anaconda3/envs/AdaLigand_stage1_py310/bin/python -u /storage/penghongen/tmp/find1_cryoatom2_adapter_candidate_20260915/gate_known_receptors.py
```

这是门控命令，不是正式运行命令；它不写入正式 CryoAtom2 产物根。第 11 次执行复用同一命令完成全部 279 项门控。

## 运行历史

失败或中断尝试统一追加在本节，不另建实验日志。

- 第 9 次 Job 执行只启动了一次性等价门控。脚手架把 calibration 顶层列表误当为含 `pdb_ids` 的对象，在任何 Chimera 调用前以 `TypeError` 结束。只修正该门控读取逻辑后，第 10 次执行通过；正式适配代码原本已支持两种 JSON 顶层格式。
- 第 12 次 Job 执行完成了 `6bgi` 适配与 probability 前向，最后的门控断言却读取了不存在的 `probability` 字段，因此返回非零。实际契约字段为 `probability_map`；修正后的只读验收通过，生产代码和前向产物无需修改。
- 第 13 次 Job 执行从首个冻结 AdaLigand release 启动正式适配，但入口继承了不含 `gemmi` 的 Pocket Plus Python，因而在模块导入时结束。两个正式目标仍为 0 份适配资产；窄修复仅把入口固定为已经通过测试的 `AdaLigand_stage1_py310` Python，服务器定向测试仍为 `2 passed`。

## 计划与实现差异

- 中性差异：正式适配代码不保存或校验文件哈希；哈希只出现在隔离门控和追溯记录中。
- 中性差异：`test_1` 按既定方案从 `test_0` 逐 PDB 事实保序派生，不重新执行 probability、blobs、centered 或 evaluate。
- 当前没有未完成范围。

## 2026-09-17 F2–F1–basic 方法消融

### 状态快照

- 新方法复用 CryoAtom2 受体实验已冻结的 `F2_semantic.json`、100 个 calibration F2 blobs、179 个 `test_0` F2 blobs、两批 F2 centered 和共享 probability，不重新执行 GPU 前向，也不重新拟合语义阈值。
- 只读候选轴门控已覆盖 calibration 100 个 PDB 和 `test_0` 179 个 PDB。`voxel_count >= 8` 的 F2 blob 分别为 2,678 和 2,407 个；来源 blob 编号、完整来源体素数和 `source_probability_mean` 均与 centered 候选轴逐项完全相同。
- Job `385002` 已提交到 `cpu` partition，QOS=`Cpu96`，申请 8 CPU；`pre_hold=0`、`after_hold=0`。2026-09-17 当前状态为 `PENDING (Priority)`，尚未创建 release、launch 或正式结果。一次任务配置把 `calibration.workers` 固定为 8；体素门槛搜索网格、coverage 阈值轴和 top-K 轴与正式 `stage1_v3.yaml` 相同。
- 首次提交的 Job `384992` 请求 64 CPU，因单节点必须同时空闲 64 核而长时间排队。用户要求改为 8 CPU 后，该 Job 在获得节点前取消，终态为 `CANCELLED by 1351`、运行时间 `00:00:00`；没有创建 release、launch、预映像或正式产物。
- 本次唯一允许新增的正式结果为 `tuning/F2_basic.json`、179 份 `f2_centered_basic_macro_selected.npz`、`test_0` 的 metrics/JSONL，以及派生 `test_1` 的 metrics/JSONL/provenance。运行前后会以文件大小和纳秒修改时间核对既有 calibration、tuning、`test_0` 与 `test_1` 文件未被修改。
- 2026-09-17 23:27，Job `385002` 在 cnode01 启动。release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_10765b4330e6/Pocket_Plus`，launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/385002/run_f2_basic_job385002_20260917T232747_a1`；8 CPU、无 GPU，运行前既有文件快照包含 3,443 个受保护文件。
- calibration 已冻结独立 F2-basic 参数：`score_threshold=0.7009013295173645`、`prefiltered_min_voxel=8`、`min_voxels=26`、`objective_beta=1`、`score_mode=basic`。23:42 状态快照为 `test_0` 逐 PDB evaluation NPZ 179/179，evaluate 已处理完整清单并进入全局汇总或 `test_1` 派生交界，未出现 traceback。

### 正式运行命令

```bash
exec bash "${TASK_PROJECT_ROOT}/tmp/find1_f2_basic_ablation_20260917/run_f2_basic.sh" cryoatom2
```

该命令依次通过 `训练与运行/sh/infer/stage1_v3.sh` 执行 `tune` 与 `evaluate`，随后从新产生的 179-PDB `test_0` 逐 PDB事实保序派生 149-PDB `test_1`。调参固定 `alpha=2`、`score_mode=basic`、`objective_beta=1`、`prefiltered_min_voxel=8`；测试固定 `artifact=centered` 和评估名 `f2_centered_basic_macro_selected`。

### 提交命令

```bash
bash 训练与运行/submit_task.sh --sh tmp/find1_f2_basic_ablation_20260917/run_f2_basic.sh --resource cpu --cpus 8 --job-name F2B_cryo -- cryoatom2
```

### 门控命令

候选轴门控使用服务器共享项目中的以下临时脚本；该命令只读正式产物，不属于正式运行命令。

```bash
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python -u /home/penghongen/My_Project/Pocket_Plus/tmp/find1_f2_basic_ablation_20260917/gate_candidate_axes.py
```

### 23:43 关键事件与恢复命令

Job `385002` 已成功写出独立 `F2_basic.json`、179 份 `test_0` 逐 PDB评估、metrics JSON 和 JSONL。随后，一次任务脚本在正式 `stage1_v3.sh` 返回后使用了未固定环境的 `python`，`derive_test1.py` 因该环境缺少 `scipy` 而退出。失败未影响已经生成的调参与 `test_0` 结果，也没有执行任何 GPU 前向。

恢复任务只派生 149-PDB `test_1` 并执行既有文件保护校验，不重复调参或 `test_0` 评估：

```bash
exec bash "${TASK_PROJECT_ROOT}/tmp/find1_f2_basic_ablation_20260917/resume_after_test0.sh" cryoatom2
```

该命令是本次中断后的正式恢复命令。窄修复只把一次任务脚本的 Python 解释器固定为 Pocket Plus 正式环境；生产推理代码、科学参数和已有正式产物均不改变。

恢复 Job `385054` 已按 `cpu` partition、QOS=`Cpu96`、8 CPU 提交，`pre_hold=0`、`after_hold=0`。提交前已经用固定解释器完成 `scipy` 与正式聚合模块导入门控，并通过恢复脚本语法检查。

```bash
bash 训练与运行/submit_task.sh --sh tmp/find1_f2_basic_ablation_20260917/resume_after_test0.sh --resource cpu --cpus 8 --job-name F2B_cryo_r -- cryoatom2
```

### 完成与验收

- Job `385054` 于 2026-09-17 23:55 成功完成：`test_1` JSONL 为 149 行，保护校验报告 `protected_count=3443`、`new_count=185`、`status=PASS`。
- 最终只读总门控通过清单顺序、`test_1` 精确子集关系、P/R–F1、macro 平均、top-K 与 provenance 验收。
- `test_0` semantic micro/macro F1 为 `0.5596455116747411/0.48126304754496363`；`test_1` 为 `0.535072500247211/0.48769792005280893`。
- 正式结果和完整指标已增量写入 AdaLigand 本地目录 `收口の结果/Stage1/Find_1(pdb_centric_v2)/使用cryoatom2预测的受体/` 的三份文档，仍保持未提交且未上传服务器。
