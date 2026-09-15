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
