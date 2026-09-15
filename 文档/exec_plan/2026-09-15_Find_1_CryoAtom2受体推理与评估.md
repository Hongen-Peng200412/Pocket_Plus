# Find_1 CryoAtom2 受体推理与评估执行记录

本文记录 `Find_1` PDB-centric-v2 模型使用 CryoAtom2 最终预测受体时的 Stage1 calibration、F1 basic 与 F2 Gaussian 选参、`test_0` 评估、`test_1` 保序派生和 calibration Gaussian `score/selected` 回填。真实受体实验见 `文档/exec_plan/2026-09-12_Find_1真实受体推理与评估.md`；两种受体条件的共同状态见 `文档/exec_plan/2026-09-15_Find_1不同受体条件推理与评估总日志.md`。

## 当前状态

- 2026-09-15：任务开始。CryoAtom2 calibration 100 项和 `test_0` 179 项最终 CIF 已由 AdaLigand 独立验收。
- Job `368455` 现已回到 `try_lock_368455`，H100×2、64 CPU、节点 `hnode01`；`after_lock_368455` 始终保留。
- 受体适配候选入口已通过服务器 AdaLigand 单元测试和 279-PDB 真实受体全量等价门控；正式 CryoAtom2 适配和推理尚未启动。

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

尚未执行。适配和推理命令将在冻结 release 并核对动态命令后分别补入；本节不记录测试、门控或只读检查命令。

## 门控与测试

- 服务器单元测试：`2 passed`。
- Job `368455` 第 10 次执行使用隔离门控，固定重建 calibration 清单前三项。`6bgi` 含 7,770 个受体重原子和 `262³` 密度网格，`6dqn` 含 69,380 个受体重原子和 `418³` 密度网格，`6rec` 含 53,603 个受体重原子和 `506³` 密度网格。三者的十个 `receptor_tokens` 数组均逐项相同，`sim.npy` 均逐位相同，最大绝对误差均为 `0.0`。
- 门控结果：`/storage/penghongen/tmp/find1_cryoatom2_adapter_gate_20260915/Find_1_pdb_centric_2_job368455_20260915T114442_a10/gate_result.json`。
- Job `368455` 第 11 次执行覆盖 calibration 100 项与 `test_0` 179 项共 279 个唯一真实受体。十个 token 数组全部逐项相同，279 份 `sim.npy` 全部逐位相同，全局最大绝对误差为 `0.0`。
- 全量门控结果：`/storage/penghongen/tmp/find1_cryoatom2_adapter_gate_20260915/Find_1_pdb_centric_2_job368455_20260915T120839_a11/gate_result.json`。

第 10 次 Job 执行的门控命令：

```bash
exec env PYTHONPATH="/storage/penghongen/tmp/find1_cryoatom2_adapter_candidate_20260915/测评数据代码/cryoatom2_受体Adapter:/home/penghongen/My_Project/AdaLigand/Data_Preprocessing/Ori_Data" /home/penghongen/anaconda3/envs/AdaLigand_stage1_py310/bin/python -u /storage/penghongen/tmp/find1_cryoatom2_adapter_candidate_20260915/gate_known_receptors.py
```

这是门控命令，不是正式运行命令；它不写入正式 CryoAtom2 产物根。第 11 次执行复用同一命令完成全部 279 项门控。

## 运行历史

失败或中断尝试统一追加在本节，不另建实验日志。

- 第 9 次 Job 执行只启动了一次性等价门控。脚手架把 calibration 顶层列表误当为含 `pdb_ids` 的对象，在任何 Chimera 调用前以 `TypeError` 结束。只修正该门控读取逻辑后，第 10 次执行通过；正式适配代码原本已支持两种 JSON 顶层格式。

## 计划与实现差异

- 中性差异：正式适配代码不保存或校验文件哈希；哈希只出现在隔离门控和追溯记录中。
