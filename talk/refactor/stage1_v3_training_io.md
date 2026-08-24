# Stage1 V3 训练 I/O 整理清单

本文以 `8ff69abd608192d85686f73da2d1f9f15fe67a1e` 为整理基点，逐文件记录 Stage1 V3 训练 I/O 实现、旧代码删除和文档更新。正式训练与后续推理只调用一套 `Stage1Dataset` 裁块逻辑；历史实现仅由 Git 保存。

## 统一决策

- 完整体数组改从 `exp.npy`、`sim.npy`、`union_mask.npy` 和 `ligand_dist.npy` 内存映射，NPZ 只保留小型元数据和稀疏字段。
- Dataset 返回 49 维受体基础特征与独立 `is_backbone`；模型边界在需要 50 维时拼接。
- V3 逐 PDB bias/context 几何候选池保持不变。活动训练按 `pdb_foreground_box_num=25`、`pdb_foreground_fraction_target=0.5`、`pdb_occurrence_foreground_box_cap=5` 动态选择；验证冻结相同规则的 epoch 0 到 `validation_selection_pdb_centric.npz`。原 `0:5:5`/`0:1:1` 文件只作为历史构建记录保留。
- 单卡使用 16 CPU/16 workers；双卡总计 32 CPU，每个 DDP rank 16 workers。
- DataLoader 使用 `prefetch_factor=4` 和 `persistent_workers=false`。
- U-Net 主链版保留三类结构头和 0.05/0.05/0.3 损失；无主链版保留结构头，把前两项权重设为 0，双卡使用 unused-parameter 检查兼容配置。
- 本轮只准备代码和一键入口；训练启动必须等待用户另行明确授权。

## 逐文件清单

“当前职责”指基点提交中的职责；新文件写明新增职责。“测试”同时记录本轮用于保护该文件的测试或明确说明其退出测试集。

| 文件 | 当前职责 | 调用者 | 生命周期 | 测试 | 目标位置 | 动作 | 理由 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Bundle_of_Maps/simulated_map/gen_chimera_cmds.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Bundle_of_Maps/simulated_map/get_receptor_from_PDB.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Bundle_of_Maps/simulated_map/notes_of_dataset.md` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `CLAUDE.md` | 当前代码或数据契约说明 | 开发者与运维人员 | 活动文档 | 由 rg、审查与实现交叉核对 | 原位保留 | 修改 | 删除旧 NPZ、旧比例和旧入口陈述 |
| `Docking/analyze_runtime_budget.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/collect_batch_summary.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/__init__.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/config.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/instance_postprocess.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/io_utils.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/matching.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/records.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/rosetta.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/runner.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/docking_pipeline/shape_scoring.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/evaluation/run_evaluation.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/evaluation/run_oracle_evaluation.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/readme.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/run_docking_batch.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/run_docking_sample.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/run_easy20_prescan.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/run_instance_prescan.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/run_oracle_docking_batch.py` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/collect_easy20_summary_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/docking_cpu_batch.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/easy20_prescan_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/evaluation_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/oracle_easy20_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/oracle_evaluation_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/oracle_single_sample_parallel_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/realistic_easy20_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/sbatch/realistic_single_sample_parallel_cpu.sbatch` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接工具.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/7zdf_emerald_id_exec_plan.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/docking_evaluation_exec_plan.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/docking_master_exec_plan.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/docking_metrics_readable.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/docking_ml_scoring_exec_plan.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/full80_docking_exec_plan.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/oracle_easy20_exec_plan.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/oracle_easy20_experiment.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/oracle_easy20_readable.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/progress1.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Docking/分子对接记录/progress1_readable.md` | 旧分子对接实验、脚本或记录 | 已无当前 Stage1 调用者 | 历史代码或历史记录 | 不进入当前 Stage1 测试集 | 仅保留于 Git 历史 | 删除 | 不属于当前 Stage1 训练与后续推理主线 |
| `Make_Data/PDB_processor/__init__.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/config.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/error_logger.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/features/__init__.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/features/atom_features.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/features/residue_features.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/geometry/__init__.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/geometry/coordinate_reconstruction.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/geometry/graph_builder.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/geometry/local_frames.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/ligand_candidates.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/PDB_processor/parser.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/compute_statistics.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/labels/__init__.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/labels/filter_config.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/labels/instance_labels.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/labels/ligand_filter.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/log/v1(在原版基础上分辨率变为1.0A)/process_and_label_err_216527_0.txt` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/log/v1(在原版基础上分辨率变为1.0A)/process_and_label_err_216527_1.txt` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/log/v1(在原版基础上分辨率变为1.0A)/process_and_label_err_216527_2.txt` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/log/v1(在原版基础上分辨率变为1.0A)/process_and_label_out_216527_0.txt` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/log/v1(在原版基础上分辨率变为1.0A)/process_and_label_out_216527_1.txt` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/log/v1(在原版基础上分辨率变为1.0A)/process_and_label_out_216527_2.txt` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/notes_of_dataset.md` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/process_and_label.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/readme.md` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/sbatch/process_and_label_v0.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/sbatch/process_and_label_v1.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/sbatch/process_and_label_v2_mod4.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/sbatch/process_and_label_v2_mod5.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/sbatch/process_and_label_v2_raw4.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/sbatch/process_and_label_v2_raw5.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `Make_Data/split_data/generate_full_json.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `configs/dataset/stage1_find.yaml` | Stage1 Dataset Hydra 配置 | src/train.py | 正式配置 | tests/test_adaligand_stage1_configs.py | 原位保留 | 修改 | 切换 V3 根目录并移除 fraction |
| `configs/dataset/stage1_unet_c1.yaml` | Stage1 Dataset Hydra 配置 | src/train.py | 正式配置 | tests/test_adaligand_stage1_configs.py | 原位保留 | 修改 | 切换 V3 根目录并移除 fraction |
| `configs/experiment/unet_c1.yaml` | U-Net 实验组合配置 | 训练脚本与 Hydra | 正式配置 | tests/test_adaligand_stage1_configs.py | 原位保留 | 修改 | 保留结构头并由脚本设置辅助损失权重 |
| `configs/train/stage1_cpc1.yaml` | Stage1 DataLoader 与训练资源配置 | src/train.py | 正式配置 | tests/test_adaligand_stage1_configs.py | 原位保留 | 修改 | 保持预取 4；Find_1 每 rank 30 workers，其他当前入口每 rank 16 workers |
| `ops/box_pool_2/build_box_pool_2.py` | 第二版 split 或 BOX pool 构建 | 旧数据准备命令 | 历史代码 | 被 V3 数据准备测试取代 | 仅保留于 Git 历史 | 删除 | 正式消费者只读取已验收的 V3 产物 |
| `ops/box_pool_2/build_box_pool_2.sh` | 第二版 split 或 BOX pool 构建 | 旧数据准备命令 | 历史代码 | 被 V3 数据准备测试取代 | 仅保留于 Git 历史 | 删除 | 正式消费者只读取已验收的 V3 产物 |
| `ops/box_pool_2/finalize_box_pool_2.sh` | 第二版 split 或 BOX pool 构建 | 旧数据准备命令 | 历史代码 | 被 V3 数据准备测试取代 | 仅保留于 Git 历史 | 删除 | 正式消费者只读取已验收的 V3 产物 |
| `ops/materialize_filtered_stage1_preparation.py` | 第二版 preparation 物化入口或测试 | 旧运维命令 | 历史代码 | 旧专用测试退出 | 仅保留于 Git 历史 | 删除 | 会重新发布旧路径与旧请求契约 |
| `ops/materialize_filtered_stage1_preparation.sh` | 第二版 preparation 物化入口或测试 | 旧运维命令 | 历史代码 | 旧专用测试退出 | 仅保留于 Git 历史 | 删除 | 会重新发布旧路径与旧请求契约 |
| `ops/stage1_data_preparation/build_box_pool_3.py` | V3 BOX pool 构建入口 | Slurm 数据准备脚本 | 可复用运维代码 | ops/stage1_data_preparation/tests/test_build_box_pool_3.py | 原位保留 | 修改 | 改用同目录 utils 并保持构建契约 |
| `ops/stage1_data_preparation/README.md` | V3 数据准备的字段、路径与运行说明 | 开发者与运维人员 | 活动文档 | 由数据准备测试和执行记录交叉核对 | 原位保留 | 修改 | 补充 utils 布局和统一 Dataset 的训练消费状态 |
| `ops/stage1_data_preparation/EXECUTION.md` | V3 一次性迁移、split 与 BOX pool 执行证据 | 开发者与运维人员 | 执行记录 | 由正式 Job 证据和数据准备测试保护 | 原位保留 | 修改 | 连接后续训练 I/O 适配并保持数据准备范围边界 |
| `processedPDB_EMDB_binder/bind.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/debug/check.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/debug/gpu.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/notes_of_dataset.md` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/readme.md` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v1.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_mod4_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_mod4_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_mod5_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_mod5_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_raw4_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_raw4_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_raw5_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_bind/bind_v2_raw5_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v0.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v1.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_mod4_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_mod4_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_mod5_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_mod5_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_raw4_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_raw4_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_raw5_10A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/sbatch_split/split_and_select_box_CPU_v2_raw5_15A.sbatch` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/split_and_select_box.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/split_data/generate_full_json.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/utils/__init__.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/utils/mrc_tools.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `processedPDB_EMDB_binder/utils/network_tools.py` | 旧数据生成或绑定实现 | 已无活动调用者 | 历史代码 | 旧测试或运行日志；不再进入当前测试集 | 仅保留于 Git 历史 | 删除 | V3 数据已经冻结，旧实现仍假设 NPZ 内嵌大数组 |
| `src/artifacts/readme.md` | 当前代码或数据契约说明 | 开发者与运维人员 | 活动文档 | 由 rg、审查与实现交叉核对 | 原位保留 | 修改 | 删除旧 NPZ、旧比例和旧入口陈述 |
| `src/datasets/README_STAGE1.md` | 当前代码或数据契约说明 | 开发者与运维人员 | 活动文档 | 由 rg、审查与实现交叉核对 | 原位保留 | 修改 | 删除旧 NPZ、旧比例和旧入口陈述 |
| `src/datasets/ops/stage1_box_pool.py` | 第二版 split 或 BOX pool 构建 | 旧数据准备命令 | 历史代码 | 被 V3 数据准备测试取代 | 仅保留于 Git 历史 | 删除 | 正式消费者只读取已验收的 V3 产物 |
| `src/datasets/ops/stage1_split.py` | 第二版 split 或 BOX pool 构建 | 旧数据准备命令 | 历史代码 | 被 V3 数据准备测试取代 | 仅保留于 Git 历史 | 删除 | 正式消费者只读取已验收的 V3 产物 |
| `src/datasets/readme.md` | 当前代码或数据契约说明 | 开发者与运维人员 | 活动文档 | 由 rg、审查与实现交叉核对 | 原位保留 | 修改 | 删除旧 NPZ、旧比例和旧入口陈述 |
| `src/datasets/stage1_collate.py` | Stage1 batch 拼装 | DataLoader | 正式代码 | tests/datasets/test_stage1_dataset.py | 原位保留 | 修改 | 拼接独立 atom_is_backbone |
| `src/datasets/stage1_dataset.py` | 统一训练与推理 80³ 物化 | src/train.py 与 src/inference | 正式代码 | tests/datasets/test_stage1_dataset.py | 原位保留 | 替换 | 使用 NPY mmap 并返回 49 维特征与主链标志 |
| `src/datasets/stage1_requests.py` | V3 BOX 请求解析与每周期选择 | Stage1Dataset 与推理清单脚本 | 正式代码 | tests/datasets/test_stage1_dataset.py | 原位保留 | 替换 | 按 PDB 分配 bias/context 请求，并稳定解析冻结验证索引 |
| `src/inference/README.md` | 当前代码或数据契约说明 | 开发者与运维人员 | 活动文档 | 由 rg、审查与实现交叉核对 | 原位保留 | 修改 | 删除旧 NPZ、旧比例和旧入口陈述 |
| `src/inference/assembly.py` | Stage1 推理数据装配 | src/inference/cli.py | 正式代码 | tests/inference/test_stage1_assembly_cli.py | 原位保留 | 修改 | 读取 union_mask.npy 并复用统一 Dataset |
| `src/inference/utils/receptor_strip.py` | 推理受体裁剪工具 | 推理装配 | 正式代码 | 推理测试 | 原位保留 | 修改 | 移除已删除历史目录说明 |
| `src/model/stage1_model.py` | Stage1 模型输入规范化 | 训练与推理 wrapper | 正式代码 | tests/model/test_stage1_model.py | 原位保留 | 修改 | 在模型边界按期望维数拼接主链标志 |
| `src/train.py` | 训练 Dataset 与 DataLoader 装配 | Hydra 训练入口 | 正式代码 | 全量 pytest 与配置测试 | 原位保留 | 修改 | 传入 prefetch_factor 并禁止常驻 worker |
| `src/wrappers/voxel_point_stage1_losses.py` | Stage1 结构辅助损失 | 训练 wrapper | 正式代码 | 模型与损失测试 | 原位保留 | 修改 | 使距离监督说明与有限裁块契约一致 |
| `tests/datasets/test_materialize_filtered_stage1_preparation.py` | 第二版 preparation 物化入口或测试 | 旧运维命令 | 历史代码 | 旧专用测试退出 | 仅保留于 Git 历史 | 删除 | 会重新发布旧路径与旧请求契约 |
| `tests/datasets/test_stage1_dataset.py` | 本轮正式行为回归测试 | pytest | 测试代码 | 由全量 pytest 自身执行 | 原位保留 | 修改 | 覆盖 V3 NPY、请求、模型边界、推理与配置 |
| `tests/datasets/test_stage1_split_pool.py` | 旧 split、pool 或配置断言 | pytest | 历史测试 | 由 V3 Dataset 与配置测试取代 | 仅保留于 Git 历史 | 删除 | 断言已经退出的入口或不存在配置 |
| `tests/inference/test_stage1_assembly_cli.py` | 本轮正式行为回归测试 | pytest | 测试代码 | 由全量 pytest 自身执行 | 原位保留 | 修改 | 覆盖 V3 NPY、请求、模型边界、推理与配置 |
| `tests/model/test_stage1_model.py` | 本轮正式行为回归测试 | pytest | 测试代码 | 由全量 pytest 自身执行 | 原位保留 | 修改 | 覆盖 V3 NPY、请求、模型边界、推理与配置 |
| `tests/test_adaligand_stage1_configs.py` | 本轮正式行为回归测试 | pytest | 测试代码 | 由全量 pytest 自身执行 | 原位保留 | 修改 | 覆盖 V3 NPY、请求、模型边界、推理与配置 |
| `tests/test_cpc_v3_configs.py` | 旧 split、pool 或配置断言 | pytest | 历史测试 | 由 V3 Dataset 与配置测试取代 | 仅保留于 Git 历史 | 删除 | 断言已经退出的入口或不存在配置 |
| `utils/validate_density_map_pairs.py` | 密度图配对验证工具 | 人工运维 | 可复用运维代码 | 全量 pytest 导入检查 | 原位保留 | 修改 | 改用仍保留的项目 mrc_tools |
| `训练与运行/README.md` | 当前代码或数据契约说明 | 开发者与运维人员 | 活动文档 | 由 rg、审查与实现交叉核对 | 原位保留 | 修改 | 删除旧 NPZ、旧比例和旧入口陈述 |
| `训练与运行/sh/Find_0.sh` | Stage1 一键训练入口 | submit_task.sh 与人工提交 | 正式运行脚本 | tests/test_adaligand_stage1_configs.py 与 bash -n | 原位保留 | 修改 | 统一 V3 数据、资源和模型变体参数 |
| `训练与运行/sh/Find_1.sh` | Stage1 一键训练入口 | submit_task.sh 与人工提交 | 正式运行脚本 | tests/test_adaligand_stage1_configs.py 与 bash -n | 原位保留 | 修改 | 统一 V3 数据、资源和模型变体参数 |
| `训练与运行/sh/infer/prepare_inference_pdb_lists.sh` | 推理 PDB 清单生成 | 人工 CPU 提交 | 正式运行脚本 | bash -n 与请求解析测试 | 原位保留 | 修改 | 读取 V3 split 并使用公开 PDB 清单解析器 |
| `训练与运行/sh/train_2/Find_0.sh` | 重复的旧训练入口 | 人工提交 | 历史脚本 | 由当前正式脚本配置测试取代 | 仅保留于 Git 历史 | 删除 | 避免多套训练入口产生参数漂移 |
| `训练与运行/sh/train_2/Find_1.sh` | 重复的旧训练入口 | 人工提交 | 历史脚本 | 由当前正式脚本配置测试取代 | 仅保留于 Git 历史 | 删除 | 避免多套训练入口产生参数漂移 |
| `训练与运行/sh/train_2/unet_c1.sh` | 重复的旧训练入口 | 人工提交 | 历史脚本 | 由当前正式脚本配置测试取代 | 仅保留于 Git 历史 | 删除 | 避免多套训练入口产生参数漂移 |
| `训练与运行/sh/train_3/Find_0.sh` | 重复的旧训练入口 | 人工提交 | 历史脚本 | 由当前正式脚本配置测试取代 | 仅保留于 Git 历史 | 删除 | 避免多套训练入口产生参数漂移 |
| `训练与运行/sh/train_3/Find_1.sh` | 重复的旧训练入口 | 人工提交 | 历史脚本 | 由当前正式脚本配置测试取代 | 仅保留于 Git 历史 | 删除 | 避免多套训练入口产生参数漂移 |
| `训练与运行/sh/train_3/unet_c1.sh` | 重复的旧训练入口 | 人工提交 | 历史脚本 | 由当前正式脚本配置测试取代 | 仅保留于 Git 历史 | 删除 | 避免多套训练入口产生参数漂移 |
| `训练与运行/sh/unet_c1.sh` | Stage1 一键训练入口 | submit_task.sh 与人工提交 | 正式运行脚本 | tests/test_adaligand_stage1_configs.py 与 bash -n | 原位保留 | 修改 | 统一 V3 数据、资源和模型变体参数 |
| `ops/stage1_data_preparation/utils/__init__.py` | V3 数据准备几何与冻结工具 | build_box_pool_3.py | 可复用运维代码 | ops/stage1_data_preparation/tests | 指定 utils 目录 | 新增 | 集中用户指定的 occurrence、seed、bias/context 与 validation 函数 |
| `ops/stage1_data_preparation/utils/box_pool.py` | V3 数据准备几何与冻结工具 | build_box_pool_3.py | 可复用运维代码 | ops/stage1_data_preparation/tests | 指定 utils 目录 | 新增 | 集中用户指定的 occurrence、seed、bias/context 与 validation 函数 |
| `talk/refactor/stage1_v3_training_io.md` | 本轮大规模整理的逐文件清单 | 实施者与两类独立审查 | 任务级规划记录 | 由 Git diff 与审查复核 | 原位保留 | 新增 | 满足大规模整理前后的可追溯要求 |
| `训练与运行/sh/unet_c1_no_mainchain.sh` | Stage1 一键训练入口 | submit_task.sh 与人工提交 | 正式运行脚本 | tests/test_adaligand_stage1_configs.py 与 bash -n | 原位保留 | 新增 | 统一 V3 数据、资源和模型变体参数 |

## 核验边界

整理前与整理后分别运行分层测试；最终以全量 pytest、四个 shell 的 `bash -n`、Hydra 配置断言、`git diff --check` 和两类独立审查为准。关键产物比较覆盖路径、字段、shape、dtype、XYZ/ZYX、Å 单位与缺失值语义。服务器正式训练不属于本轮未授权的核验步骤。
