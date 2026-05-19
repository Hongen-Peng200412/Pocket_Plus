# 分子对接任务入口

本目录是 Pocket Plus 下游分子对接工作的入口。它面向两类读者：

* 用户：快速知道当前已经做到了哪里、结果能说明什么、不能说明什么。
* AI agent：快速恢复上下文，继续生成输入、运行 Rosetta、审计输出、扩展样本与匹配算法。

## 当前目标

Pocket Plus 的神经网络会从 cryo-EM 密度图中预测 ligand 的大致空间位置。本目录记录的工作，是把这些预测位点转化为 Rosetta / EMERALD-ID 风格的 density-guided ligand docking 流程：

1. 读取服务器上的推理结果。
2. 找到每个样本的预测 ligand instance 和候选 mol2。
3. 为候选 ligand 生成 Rosetta params。
4. 在预测位点附近构造 receptor-ligand 复合物。
5. 使用真实 EMDB map 运行 `GALigandDock`。
6. 汇总 Rosetta 分数、网络形状分数，并用匈牙利匹配形成 site-ligand 分配。

本阶段的重点是流程跑通与可复用化，不是宣布最终 docking benchmark 成功率。

## 推荐阅读顺序

1. `Docking/分子对接记录/progress1_readable.md`
   作为面向用户的阶段报告，或者为 AI agent 提供的更详细信息。详细解释“成功”定义、7zdf、多样本结果、匈牙利匹配、矩形匈牙利匹配和当前局限。
2. `Docking/分子对接记录/progress1.md`
   面向 AI agent 的简明进度摘要。适合下一轮 agent 快速接手。
3. `Docking/分子对接记录/7zdf_emerald_id_exec_plan.md`
   7zdf 起步实验的执行记录与关键决策日志。
4. `Docking/分子对接工具.md`
   Rosetta、服务器路径、数据源、命令参数和注意事项。
5. `Docking/docking_pipeline/`
   当前流程的 Python 化骨架。它不是最终生产流水线，但已经按可扩展模块拆分：数据读取、Rosetta 输入生成、shape score、匈牙利匹配和运行编排。
6. `Ligand/readme.md` 与 `Ligand/notes_of_dataset.md`
   ligand enrichment 数据集说明，尤其是 `ligand_mapping.csv`、`native_mol2_path`、`status`、`emerald_id_input_status` 等字段。

## 本地与服务器边界

本地可编辑范围：

```text
C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking
```

服务器运行产物目录：

```text
/home/penghongen/分子对接尝试
```

服务器上可以读取项目、Rosetta、推理结果和 ligand 数据，但不要写入 `/home/penghongen/分子对接尝试` 以外的位置，除非用户明确授权；另外，可以编辑本地的 C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking 并使用固定的同步脚本 "C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat" 以此在服务器上使用新代码。你也可以参考 C:\Users\15919\OneDrive\My_Project\Pocket_Plus\sbatch 提交任务（并维持使用 try_lock 等功能），但是在一轮任务里最多只能提交2张A100,48核cpu（提示：cpu 通常不用排队）。

除此以外的任何行为都需要得到用户的明确授权！

## 当前服务器产物

已完成的主要产物位于：

```text
/home/penghongen/分子对接尝试/7zdf_round1_site001
/home/penghongen/分子对接尝试/7zdf_round2_all_sites
/home/penghongen/分子对接尝试/multi_sample_smoke_8dd7_8x9s
/home/penghongen/分子对接尝试/multi_sample_8pmd_rectangular
```

这些目录中保留了 `audit/`、`inputs/`、`params/`、`xml/`、`logs/`、`outputs/` 等子目录。继续工作时优先读 `audit/*.md` 和 `audit/*.json`。

## 下一步建议

1. 把 `Docking/docking_pipeline` 中的骨架脚本同步到服务器允许目录，先复现实验结果（这是因为目前的结果大多是以命令的方式得到的，但是想要发论文就要尽量用落盘化的脚本）。
2. 将 `nstruct=1` 的低成本 smoke test 扩展为可配置的批处理。
3. 改进 shape score本身：从径向分布 proxy 升级到 docking 后 ligand 空间形状与网络 instance voxel 的直接比较。
4. 给匈牙利匹配加入虚拟实例，显式惩罚“忽略预测 instance”和“漏掉真实 ligand”。
5. 在目前的所有已有样本上（未来可以再获得约 300 个验证样本）上用传统机器学习或网格搜索学习打分权重。