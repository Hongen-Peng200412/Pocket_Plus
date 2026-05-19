# progress1：AI agent 快速进度摘要

本文档给后续 AI agent 快速恢复上下文。面向用户的详细版见 `progress1_readable.md`。

## 当前结论

已跑通从 Pocket Plus 推理 instance 到 Rosetta `GALigandDock` 的低成本闭环：

```text
推理输出 -> 预测 site -> 候选 mol2 -> params -> receptor-ligand complex
-> 真实 EMDB map -> GALigandDock -> scorefile -> shape score -> 匹配
```

这里的“成功”是流程成功，即 workflow success，不是 RMSD 成功。定义为：

```text
Rosetta 进程 returncode == 0
输出 PDB 存在
scorefile 存在
scorefile 或日志中可解析 dG 等字段
```

## 已完成服务器产物

```text
/home/penghongen/分子对接尝试/7zdf_round1_site001
/home/penghongen/分子对接尝试/7zdf_round2_all_sites
/home/penghongen/分子对接尝试/multi_sample_smoke_8dd7_8x9s
/home/penghongen/分子对接尝试/multi_sample_8pmd_rectangular
```

每个目录优先读：

```text
audit/*.md
audit/*.json
logs/*.stdout.log
outputs/**/**/*_score.sc
```

## 已验证样本

```text
7zdf round1: 2 receptors x 2 ligands x 1 site = 4/4 成功
7zdf round2: 2 receptors x 2 ligands x 2 sites = 8/8 成功
8dd7: 2 receptors x 1 ligand x 1 site = 2/2 成功
8x9s: 2 receptors x 1 ligand x 1 site = 2/2 成功
8pmd: 2 receptors x 2 ligand mol2 x 3 sites = 12/12 成功
```

总计当前 smoke / first-pass 矩阵：

```text
28/28 Rosetta 作业流程成功
```

这不是科学意义上的 docking accuracy。

## 核心数据源

推理输出：

```text
/home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/stage2_threshold_component_policy
```

ligand mapping：

```text
/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv
```

分辨率：

```text
/storage/penghongen/EMDB_PDB_resolution_3.5.csv
```

true receptor：

```text
/storage/chenzhaoyang/cryo_em/CIF_3.5_atom/{PDB_ID}.cif
```

cryoatom receptor：

```text
/storage/chenzhaoyang/cryo_em/result_split/{pdb_id_lower}/{pdb_id_lower}.cif
```

真实 map 来自 infer cache 的 `meta_json["map_path"]`。不要把 `sim_map_path` 用作 docking density。

## 关键 Rosetta 设置

Rosetta：

```text
/home/penghongen/software/rosetta/bin/rosetta_scripts_430
```

必须加：

```text
-corrections::gen_potential true
```

XML 中：

```text
scorefxn = beta_genpot
runmode = VSX
reference_pool = map
use_pharmacophore = false
nstruct = 1
```

## 匹配实现现状

当前 matching cost：

```text
combined_cost = 0.7 * minmax(dG) + 0.3 * minmax(network_shape_score)
```

`network_shape_score` 当前是预测 instance voxel 与 ligand mol2 原子云的径向分布差异，越低越好。

7zdf 方阵匹配：

```text
2 sites x 2 ligands -> 普通一一匹配
```

8pmd 矩形匹配：

```text
3 sites x 2 ligand mol2 -> 选择 2 个 site 分配给 2 个 ligand，剩下 1 个 site unmatched
```

当前还没有加入虚拟实例。下一版应把虚拟预测 instance 和虚拟 ligand 都放入二部图，使“忽略一个预测 instance”和“漏掉一个真实 ligand”都成为显式、有参数的损失项。

## 当前代码骨架

新增代码位于：

```text
Docking/docking_pipeline/
```

模块职责：

```text
config.py: 路径、Rosetta 参数、候选 ligand 过滤配置
records.py: dataclass 记录结构
io_utils.py: JSON/CSV/NPZ、CIF->PDB、mol2 读取
rosetta.py: params、complex、XML、命令构造
shape_scoring.py: 径向 shape score 与后续 shape scorer 接口
matching.py: 方阵、矩形、虚拟实例匹配
runner.py: 样本级编排骨架
```

## 下一步优先级

1. 在服务器允许目录复现 `Docking/docking_pipeline`。
2. 进一步改进，比如：在 shape score 的基础上额外考虑 docking 后 ligand 空间形状 vs 网络 instance voxel；给匹配设计虚拟实例成本函数等等。
3. 扩展 300 个验证样本，保存特征表。
4. 用网格搜索或随机森林学习 cost 权重。