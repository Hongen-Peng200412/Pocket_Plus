# 7zdf EMERALD-ID / Rosetta 对接执行记录

本文档记录从 `7zdf` 开始，把 Pocket Plus 推理位点接到 Rosetta density-guided ligand docking 的过程。它不是最终论文式报告，而是可恢复、可审计、可继续扩展的执行记录。

## 写入边界

本地当前可编辑范围：

```text
C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking
```

服务器本任务产物目录：

```text
/home/penghongen/分子对接尝试
```

以及，只能使用这个特定的同步脚本 "C:\Users\15919\OneDrive\My_Project\scrips++\run_sync.bat"。服务器上不要写入该目录以外的位置。可以只读访问推理结果、Rosetta 安装、ligand 数据和结构数据。

## 起始目标

第一阶段的目标是验证以下闭环能否跑通：

```text
Pocket Plus 推理 instance
-> 预测 docking center
-> 样本候选 mol2
-> Rosetta params
-> receptor-ligand 复合物
-> 真实 EMDB map
-> GALigandDock
-> scorefile / PDB / 日志
-> site-ligand 评分与匹配
```

当前阶段不宣称 docking 精度已经达标。这里的“成功”仅指流程成功：输入可生成、Rosetta 可读入、作业正常结束、scorefile 和输出结构存在，并能被脚本解析。

## 已完成进度

* 选择 `ligand_base2_new/stage2_threshold_component_policy` 作为第一阶段推理结果来源。
* 确认 `ligand_base1`、`ligand_base2`、`ligand_base2_new` 是三个不同模型输出，不混用。
* 确认 docking density 使用 infer cache 中 `meta_json["map_path"]` 指向的真实 EMDB map。
* 明确不使用 `sim_map_path` 作为 docking density；它是 cryoatom receptor 的模拟 map，与本轮 Rosetta density-guided docking 目标不同。
* 候选 ligand 来自 `/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv`。
* 独立金属离子不作为 ligand docking 候选。
* 对每个样本使用两套 receptor：`true_receptor` 和 `cryoatom_receptor`。
* 完成 `7zdf` 单位点、全位点、多样本 smoke、`8pmd` 矩形匹配测试。

## 修订说明：2026-05-19

本节替代早期英文笔记中类似 `Revision note 2026-05-19: Completed read-only audit...` 的简写说法。此前已经对 `7zdf` 完成服务器只读审计，确认推理输出、候选 ligand、真实受体、cryoatom 预测受体、真实 EMDB map、分辨率表和 Rosetta 工具链均可定位。审计结果随后被写入服务器允许目录，但第一次从 Windows 侧向 SSH 命令直接传入中文路径时，`/home/penghongen/分子对接尝试` 被错误转写成 `/home/penghongen/??????`。用户已经手动删除该错误目录。

从这次修订开始，本地可编辑文档集中迁移到：

```text
C:\Users\15919\OneDrive\My_Project\Pocket_Plus\Docking
```

服务器侧仍只允许在任务产物目录内写入：

```text
/home/penghongen/分子对接尝试
```

后续如果需要从 Windows 发起服务器命令并涉及中文路径，建议在服务器端用 Python 的 Unicode 转义字符串构造路径，或者先 `cd /home/penghongen` 后由服务器 shell 自己解析中文目录名，避免再次出现路径乱码。

## 样本选择：7zdf

`7zdf` 适合作为首个样本，因为：

```text
voxel_f1 = 0.8097799511002445
instance_f1 = 1.0
instance_recall = 1.0
num_pred_instances = 2
num_gt_instances = 3
非独立金属候选 ligand = ISW, ANP
独立金属条目 = MG，已排除
```

`ISW` 内部含 `FE`，不是独立金属离子，但结果解释必须标记为“含金属 ligand”风险。

## 7zdf 只读审计

审计读取了：

```text
/home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/stage2_threshold_component_policy/7zdf/summary.json
/home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/stage2_threshold_component_policy/7zdf/metrics.json
/home/penghongen/My_Project/feedback_plus/infer_out/ligand_base2_new/stage2_threshold_component_policy/7zdf/voxel_candidates.json
/home/penghongen/My_Project/feedback_plus/infer_cache/ligand_base2_new/7zdf.npz
/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv
```

真实 map：

```text
/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc/emd_14644.map
```

分辨率：

```text
2.94 Å
```

true receptor：

```text
/storage/chenzhaoyang/cryo_em/CIF_3.5_atom/7ZDF.cif
```

cryoatom receptor：

```text
/storage/chenzhaoyang/cryo_em/result_split/7zdf/7zdf.cif
```

## 中文路径编码事故与处理

第一次在 Windows 侧向 SSH 传递中文路径时，`/home/penghongen/分子对接尝试` 被错误写成了字面量：

```text
/home/penghongen/??????
```

用户已手动删除该错误目录。后续脚本改为在服务器端用 Python Unicode 转义构造路径：

```python
allowed = Path("/home/penghongen") / "\u5206\u5b50\u5bf9\u63a5\u5c1d\u8bd5"
```

这会得到正确路径：

```text
/home/penghongen/分子对接尝试
```

后续不要直接把中文路径嵌入 Windows shell 命令字符串。

## 位点选择

`7zdf` 有两个预测 instance：

```text
site001:
  instance_id = 1
  voxel_count = 514
  score_mean = 0.9883579663961314
  score_max = 0.9999430775642395
  center_world_xyz = [143.2909979129116, 129.1227468209276, 103.72027994721316]

site002:
  instance_id = 2
  voxel_count = 649
  score_mean = 0.9883437724253062
  score_max = 0.9997135400772095
  center_world_xyz = [112.86540356012082, 123.7779846499532, 135.61299348029038]
```

单位点第一轮选择 `site001`，规则是：

1. 优先 `score_max` 高。
2. 若接近，则看 `score_mean` 和 `voxel_count`。
3. 不使用真实 ligand 距离选择位点，避免用 GT 泄漏来挑点。

## Rosetta 输入生成

候选 mol2：

```text
ISW: /storage/penghongen/CIF_Ligand/rcsb_ligand_mol2/7zdf/0_ISW_C_601.mol2
ANP: /storage/penghongen/CIF_Ligand/rcsb_ligand_mol2/7zdf/2_ANP_E_603.mol2
```

使用：

```text
/home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main/source/scripts/python/public/molfile_to_params.py
```

生成：

```text
ANP.params
ANP_0001.pdb
ISW.params
ISW_0001.pdb
```

true receptor CIF 可被 Rosetta 直接读取，但流程中也派生了 PDB，便于统一拼接复合物。cryoatom CIF 因两字符 author chain ID 导致 Rosetta mmCIF 解析失败，因此转换为单字符 chain PDB 后使用。

## 可用 GALigandDock 协议

已验证有效的关键设置：

```text
scorefxn = beta_genpot
命令行必须包含 -corrections::gen_potential true
runmode = VSX
reference_pool = map
use_pharmacophore = false
local_resolution = 样本分辨率
nstruct = 1
```

`nstruct=1` 是低成本流程验证设置，不是最终实验设置。

## 7zdf round1：单位点

服务器目录：

```text
/home/penghongen/分子对接尝试/7zdf_round1_site001
```

矩阵：

```text
2 套 receptor x 2 个 ligand = 4 个 docking
```

全部成功。主要结果：

```text
true_receptor + ANP: dG = 1.59653
true_receptor + ISW: dG = -30.2953
cryoatom_receptor + ANP: dG = 1.20128
cryoatom_receptor + ISW: dG = -30.7456
```

注意：round1 是低成本 smoke test，只能说明流程和相对分数可解析，不能说明 RMSD 成功率。

## 7zdf round2：所有预测位点

服务器目录：

```text
/home/penghongen/分子对接尝试/7zdf_round2_all_sites
```

矩阵：

```text
2 个 site x 2 个 ligand x 2 套 receptor = 8 个 docking
```

全部成功。

额外计算了初版网络形状分数：比较预测 instance 体素点的径向分布，与 ligand mol2 原子云的径向分布。它是方向不变的粗略 proxy，不是最终 shape matcher。

匈牙利匹配采用：

```text
combined_cost = 0.7 * minmax(dG) + 0.3 * minmax(network_shape_score)
```

其中成本越低越好。

结果：

```text
true_receptor: site001 -> ANP, site002 -> ISW
mean_receptor: site001 -> ANP, site002 -> ISW
cryoatom_receptor: site001 -> ISW, site002 -> ANP
```

这说明 receptor 来源会影响打分，是后续需要系统评估的风险。

## 多样本 smoke

服务器目录：

```text
/home/penghongen/分子对接尝试/multi_sample_smoke_8dd7_8x9s
```

样本：

```text
8dd7: FAD, 1 个预测位点
8x9s: DHT, 1 个预测位点
```

每个样本运行：

```text
1 个 site x 1 个 ligand x 2 套 receptor = 2 个 docking
```

两个样本共 4 个 docking，全部成功。

## 8pmd：矩形匹配测试

服务器目录：

```text
/home/penghongen/分子对接尝试/multi_sample_8pmd_rectangular
```

`8pmd` 有：

```text
3 个预测 site
2 个 ATP mol2 条目
2 套 receptor
```

运行矩阵：

```text
3 x 2 x 2 = 12 个 docking
```

全部成功。

因为两个 ligand 都是 ATP，为避免 Rosetta 三字符 residue name 冲突，流程分配了：

```text
ATP_01 -> L01
ATP_02 -> L02
```

文档和输出中保留 `ATP_01`、`ATP_02` 作为候选 mol2 标签。

矩形匈牙利匹配含义：

* 预测 site 数量为 3。
* ligand mol2 数量为 2。
* 只需要把 2 个 ligand 分配给 2 个 site。
* 允许剩余 1 个 site unmatched。

当前实现用穷举组合等价实现矩形匈牙利匹配：从 3 个 site 中选 2 个，再对 2 个 ligand 做一一匹配，选总成本最低的方案。

## 当前局限

1. 所有 docking 都是 `nstruct=1`，只用于流程验证。
2. 当前“成功率”不是 RMSD 成功率，而是流程作业成功率。
3. shape score 是径向分布 proxy，可能还需要考虑真实三维形状重叠。
4. 尚未加入虚拟 instance。
5. 尚未用验证集学习权重。
6. 含金属 ligand 需要单独标记和解释。

## 下一步

1. 把 `Docking/docking_pipeline` 脚本同步到服务器允许目录并复现已有结果。
2. 将 shape score 改为比较 docking 后 ligand 空间形状与预测 instance voxel。
3. 在匈牙利匹配中加入虚拟实例。
4. 扩展到更多样本，并保留 `true_receptor` 与 `cryoatom_receptor` 的分歧审计。
5. 设计验证集和传统机器学习调参接口。