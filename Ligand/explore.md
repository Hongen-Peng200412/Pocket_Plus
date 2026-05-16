# RCSB 单源获取 Make_Data 配体 SMILES 与原生 mol2 的可行性探索

## 背景

当前 `Make_Data` 主用数据集为 `/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/`。每个 PDB 子目录中已有：

```text
candidates.npz
labels.npz
atoms.npz
residues.npz
graph.npz
```

其中 `labels.npz` 记录筛选后 ligand instance，`candidates.npz` 记录过滤前候选配体的残基名、链 ID、残基编号、插入码、重原子数和坐标。根据 `Make_Data/notes_of_dataset.md`，小分子配体类别为：

```text
ligand_class_ids == 4  # small_molecule
```

本探索目标是确认：**是否可以只依赖 RCSB PDB，给当前 Make_Data 识别出的 class-4 ligand instance 高置信补充 SMILES 与 RCSB 原生 mol2 文件**。

## 核心结论

**RCSB 单源路线可行。**

对本地下载的 Make_Data 样例目录：

```text
C:\Users\15919\Desktop\服务器上的部分数据\home-penghongen-My_Project-Data-DATA_v2_raw4-parsed_pdb
```

进行了端到端实验。结果显示：

```text
class-4 ligand instances: 58
PASS_HIGH: 58 / 58
失败: 0 / 58
```

这说明在已测试样例中，RCSB PDB 可以同时提供：

```text
1. full structure CIF
2. RCSB chemical component SMILES / SMILES_stereo / InChI / InChIKey
3. ligand instance native mol2
4. 足够精确的 ligand instance 坐标，用于和 Make_Data candidate_coords 对齐
```

## 实验样本

优先完整测试了用户指定的 4 个样本：

| PDB ID | class-4 ligand 数 | 结果 |
|---|---:|---|
| `3j7a` | 1 | 1/1 PASS_HIGH |
| `5bki` | 4 | 4/4 PASS_HIGH |
| `5gmk` | 1 | 1/1 PASS_HIGH |
| `5i68` | 1 | 1/1 PASS_HIGH |

随后扩展到本地目录中的全部样例：

| PDB ID | class-4 ligand 数 | 结果 |
|---|---:|---|
| `3j79` | 0 | 无 class-4 ligand |
| `3j7a` | 1 | 1/1 PASS_HIGH |
| `5bki` | 4 | 4/4 PASS_HIGH |
| `5gae` | 0 | 无 class-4 ligand |
| `5gmk` | 1 | 1/1 PASS_HIGH |
| `5i68` | 1 | 1/1 PASS_HIGH |
| `5iqr` | 1 | 1/1 PASS_HIGH |
| `5irx` | 20 | 20/20 PASS_HIGH |
| `5irz` | 24 | 24/24 PASS_HIGH |
| `5is0` | 4 | 4/4 PASS_HIGH |
| `5lzd` | 2 | 2/2 PASS_HIGH |

通过的 CCD ID 包括：

```text
34G, 6ES, 6ET, 6EU, 6O8, 6O9, 6OE, CL, FAD, GNP, GTP, PAR, PGW
```

## 实验流程

### 1. 从 Make_Data 读取 ligand instance

对每个 `{pdb_id}` 子目录读取：

```text
labels.npz
candidates.npz
```

只处理：

```text
ligand_class_ids == 4
```

每个 ligand instance 使用以下 Make_Data 字段作为本地主键和对齐依据：

```text
pdb_id
candidate_id
resname / CCD ID
chain_id
res_id
insertion_code
n_heavy_atoms
candidate_coords_{candidate_id}
```

### 2. 下载 RCSB full CIF

使用：

```text
https://files.rcsb.org/download/{PDB_ID}.cif
```

下载 PDB 对应的 full structure CIF。

### 3. 从 RCSB CIF 中建立 ligand instance 对应关系

关键表为：

```text
_pdbx_nonpoly_scheme
_pdbx_branch_scheme
```

实验中发现，Make_Data 的 `chain_id/res_id` 在测试样例中对应：

```text
Make_Data chain_id -> _pdbx_nonpoly_scheme.pdb_strand_id
Make_Data res_id   -> _pdbx_nonpoly_scheme.pdb_seq_num
Make_Data resname  -> _pdbx_nonpoly_scheme.mon_id / pdb_mon_id / auth_mon_id
```

匹配后得到 RCSB ModelServer 下载 ligand mol2 所需的：

```text
label_asym_id = _pdbx_nonpoly_scheme.asym_id
residue number = _pdbx_nonpoly_scheme.pdb_seq_num
```

> [!IMPORTANT]
> RCSB ModelServer ligand endpoint 的参数名是 `auth_seq_id`，但在这些样例中实际需要传入 RCSB 页面可见的 PDB residue number，即 `_pdbx_nonpoly_scheme.pdb_seq_num`。如果误用 `_pdbx_nonpoly_scheme.auth_seq_num`，会出现 `element_count: 0`，导致误判为 mol2 缺失。

大规模运行后发现，`RCSB_INSTANCE_MATCH_FAILED` 主要集中在 `NAG/MAN/BMA/FUC/GAL` 等糖类或支链糖残基。原因是这些残基通常不在 `_pdbx_nonpoly_scheme`，而在 `_pdbx_branch_scheme`。正式程序已补充 branch scheme 匹配：

```text
Make_Data chain_id -> _pdbx_branch_scheme.pdb_asym_id
Make_Data res_id   -> _pdbx_branch_scheme.pdb_seq_num
Make_Data resname  -> _pdbx_branch_scheme.mon_id / pdb_mon_id / auth_mon_id
```

继续诊断剩余 1.4% 非糖类失败后发现，许多特殊 HETATM/修饰残基不在 `_pdbx_nonpoly_scheme` 或 `_pdbx_branch_scheme` 中，但在 `_atom_site` 中可以通过 `auth_asym_id/auth_seq_id` 或 `label_asym_id/label_seq_id` 唯一定位。例如：

```text
6AP1 ACE chain G res 0
6VMI Y5P chain A5 res 1
6VMI P5P chain A5 res 2
7FGI GTA chain M res 1003
8CEP KBE/DPP/UAL/MYN chain V res 1-4
9IF4 S0R chain Y res 1
```

正式程序已补充 `_atom_site` 兜底匹配。若标识仍无法唯一化，则在正式全量运行中继续使用 Make_Data `candidate_coords_{id}` 做坐标兜底匹配。

### 4. 下载 RCSB native ligand mol2

使用 RCSB ModelServer ligand endpoint 下载：

```text
https://models.rcsb.org/v1/{pdb_id}/ligand?auth_seq_id={pdb_seq_num}&label_asym_id={label_asym_id}&encoding=mol2&filename={filename}
```

本探索只接受 RCSB 原生 mol2：

```text
不允许 SDF -> mol2 转换
不允许 mmCIF -> mol2 转换
不允许 PubChem / Q-BioLiP / ChEBI fallback
```

### 5. 获取 RCSB chemical component descriptors

使用：

```text
https://data.rcsb.org/rest/v1/core/chemcomp/{CCD_ID}
```

读取：

```text
rcsb_chem_comp_descriptor.SMILES
rcsb_chem_comp_descriptor.SMILES_stereo
rcsb_chem_comp_descriptor.InChI
rcsb_chem_comp_descriptor.InChIKey
```

### 6. 多角度校验

每个 class-4 ligand instance 执行以下校验：

| 校验 | 含义 | 通过标准 |
|---|---|---|
| Make_Data vs RCSB CIF 重原子数 | 本地 candidate 是否和 RCSB full CIF 中同一 ligand instance 数量一致 | 必须一致 |
| RCSB CIF vs native mol2 重原子数 | RCSB full CIF 与 RCSB native mol2 是否同一 ligand instance | 必须一致 |
| RCSB CIF vs native mol2 元素组成 | mol2 是否对应同一化学实例 | 必须一致 |
| Make_Data vs RCSB CIF 坐标 | 本地解析坐标是否对应 RCSB CIF | 同元素最近邻 median <= 0.05 Å 且 max <= 0.20 Å |
| RCSB CIF vs native mol2 坐标 | RCSB native mol2 是否为 instance coordinates | 同元素最近邻 median <= 0.05 Å 且 max <= 0.20 Å |

实际全样例统计：

```text
RCSB CIF vs mol2 median distance max: 0.0 Å
RCSB CIF vs mol2 max distance max:    0.0 Å
Make_Data vs RCSB CIF median max:     1.27e-05 Å
Make_Data vs RCSB CIF max max:        1.87e-05 Å
```

## 工具输入合规性观察

本探索不运行 docking，只判断下载到的 SMILES/mol2 是否满足或接近满足工具输入要求。

### DockEM

DockEM 需要：

```text
protein.mol2
ligand.mol2
bindingsite_pre.dat
map.mrc
resolution
sampling 参数
```

本探索得到：

```text
48 records: FORMAT_OK
10 records: FORMAT_OK_WITH_WARNING
```

warning 主要是：

```text
MOL2_HAS_NO_HYDROGEN
MOL2_HAS_NO_BONDS  # 例如单原子 chloride
```

这不影响 `(SMILES, native mol2)` pair 获取成功，但 DockEM 前可能仍需按工具要求加氢或做特定参数准备。

### EMERALD-ID

本探索将 EMERALD-ID 视为 ligand library docking / identification 流程，只检查：

```text
ligand ID
native mol2
SMILES / InChIKey
```

结果：

```text
58 records: FORMAT_OK_FOR_LIBRARY_ENTRY
```

### PocketXMol

PocketXMol 可使用：

```text
protein structure
pocket center
SMILES 或 ligand structure
```

结果：

```text
58 records: FORMAT_OK_SMILES_AND_STRUCTURE
```

## 当前限制

1. 本地 `Pocket_Plus_windows` 环境没有 RDKit/OpenBabel，因此本探索没有执行 `mol2 -> InChIKey` 的化学重构交叉验证。
2. 当前强结论只覆盖 `ligand_class_ids == 4` 的 small molecule，不覆盖 peptide/nucleic ligand。
3. 当前结论基于本地下载的部分 Make_Data 样例。正式程序需要在服务器完整 raw4 数据集上批量运行并输出覆盖率统计。

## 对正式管线的建议

短期正式管线可以采用：

```text
Make_Data ligand instance
-> RCSB full CIF
-> _pdbx_nonpoly_scheme / _pdbx_branch_scheme 对齐
-> RCSB native ligand mol2
-> RCSB chemcomp SMILES/InChI/InChIKey
-> 坐标/重原子/元素组成校验
-> 保存 mapping.csv + mapping.jsonl
```

Q-BioLiP 暂时不作为 `(SMILES, mol2)` 获取的必要来源，后续更适合作为：

```text
biological relevance
binding residues
interaction/site annotation
```

的补充层。

## 正式程序说明

上述路线已经落成自动化程序：

```text
Ligand/rcsb_enrichment/run_enrichment.py
```

服务器推荐通过：

```text
Ligand/sbatch/rcsb_ligand_enrichment_raw4.sbatch
```

提交。该 sbatch 使用 5 个 array task，每个 task 8 个 CPU，并在每个 task 内用 `joblib` 按 PDB 并行。

正式输出根目录：

```text
/storage/penghongen/CIF_Ligand
```

主要产物：

```text
rcsb_full_cif/                         # RCSB full CIF 缓存
rcsb_chemcomp_cache/                   # RCSB chemcomp JSON 缓存
rcsb_ligand_mol2/                      # RCSB native ligand mol2
mapping/parts/                         # array task 产出的分片 mapping
mapping/ligand_mapping.csv             # merge 后或非 array 模式总表
mapping/ligand_mapping.jsonl           # merge 后或非 array 模式 JSONL 总表
mapping/failed_cases.csv               # 失败样本表
reports/validation_summary.json        # 总统计摘要
```

array 任务完成后，使用：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
python -m Ligand.rcsb_enrichment.merge_outputs \
  --output-root /storage/penghongen/CIF_Ligand \
  --array-count 5
```

合并分片结果。完整文件级和字段级说明见：

```text
Ligand/notes_of_dataset.md
```
