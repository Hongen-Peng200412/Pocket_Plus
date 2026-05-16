# CIF_Ligand 数据集说明

若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；本文只是下游分子对接数据链路中的子说明。

本文是 `/storage/penghongen/CIF_Ligand` 的最终数据说明与字段索引，面向后续 AI agent、训练/推理脚本、质检脚本和 docking 输入准备脚本使用。

本文用于定位产物和理解字段，但不替代实际 enrichment 代码、RCSB 返回结果、真实 JSON/CSV/MOL2 文件和工具输入校验。若字段说明与实现或产物冲突，以实现和真实产物为准。

旧版探索式说明已归档到：

```text
Ligand/doc/archive/notes_of_dataset_legacy_2026-05-16.md
```

更完整的背景、探索过程和执行过程见：

```text
Ligand/doc/explore.md        # 为什么选择 RCSB 单源路线
Ligand/doc/implement_plan.md # 正式实现计划
Ligand/doc/exec_plan.md      # 实际执行、调试和验收记录
```

## 1. 目标与结论

本数据集为 Make_Data 中 `ligand_class_ids == 4` 的 small molecule ligand instance 补充 RCSB 单源信息：

```text
SMILES / stereo SMILES
InChI / InChIKey
RCSB native mol2
Make_Data candidate_id <-> RCSB ligand instance 映射关系
多角度校验结果
DockEM / EMERALD-ID / PocketXMol 输入格式初筛状态
```

最终服务器验收结果：

```text
class4_candidate_count = 190256
PASS_HIGH = 190093
RCSB_NATIVE_MOL2_MISSING = 156
VALIDATION_FAILED = 7
RCSB_INSTANCE_MATCH_FAILED = 0
```

成功率：

```text
全量 class-4 口径:
190093 / 190256 = 99.914%

排除 RCSB ModelServer native mol2 外部缺失口径:
190093 / (190256 - 156) = 99.996%
```

结论：RCSB 单源 enrichment 路线已达到短期方案验收标准。`RCSB_NATIVE_MOL2_MISSING` 是 RCSB ModelServer 外部源缺失；最终 7 条 `VALIDATION_FAILED` 是保守阈值下保留的 hard cases。

## 2. 服务器路径

正式产物统一写入：

```text
/storage/penghongen/CIF_Ligand/
```

| 路径 | 内容 | 生成入口 |
|---|---|---|
| `/storage/penghongen/CIF_Ligand/rcsb_full_cif/` | RCSB full structure CIF，每个 PDB 一个 `{PDB}.cif`。 | `Ligand/rcsb_enrichment/rcsb_client.py::download_full_cif()` |
| `/storage/penghongen/CIF_Ligand/rcsb_chemcomp_cache/` | RCSB chemical component JSON，每个 CCD 一个 `{CCD}.json`。 | `Ligand/rcsb_enrichment/rcsb_client.py::download_chemcomp()` |
| `/storage/penghongen/CIF_Ligand/rcsb_ligand_mol2/{pdb_id}/` | RCSB native ligand mol2，每个 Make_Data class-4 candidate 一个 mol2。 | `Ligand/rcsb_enrichment/rcsb_client.py::download_native_mol2()` |
| `/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv` | merge 后的总映射表；一行对应一个 Make_Data ligand candidate。 | `Ligand/rcsb_enrichment/io_utils.py::write_mapping_outputs()` |
| `/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.jsonl` | 与 CSV 同语义的 JSONL 总表；嵌套字段保留为 JSON 对象。 | `Ligand/rcsb_enrichment/io_utils.py::write_mapping_outputs()` |
| `/storage/penghongen/CIF_Ligand/mapping/failed_cases.csv` | 总表中非 `PASS_HIGH` 且非 `SKIPPED_NON_SMALL_MOLECULE` 的记录。 | `Ligand/rcsb_enrichment/io_utils.py::write_mapping_outputs()` |
| `/storage/penghongen/CIF_Ligand/mapping/parts/` | sbatch array 模式下每个 array task 的 part CSV/JSONL。 | `Ligand/rcsb_enrichment/run_enrichment.py` |
| `/storage/penghongen/CIF_Ligand/reports/validation_summary.json` | merge 后的总统计摘要。 | `Ligand/rcsb_enrichment/io_utils.py::build_summary()` |
| `/storage/penghongen/CIF_Ligand/reports/parts/` | sbatch array 模式下每个 array task 的 part summary。 | `Ligand/rcsb_enrichment/run_enrichment.py` |

## 3. 输入来源

### 3.1 raw.json 外层过滤

默认读取：

```text
/home/penghongen/My_Project/Data/raw.json
```

格式：

```text
list[dict[str, str]]
```

单个元素示例：

```json
{"emd_63092": "9LHB"}
```

处理规则：

```text
1. PDB ID 统一大写后，与 parsed_pdb 子目录匹配。
2. raw.json 缺省或 CLI 传入空字符串时，不做外层过滤，扫描 parsed_pdb 全部子目录。
3. raw.json 文件存在但内容为 [] 时，表示显式空过滤集，处理 0 个样本。
```

### 3.2 Make_Data parsed_pdb

默认读取：

```text
/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/
```

当前程序使用：

```text
labels.npz      # ligand_candidate_ids, ligand_class_ids, ligand_resnames
candidates.npz  # resnames, chain_ids, res_ids, insertion_codes, n_heavy_atoms, candidate_coords_{id}
```

只下载和校验：

```text
ligand_class_ids == 4
```

其它 ligand 只写入 `SKIPPED_NON_SMALL_MOLECULE` 统计，不下载 RCSB ligand 文件。

## 4. RCSB 对齐逻辑

对每个 Make_Data class-4 candidate，程序按以下顺序建立 RCSB ligand instance 映射。

### 4.1 nonpoly scheme

使用 `_pdbx_nonpoly_scheme`：

```text
Make_Data resname        -> mon_id / pdb_mon_id / auth_mon_id
Make_Data chain_id       -> pdb_strand_id
Make_Data res_id         -> pdb_seq_num，必要时尝试 auth_seq_num
Make_Data insertion_code -> pdb_ins_code
```

成功时 `match_method` 为：

```text
PDB_STRAND_PDB_SEQ
PDB_STRAND_AUTH_SEQ
```

### 4.2 branch scheme

糖链/支链配体常在 `_pdbx_branch_scheme`，而不是 `_pdbx_nonpoly_scheme`。

使用规则：

```text
Make_Data resname  -> mon_id / pdb_mon_id / auth_mon_id
Make_Data chain_id -> pdb_asym_id
Make_Data res_id   -> pdb_seq_num，必要时尝试 auth_seq_num
```

成功时 `match_method` 为：

```text
BRANCH_PDB_ASYM_PDB_SEQ
BRANCH_PDB_ASYM_AUTH_SEQ
```

重要约定：

```text
RCSB ModelServer ligand endpoint 参数名是 auth_seq_id，
但这里应传入 RCSB 页面可见 residue number，即 pdb_seq_num。
```

### 4.3 atom_site fallback

某些特殊 HETATM 或修饰残基不在 nonpoly/branch scheme 中，但 `_atom_site` 中有完整坐标和标识。

fallback 匹配策略：

```text
1. 优先用 Make_Data candidate_coords_{id} 在 _atom_site 中唯一化匹配。
2. 若坐标没有唯一命中，再按 auth_asym_id/auth_seq_id、auth_asym_id/label_seq_id、label_asym_id/label_seq_id、label_asym_id/auth_seq_id 精确匹配。
3. altLoc 选择优先保留空 alt + A/1；若只有 B/C/...，则选择排序后的第一个可用构象。
```

成功时 `match_method` 为：

```text
ATOM_SITE_COORD
ATOM_SITE_AUTH_ASYM_AUTH_SEQ
ATOM_SITE_AUTH_ASYM_LABEL_SEQ
ATOM_SITE_LABEL_ASYM_LABEL_SEQ
ATOM_SITE_LABEL_ASYM_AUTH_SEQ
```

## 5. 下载文件说明

### 5.1 RCSB full CIF

URL：

```text
https://files.rcsb.org/download/{PDB_ID}.cif
```

用途：

```text
1. 解析 _pdbx_nonpoly_scheme。
2. 解析 _pdbx_branch_scheme。
3. 解析 _atom_site 做 fallback 匹配、坐标校验和元素校验。
```

### 5.2 RCSB chemical component JSON

URL：

```text
https://data.rcsb.org/rest/v1/core/chemcomp/{CCD_ID}
```

关键字段：

```text
rcsb_chem_comp_descriptor.SMILES
rcsb_chem_comp_descriptor.SMILES_stereo
rcsb_chem_comp_descriptor.InChI
rcsb_chem_comp_descriptor.InChIKey
chem_comp.formula
chem_comp.formula_weight
```

### 5.3 RCSB native ligand mol2

URL 模板：

```text
https://models.rcsb.org/v1/{pdb_id}/ligand?auth_seq_id={pdb_seq_num}&label_asym_id={label_asym_id}&encoding=mol2&filename={filename}
```

本方案只接受 RCSB 原生 mol2：

```text
不允许 SDF -> mol2 转换
不允许 mmCIF -> mol2 转换
不允许 PubChem / Q-BioLiP / ChEBI fallback
```

文件命名：

```text
rcsb_ligand_mol2/{pdb_id}/{candidate_id}_{ccd_id}_{label_asym_id}_{pdb_seq_num}.mol2
```

## 6. mapping 字段

`ligand_mapping.csv` 和 `ligand_mapping.jsonl` 一行对应一个 Make_Data ligand candidate。

```text
emdb_id                                             # str, raw.json 中的 EMDB ID；未使用 raw.json 过滤时为空
pdb_id                                             # str, 小写 PDB ID，如 9lhb
pdb_id_upper                                       # str, 大写 PDB ID，如 9LHB
candidate_id                                       # int, Make_Data candidate ID；PDB 级失败或 raw.json 缺失记录为 -1
ligand_class_id                                    # int, Make_Data labels.npz 中的 ligand class ID
ccd_id                                             # str, Make_Data resname / RCSB CCD ID
chain_id                                           # str, Make_Data candidate chain_id
res_id                                             # int, Make_Data candidate res_id
insertion_code                                     # str, Make_Data insertion code；缺失为空
status                                             # str, PASS_HIGH 或失败/跳过状态码
match_method                                       # str, RCSB instance 匹配方法
label_asym_id                                      # str, RCSB label asym ID；ModelServer ligand endpoint 使用
pdb_strand_id                                      # str, RCSB/PDB chain ID
pdb_seq_num                                        # str, RCSB 页面可见 residue number；ModelServer auth_seq_id 参数实际使用该值
auth_seq_num                                       # str, author residue number；保留作审计，不作为 ModelServer 主参数
pdb_ins_code                                       # str, RCSB insertion code
smiles                                             # str, RCSB rcsb_chem_comp_descriptor.SMILES
smiles_stereo                                      # str, RCSB rcsb_chem_comp_descriptor.SMILES_stereo
inchi                                              # str, RCSB rcsb_chem_comp_descriptor.InChI
inchikey                                           # str, RCSB rcsb_chem_comp_descriptor.InChIKey
full_cif_path                                      # str, 下载后的 RCSB full CIF 路径
chemcomp_json_path                                 # str, RCSB chemcomp JSON 路径
native_mol2_path                                   # str, RCSB native mol2 路径
make_data_heavy_atoms                              # int, Make_Data candidate 重原子数
rcsb_cif_heavy_atoms                               # int, RCSB full CIF 中同一 ligand instance 重原子数
mol2_total_atoms                                   # int, native mol2 总原子数
mol2_heavy_atoms                                   # int, native mol2 重原子数
mol2_hydrogen_count                                # int, native mol2 氢原子数
mol2_bond_count                                    # int, native mol2 bond 条目数
mol2_has_charge_field                              # bool, mol2 atom 行是否包含可解析 charge
make_cif_coord_median                              # float, Make_Data vs RCSB CIF 最近邻距离 median，单位 A
make_cif_coord_max                                 # float, Make_Data vs RCSB CIF 最近邻距离 max，单位 A
cif_mol2_coord_median                              # float, RCSB CIF vs mol2 同元素最近邻距离 median，单位 A
cif_mol2_coord_max                                 # float, RCSB CIF vs mol2 同元素最近邻距离 max，单位 A
coord_status                                       # str, PASS / FAIL / SKIPPED
validation_errors                                  # str, 分号分隔的校验失败原因
docking_ready_warning                              # str, 分号分隔的 docking warning
dockem_input_status                                # str, DockEM 输入格式初筛状态
emerald_id_input_status                            # str, EMERALD-ID 输入格式初筛状态
pocketxmol_input_status                            # str, PocketXMol 输入格式初筛状态
download_url_full_cif                              # str, RCSB full CIF URL
download_url_chemcomp                              # str, RCSB chemcomp URL
download_url_mol2                                  # str, RCSB native mol2 URL
error_message                                      # str, 异常或失败原因摘要
source_file                                        # str, 该记录来源的总表或 part 文件路径
source_part_index                                  # int, array part 编号；非 array 总表为 -1
source_part_count                                  # int, array part 总数；非 array 总表为 -1
validation_detail                                  # dict/json, 结构化校验指标
download_attempts                                  # dict/json, full_cif/chemcomp/mol2 各自下载尝试次数
rcsb_nonpoly_scheme_row                            # dict/json, 匹配到的 scheme row 或 atom_site synthetic row
```

### 6.1 match_method 取值

`match_method` 记录 Make_Data candidate 与 RCSB ligand instance 是通过哪一路证据对齐的。

阅读原则：

```text
PDB_STRAND_*  表示来自 _pdbx_nonpoly_scheme，常见于普通非聚合物小分子。
BRANCH_*      表示来自 _pdbx_branch_scheme，常见于糖、支链糖类或 branch ligand。
ATOM_SITE_*   表示 scheme 表未能唯一命中，转而直接使用 _atom_site 分组。
FAILED        表示没有得到唯一 RCSB ligand instance。
```

详细含义：

| match_method | 来源 | 使用的关键字段 | 含义 | 可信度与注意事项 |
|---|---|---|---|---|
| `PDB_STRAND_PDB_SEQ` | `_pdbx_nonpoly_scheme` | `ccd_id == mon_id/pdb_mon_id/auth_mon_id`; `chain_id == pdb_strand_id`; `res_id == pdb_seq_num`; `insertion_code == pdb_ins_code` | Make_Data 的 residue number 与 RCSB/PDB 标准编号一致。 | 普通小分子最理想路径；通常与 RCSB 页面显示的 ligand residue number 一致。 |
| `PDB_STRAND_AUTH_SEQ` | `_pdbx_nonpoly_scheme` | 同上，但 `res_id == auth_seq_num` | Make_Data 的 residue number 更接近作者编号。 | 仍是 scheme 级匹配，但说明 PDB 标准编号与作者编号不同。 |
| `BRANCH_PDB_ASYM_PDB_SEQ` | `_pdbx_branch_scheme` | `ccd_id == mon_id/pdb_mon_id/auth_mon_id`; `chain_id == pdb_asym_id`; `res_id == pdb_seq_num` | ligand 是 branch/糖链类条目，且使用 PDB 标准编号命中。 | 修复糖类大规模失败的关键路径；NAG/MAN/BMA/FUC/GAL 等常走这里。 |
| `BRANCH_PDB_ASYM_AUTH_SEQ` | `_pdbx_branch_scheme` | 同上，但 `res_id == auth_seq_num` | ligand 是 branch/糖链类条目，且使用作者编号命中。 | 与 `BRANCH_PDB_ASYM_PDB_SEQ` 同级，但编号体系不同。 |
| `ATOM_SITE_COORD` | `_atom_site` | Make_Data `candidate_coords_{id}` 与 `_atom_site` ligand 分组坐标最近邻阈值唯一命中 | scheme 表不够用时，直接用坐标把 Make_Data ligand 锁定到 RCSB atom_site group。 | 几何证据强；若后续坐标校验失败，通常说明提取阶段或原始 Make_Data candidate 存在特殊情况。 |
| `ATOM_SITE_AUTH_ASYM_AUTH_SEQ` | `_atom_site` | `chain_id == auth_asym_id`; `res_id == auth_seq_id` | 用作者链 ID + 作者 residue number 唯一命中 atom_site group。 | 常用于 scheme 缺失但 author 标识完整的特殊 HETATM。 |
| `ATOM_SITE_AUTH_ASYM_LABEL_SEQ` | `_atom_site` | `chain_id == auth_asym_id`; `res_id == label_seq_id` | Make_Data chain 更像作者链 ID，residue number 更像 label 编号。 | 是 auth/label 标识混用时的保守 fallback。 |
| `ATOM_SITE_LABEL_ASYM_LABEL_SEQ` | `_atom_site` | `chain_id == label_asym_id`; `res_id == label_seq_id` | 用 RCSB label 链 ID + label residue number 唯一命中。 | 对 label 标识体系完整的条目有效。 |
| `ATOM_SITE_LABEL_ASYM_AUTH_SEQ` | `_atom_site` | `chain_id == label_asym_id`; `res_id == auth_seq_id` | Make_Data chain 更像 label 链 ID，residue number 更像作者编号。 | 是另一种 auth/label 混合 fallback。 |
| `FAILED` | 无唯一来源 | 无 | 找不到唯一 RCSB ligand instance。 | 最终验收中 class-4 `RCSB_INSTANCE_MATCH_FAILED = 0`，因此正式结果里不应再出现 class-4 对齐失败。 |

`ATOM_SITE_*` 的共同约束：

```text
1. 必须同时满足 CCD/resname、chain/residue/insertion_code 或坐标唯一性。
2. altLoc 优先保留空 alt、A、1；如果只有 B/C/...，选择排序后第一个可用构象。
3. 这些路径只解决“实例定位”，不自动保证 mol2、元素、坐标校验通过。
```

### 6.2 status 取值

```text
PASS_HIGH                       # SMILES/native mol2 获取成功，且全部校验通过
SKIPPED_NON_SMALL_MOLECULE      # 非 class-4 ligand，仅统计跳过
RAW_JSON_PDB_NOT_IN_PARSED_ROOT # raw.json 中 PDB 在 parsed_pdb 中不存在
PDB_LEVEL_FAILED                # PDB 级读取或 full CIF 下载失败
RCSB_INSTANCE_MATCH_FAILED      # Make_Data ligand 无法唯一匹配 RCSB ligand instance
RCSB_CHEMCOMP_DOWNLOAD_FAILED   # chemcomp JSON 下载失败
RCSB_SMILES_MISSING             # RCSB chemcomp 中缺少 SMILES
RCSB_NATIVE_MOL2_MISSING        # RCSB 原生 mol2 下载失败或返回空分子
VALIDATION_FAILED               # 文件存在但重原子、元素或坐标校验失败
```

### 6.3 工具输入状态取值

这三个字段只做 docking 输入格式初筛，不代表已经完成 docking，也不代表能量函数、密度图、口袋中心或受体结构已经准备完毕。

```text
dockem_input_status
emerald_id_input_status
pocketxmol_input_status
```

取值说明：

| 字段 | 取值 | 含义 |
|---|---|---|
| `dockem_input_status` | `FORMAT_OK` | RCSB native mol2 存在、可解析，且包含 DockEM 更希望看到的 H、bond、charge 信息。 |
| `dockem_input_status` | `FORMAT_OK_WITH_WARNING` | native mol2 存在且可作为原生结构文件使用，但有 `MOL2_HAS_NO_HYDROGEN`、`MOL2_HAS_NO_BONDS` 或 `MOL2_HAS_NO_COMPLETE_CHARGE_FIELD` 等 warning。 |
| `dockem_input_status` | `NOT_READY_NO_VALID_NATIVE_MOL2_PAIR` | 没有可用 native mol2，或该 candidate 未通过核心校验。 |
| `emerald_id_input_status` | `FORMAT_OK_FOR_LIBRARY_ENTRY` | 可以作为 EMERALD-ID 候选 ligand library 的结构条目参与后续准备。 |
| `emerald_id_input_status` | `NOT_READY_NO_VALID_NATIVE_MOL2_PAIR` | 没有可用 native mol2，不能进入当前严格单源 library。 |
| `pocketxmol_input_status` | `FORMAT_OK_SMILES_AND_STRUCTURE` | SMILES 与结构文件至少都已得到，满足 PocketXMol 输入准备的最小格式条件。 |
| `pocketxmol_input_status` | `NOT_READY_NO_VALID_SMILES_OR_STRUCTURE` | SMILES 或结构文件缺失/无效，不能进入当前严格输入集。 |

### 6.4 审计与来源字段

以下字段主要给 AI agent、质检脚本和失败诊断使用：

```text
source_file                                        # str, 当前行来自哪个 CSV/part CSV；merge 后总表中为 mapping/ligand_mapping.csv
source_part_index                                  # int, array part 编号；总表或非 array 运行为 -1
source_part_count                                  # int, array part 总数；总表或非 array 运行为 -1
validation_detail                                  # dict/json, 重原子数、元素组成、坐标距离等结构化校验细节
download_attempts                                  # dict/json, full_cif/chemcomp/mol2 各自下载尝试次数与最终状态
rcsb_nonpoly_scheme_row                            # dict/json, 历史字段名；实际可保存 nonpoly row、branch row 或 atom_site synthetic row
```

`rcsb_nonpoly_scheme_row` 的解释：

```text
1. match_method 为 PDB_STRAND_* 时，它来自 _pdbx_nonpoly_scheme。
2. match_method 为 BRANCH_* 时，它来自 _pdbx_branch_scheme，但被规范化成相同字段集合，方便后续统一读取。
3. match_method 为 ATOM_SITE_* 时，它是由 _atom_site 分组构造的 synthetic row，不是原始 scheme 行。
4. 因为早期字段名已经进入 mapping schema，为兼容旧脚本保留 rcsb_nonpoly_scheme_row 这个名字。
```

## 7. 校验规则

`PASS_HIGH` 需要同时满足：

```text
1. RCSB SMILES 或 stereo SMILES 存在。
2. RCSB native mol2 存在且可解析。
3. Make_Data 重原子数 == RCSB CIF 重原子数。
4. RCSB CIF 重原子数 == mol2 重原子数。
5. RCSB CIF 元素组成 == mol2 元素组成。
6. Make_Data vs RCSB CIF 坐标最近邻 median <= 0.05 A 且 max <= 0.20 A。
7. RCSB CIF vs mol2 同元素最近邻 median <= 0.05 A 且 max <= 0.20 A。
```

工具 readiness 只做格式初筛，不代表已经完成 docking：

```text
FORMAT_OK
FORMAT_OK_WITH_WARNING
FORMAT_OK_FOR_LIBRARY_ENTRY
FORMAT_OK_SMILES_AND_STRUCTURE
NOT_READY_NO_VALID_NATIVE_MOL2_PAIR
NOT_READY_NO_VALID_SMILES_OR_STRUCTURE
```

常见 warning：

```text
MOL2_HAS_NO_HYDROGEN
MOL2_HAS_NO_BONDS
MOL2_HAS_NO_COMPLETE_CHARGE_FIELD
```

## 8. summary 字段

`reports/validation_summary.json` 字段：

```text
raw_json_entries                                   # int, raw.json 原始条目数；未使用 raw.json 过滤时为 0；array merge 后取各 part 的最大值
unique_pdb_ids_in_raw_json                         # int, raw.json 中唯一 PDB ID 数；PDB ID 已统一大写去重；未使用 raw.json 时为 0
matched_parsed_pdb_count                           # int, raw.json 过滤后实际在 parsed_pdb 中找到并参与处理的 PDB 数；merge 时为各 part 求和
missing_parsed_pdb_count                           # int, raw.json 中存在但 parsed_pdb 根目录下缺失的 PDB 数；这些 PDB 会写 RAW_JSON_PDB_NOT_IN_PARSED_ROOT 记录
pdb_with_class4_ligands                            # int, 至少包含 1 个 ligand_class_ids == 4 candidate 的 PDB 数
total_ligand_records_from_labels                   # int, 从 labels.npz 读到的全部 ligand 记录数；包含 class-4 与非 class-4
class4_candidate_count                             # int, 本轮真正尝试 RCSB enrichment 的 small molecule candidate 数，即 ligand_class_ids == 4
skipped_non_small_molecule_count                   # int, ligand_class_ids != 4 的跳过记录数；这些记录不下载 CIF/chemcomp/mol2
status_counts                                      # dict[str, int], mapping.status 的计数；用于判断 PASS_HIGH、RCSB_NATIVE_MOL2_MISSING、VALIDATION_FAILED 等最终分布
dockem_input_status_counts                         # dict[str, int], DockEM 输入格式初筛状态计数；只判断 native mol2 是否存在、可解析、是否有 H/charge/bond warning
emerald_id_input_status_counts                     # dict[str, int], EMERALD-ID 输入格式初筛状态计数；用于候选 ligand library 条目格式可用性统计
pocketxmol_input_status_counts                     # dict[str, int], PocketXMol 输入格式初筛状态计数；要求 SMILES 与结构文件至少有一组可用
output_root                                        # str, 本次运行写出的输出根目录；正式服务器路径为 /storage/penghongen/CIF_Ligand
created_at                                         # str, summary 写出时间，本地时区下 ISO 字符串，如 2026-05-16T13:45:24
array_index                                        # int 或 null, array 运行时的当前分片编号；只存在于 part summary；总表通常无该字段
array_count                                        # int 或 null, array 运行时的总分片数；只存在于 part summary；总表通常无该字段
merged_from_array_count                            # int, merge_outputs 合并的 part 总数；只存在于 merge 后总 summary
```

统计口径补充：

```text
1. raw_json_entries / unique_pdb_ids_in_raw_json 描述外层过滤文件，不描述 parsed_pdb 实际存在情况。
2. matched_parsed_pdb_count + missing_parsed_pdb_count 是 raw.json 过滤后与 parsed_pdb 对齐的 PDB 级结果。
3. total_ligand_records_from_labels 是 Make_Data ligand 记录总量；class4_candidate_count 是其中真正下载 RCSB ligand 文件的子集。
4. skipped_non_small_molecule_count 不是失败，只是本轮明确不处理 class-2/3 或其它非 class-4 ligand。
5. status_counts 是主验收口径；三个 *_input_status_counts 是 docking 工具输入格式口径，不能替代 status_counts。
6. array part 的 summary 只覆盖该分片；merge 后总 summary 由所有 part 的 JSONL 重新统计得到。
```

最终验收 summary：

```text
class4_candidate_count = 190256
PASS_HIGH = 190093
RCSB_NATIVE_MOL2_MISSING = 156
VALIDATION_FAILED = 7
RCSB_INSTANCE_MATCH_FAILED = 0

DockEM ready = 190093
EMERALD-ID ready = 190093
PocketXMol ready = 190093
NOT_READY = 163 = 156 native mol2 missing + 7 validation hard cases
```

## 9. array part 与合并

array 模式写出：

```text
mapping/parts/ligand_mapping_part_{i}_of_{n}.csv
mapping/parts/ligand_mapping_part_{i}_of_{n}.jsonl
reports/parts/validation_summary_part_{i}_of_{n}.json
```

合并命令：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
python -m Ligand.rcsb_enrichment.merge_outputs \
  --output-root /storage/penghongen/CIF_Ligand \
  --array-count 5
```

若从任意目录手动运行，需要先进入仓库根目录或设置：

```bash
export PYTHONPATH=/home/penghongen/My_Project/Pocket_Plus:$PYTHONPATH
```

## 10. 最终残留失败

最终 7 条 `VALIDATION_FAILED` 是：

```text
6JLU / CLA / chain 18 / res 311
  Make_Data 重原子数为 5，RCSB CIF/native mol2 重原子数为 46。

7V68 / IXO / chain R / res 501
7V68 / 2CU / chain R / res 502
9O7S / 1KP / chain E / res 201
9O7S / 1KP / chain F / res 201
9O7S / 1KP / chain G / res 201
9O7S / 1KP / chain H / res 201
  重原子数一致，但 Make_Data vs RCSB CIF 坐标超过严格阈值。
```

最终 156 条 `RCSB_NATIVE_MOL2_MISSING` 主要来自 RCSB ModelServer 返回：

```text
HTTP 404: Could not find source file for 'pdb-bcif/{pdb_id}'
```

在“只接受 RCSB 原生 native mol2”的规则下，这些记录应保留为外部源缺失。
