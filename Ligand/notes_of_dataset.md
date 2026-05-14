# CIF_Ligand 数据集说明

本文记录 `Ligand/rcsb_enrichment` 自动化程序生成的 RCSB ligand enrichment 数据。该数据集的目标是为 Make_Data 中 `ligand_class_ids == 4` 的 small molecule ligand instance 补充 RCSB 单源的 SMILES、InChI/InChIKey 和原生 native mol2 文件，并保存 Make_Data ligand 与 RCSB ligand instance 的对应关系。

## 1. 服务器路径总览

正式产物统一写入：

```text
/storage/penghongen/CIF_Ligand/
```

| 路径 | 内容 | 生成入口 |
|---|---|---|
| `/storage/penghongen/CIF_Ligand/rcsb_full_cif/` | RCSB full structure CIF，每个 PDB 一个 `{PDB}.cif`。 | `Ligand/rcsb_enrichment/rcsb_client.py::download_full_cif()` |
| `/storage/penghongen/CIF_Ligand/rcsb_chemcomp_cache/` | RCSB chemical component JSON，每个 CCD 一个 `{CCD}.json`。 | `Ligand/rcsb_enrichment/rcsb_client.py::download_chemcomp()` |
| `/storage/penghongen/CIF_Ligand/rcsb_ligand_mol2/{pdb_id}/` | RCSB native ligand mol2，每个 Make_Data class-4 candidate 一个 mol2。 | `Ligand/rcsb_enrichment/rcsb_client.py::download_native_mol2()` |
| `/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv` | 非 array 模式或 merge 后的总映射表，一行对应一个 Make_Data ligand candidate。 | `Ligand/rcsb_enrichment/io_utils.py::write_mapping_outputs()` |
| `/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.jsonl` | 与 CSV 同语义的 JSONL 总映射表，保留嵌套字段。 | `Ligand/rcsb_enrichment/io_utils.py::write_mapping_outputs()` |
| `/storage/penghongen/CIF_Ligand/mapping/failed_cases.csv` | 总表中非 `PASS_HIGH` 且非 `SKIPPED_NON_SMALL_MOLECULE` 的失败样本。 | `Ligand/rcsb_enrichment/io_utils.py::write_mapping_outputs()` |
| `/storage/penghongen/CIF_Ligand/mapping/parts/` | sbatch array 模式下每个 array task 的 part CSV/JSONL。 | `Ligand/rcsb_enrichment/run_enrichment.py` |
| `/storage/penghongen/CIF_Ligand/reports/validation_summary.json` | 非 array 模式或 merge 后的总统计摘要。 | `Ligand/rcsb_enrichment/io_utils.py::build_summary()` |
| `/storage/penghongen/CIF_Ligand/reports/parts/` | sbatch array 模式下每个 array task 的 part summary。 | `Ligand/rcsb_enrichment/run_enrichment.py` |

## 2. 输入来源

### 2.1 raw.json 外层过滤

默认读取：

```text
/home/penghongen/My_Project/Data/raw.json
```

格式为：

```text
list[dict[str, str]]
```

每个元素形如：

```json
{"emd_63092": "9LHB"}
```

处理规则：

```text
1. PDB ID 统一大写后与 parsed_pdb 子目录匹配。
2. raw.json 缺省或 CLI 传入空字符串时不过滤，直接扫描 parsed_pdb 全部子目录。
3. raw.json 文件存在但内容为 [] 时，表示显式空过滤集，处理 0 个样本。
```

### 2.2 Make_Data parsed_pdb

默认读取：

```text
/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/{pdb_id}/
```

当前程序使用：

```text
labels.npz      # 读取 ligand_candidate_ids 与 ligand_class_ids
candidates.npz  # 回查 resnames/chain_ids/res_ids/insertion_codes/n_heavy_atoms/candidate_coords_{id}
```

只处理：

```text
ligand_class_ids == 4
```

其它 ligand 只写入 `SKIPPED_NON_SMALL_MOLECULE` 统计，不下载 RCSB ligand 文件。

## 3. 文件级说明

### 3.1 `rcsb_full_cif/{PDB}.cif`

RCSB full structure CIF。

用途：

```text
1. 读取 _pdbx_nonpoly_scheme 建立 Make_Data ligand 与 RCSB ligand instance 的对应关系。
2. 读取 _atom_site 提取 RCSB full CIF 中同一 ligand instance 的重原子坐标和元素组成。
```

### 3.2 `rcsb_chemcomp_cache/{CCD}.json`

RCSB chemical component descriptor 缓存。

关键字段：

```text
rcsb_chem_comp_descriptor.SMILES         # str, RCSB 常规 SMILES
rcsb_chem_comp_descriptor.SMILES_stereo  # str, RCSB stereo SMILES
rcsb_chem_comp_descriptor.InChI          # str, InChI
rcsb_chem_comp_descriptor.InChIKey       # str, InChIKey
chem_comp.formula                        # str, 化学式
chem_comp.formula_weight                 # float 或 str, 分子量
```

### 3.3 `rcsb_ligand_mol2/{pdb_id}/{candidate_id}_{ccd_id}_{label_asym_id}_{pdb_seq_num}.mol2`

RCSB native ligand mol2。

命名字段：

```text
candidate_id   # int, Make_Data candidate ID
ccd_id         # str, 配体 CCD ID
label_asym_id  # str, RCSB _pdbx_nonpoly_scheme.asym_id
pdb_seq_num    # str, RCSB _pdbx_nonpoly_scheme.pdb_seq_num
```

注意：

```text
1. 第一版只接受 RCSB 原生 mol2，不允许 SDF/mmCIF 转 mol2。
2. 部分 mol2 可能没有氢原子，这会在 docking_ready_warning 中标记为 MOL2_HAS_NO_HYDROGEN。
3. 单原子 ligand 可能没有 bond，这会标记为 MOL2_HAS_NO_BONDS。
```

## 4. mapping 字段说明

`ligand_mapping.csv` 与 `ligand_mapping.jsonl` 一行对应一个 Make_Data ligand candidate。

```text
emdb_id                                             # str, raw.json 中的 EMDB ID；未使用 raw.json 过滤时为空字符串
pdb_id                                             # str, 小写 PDB ID，如 9lhb
pdb_id_upper                                       # str, 大写 PDB ID，如 9LHB
candidate_id                                       # int, Make_Data candidate ID；PDB 级失败或 raw.json 缺失记录为 -1
ligand_class_id                                    # int, Make_Data labels.npz 中的 ligand class ID
ccd_id                                             # str, Make_Data resname / RCSB CCD ID
chain_id                                           # str, Make_Data candidate chain_id
res_id                                             # int, Make_Data candidate res_id
insertion_code                                     # str, Make_Data insertion code，缺失为空字符串
status                                             # str, PASS_HIGH 或失败/跳过状态码
match_method                                       # str, PDB_STRAND_PDB_SEQ / PDB_STRAND_AUTH_SEQ / FAILED
label_asym_id                                      # str, RCSB _pdbx_nonpoly_scheme.asym_id
pdb_strand_id                                      # str, RCSB _pdbx_nonpoly_scheme.pdb_strand_id
pdb_seq_num                                        # str, RCSB _pdbx_nonpoly_scheme.pdb_seq_num
auth_seq_num                                       # str, RCSB _pdbx_nonpoly_scheme.auth_seq_num
pdb_ins_code                                       # str, RCSB _pdbx_nonpoly_scheme.pdb_ins_code
smiles                                             # str, RCSB rcsb_chem_comp_descriptor.SMILES
smiles_stereo                                      # str, RCSB rcsb_chem_comp_descriptor.SMILES_stereo
inchi                                              # str, RCSB rcsb_chem_comp_descriptor.InChI
inchikey                                           # str, RCSB rcsb_chem_comp_descriptor.InChIKey
full_cif_path                                      # str, 下载后的 RCSB full CIF 路径
chemcomp_json_path                                 # str, RCSB chemcomp JSON 路径
native_mol2_path                                   # str, RCSB native mol2 路径
make_data_heavy_atoms                              # int, Make_Data candidate 重原子数
rcsb_cif_heavy_atoms                               # int, RCSB full CIF 同一 ligand instance 重原子数
mol2_total_atoms                                   # int, native mol2 总原子数
mol2_heavy_atoms                                   # int, native mol2 重原子数
mol2_hydrogen_count                                # int, native mol2 氢原子数
mol2_bond_count                                    # int, native mol2 bond 条目数
mol2_has_charge_field                              # bool, mol2 atom 行是否包含可解析 charge
make_cif_coord_median                              # float, Make_Data vs RCSB CIF 最近邻距离 median，单位 Å
make_cif_coord_max                                 # float, Make_Data vs RCSB CIF 最近邻距离 max，单位 Å
cif_mol2_coord_median                              # float, RCSB CIF vs mol2 同元素最近邻距离 median，单位 Å
cif_mol2_coord_max                                 # float, RCSB CIF vs mol2 同元素最近邻距离 max，单位 Å
coord_status                                       # str, PASS / FAIL / SKIPPED
validation_errors                                  # str, 分号分隔的校验失败原因
docking_ready_warning                              # str, 分号分隔的 docking warning
dockem_input_status                                # str, DockEM 输入格式检查状态
emerald_id_input_status                            # str, EMERALD-ID 输入格式检查状态
pocketxmol_input_status                            # str, PocketXMol 输入格式检查状态
download_url_full_cif                              # str, RCSB full CIF URL
download_url_chemcomp                              # str, RCSB chemcomp URL
download_url_mol2                                  # str, RCSB native mol2 URL
error_message                                      # str, 异常或失败原因摘要
source_file                                        # str, 该记录来源的总表或 part 文件路径
source_part_index                                  # int, array part 编号；非 array 总表为 -1
source_part_count                                  # int, array part 总数；非 array 总表为 -1
validation_detail                                  # dict/json, 结构化校验指标
download_attempts                                  # dict/json, full_cif/chemcomp/mol2 各自下载尝试次数
rcsb_nonpoly_scheme_row                            # dict/json, 匹配到的 _pdbx_nonpoly_scheme 原始字段
```

## 5. 状态码说明

```text
PASS_HIGH                       # RCSB SMILES 和 native mol2 获取成功，且全部校验通过
SKIPPED_NON_SMALL_MOLECULE      # 非 class-4 ligand，仅统计跳过
RAW_JSON_PDB_NOT_IN_PARSED_ROOT # raw.json 中 PDB 在 parsed_pdb 中不存在
PDB_LEVEL_FAILED                # PDB 级读取或 full CIF 下载失败
RCSB_INSTANCE_MATCH_FAILED      # Make_Data ligand 无法唯一匹配 RCSB ligand instance
RCSB_CHEMCOMP_DOWNLOAD_FAILED   # chemcomp JSON 下载失败
RCSB_SMILES_MISSING             # RCSB chemcomp 中缺少 SMILES
RCSB_NATIVE_MOL2_MISSING        # RCSB 原生 mol2 下载失败或返回空分子
VALIDATION_FAILED               # 文件存在但重原子、元素或坐标校验失败
```

## 6. summary 字段说明

`reports/validation_summary.json` 字段：

```text
raw_json_entries                                   # int, raw.json 原始条目数；未使用 raw.json 时为 0
unique_pdb_ids_in_raw_json                         # int, raw.json 中唯一 PDB 数；未使用 raw.json 时为 0
matched_parsed_pdb_count                           # int, 实际参与处理的 PDB 数
missing_parsed_pdb_count                           # int, raw.json 中存在但 parsed_pdb 缺失的 PDB 数
pdb_with_class4_ligands                            # int, 至少有 1 个 class-4 ligand 的 PDB 数
total_ligand_records_from_labels                   # int, labels.npz 中 ligand 记录总数
class4_candidate_count                             # int, class-4 ligand candidate 数
skipped_non_small_molecule_count                   # int, 非 class-4 跳过数
status_counts                                      # dict[str, int], 各 status 计数
dockem_input_status_counts                         # dict[str, int], DockEM 格式检查状态计数
emerald_id_input_status_counts                     # dict[str, int], EMERALD-ID 格式检查状态计数
pocketxmol_input_status_counts                     # dict[str, int], PocketXMol 格式检查状态计数
output_root                                        # str, 输出根目录
created_at                                         # str, 运行结束时间
array_index                                        # int 或 null, 当前 array 分片编号
array_count                                        # int 或 null, array 分片总数
merged_from_array_count                            # int, merge 输出中记录合并的 part 总数
```

## 7. array part 与读取兼容

array 模式下，每个任务写：

```text
mapping/parts/ligand_mapping_part_{i}_of_{n}.csv
mapping/parts/ligand_mapping_part_{i}_of_{n}.jsonl
reports/parts/validation_summary_part_{i}_of_{n}.json
```

合并命令：

```bash
python -m Ligand.rcsb_enrichment.merge_outputs \
  --output-root /storage/penghongen/CIF_Ligand \
  --array-count 5
```

读取兼容规则：

```text
1. 若 mapping/ligand_mapping.jsonl 存在，优先读取总表。
2. 若总表不存在，读取 mapping/parts/ligand_mapping_part_*_of_*.jsonl。
3. 每行保留 source_file / source_part_index / source_part_count，明确记录来源。
```
