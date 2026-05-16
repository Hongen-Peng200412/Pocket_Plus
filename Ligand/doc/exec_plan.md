# RCSB ligand enrichment 执行记录

本文记录 `Ligand/rcsb_enrichment` 从实现到服务器全量验收的执行过程、关键调试点和最终结果。

本文偏“做了什么、怎么修、结果如何”。为什么选择这条路线见 `Ligand/doc/explore.md`；字段和产物说明见 `Ligand/notes_of_dataset.md`；正式实现计划见 `Ligand/doc/implement_plan.md`。

## 1. 执行目标

在不重做 Make_Data 的前提下，新增独立 enrichment 程序：

```text
遍历 raw.json 过滤后的 parsed_pdb 样本
只处理 ligand_class_ids == 4 的 small molecule
对每个 Make_Data candidate 建立 RCSB ligand instance 映射
下载 RCSB full CIF / chemcomp JSON / native mol2
校验 Make_Data、RCSB CIF、native mol2 三者一致性
输出 mapping.csv / mapping.jsonl / failed_cases.csv / validation_summary.json
```

硬约束：

```text
1. 本轮限定 RCSB 单源。
2. native mol2 必须来自 RCSB ModelServer，不允许转换生成。
3. 默认复用已有缓存，只有 --force-download 才覆盖。
4. array 并发写 part 文件，不并发写总表。
5. 下载写入使用临时文件 + 原子替换。
6. 按 PDB 维度 joblib 并行。
```

## 2. 新增程序结构

核心代码：

```text
Ligand/rcsb_enrichment/
  __init__.py
  config.py
  models.py
  io_utils.py
  rcsb_client.py
  mmcif_mapping.py
  mol2_parser.py
  validation.py
  worker.py
  run_enrichment.py
  merge_outputs.py
```

提交脚本：

```text
Ligand/sbatch/rcsb_ligand_enrichment_raw4.sbatch
Ligand/sbatch/merge_rcsb_ligand_outputs.sh
```

说明文档：

```text
Ligand/notes_of_dataset.md
Ligand/doc/explore.md
Ligand/doc/implement_plan.md
Ligand/doc/exec_plan.md
Ligand/readme.md
```

## 3. 运行入口

array 任务：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
python -m Ligand.rcsb_enrichment.run_enrichment \
  --raw-json /home/penghongen/My_Project/Data/raw.json \
  --parsed-root /home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb \
  --output-root /storage/penghongen/CIF_Ligand \
  --array-index ${SLURM_ARRAY_TASK_ID} \
  --array-count ${SLURM_ARRAY_TASK_COUNT} \
  --n-jobs ${SLURM_CPUS_PER_TASK}
```

合并任务：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
python -m Ligand.rcsb_enrichment.merge_outputs \
  --output-root /storage/penghongen/CIF_Ligand \
  --array-count 5
```

如果不在仓库根目录运行，需要设置：

```bash
export PYTHONPATH=/home/penghongen/My_Project/Pocket_Plus:$PYTHONPATH
```

## 4. 本地验证

本地先用用户指定样本验证：

```text
3j7a
5bki
5gmk
5i68
```

随后扩展到本地下载的 parsed_pdb 子集。

结果：

```text
class-4 ligand instances = 58
PASS_HIGH = 58
失败 = 0
```

验证覆盖：

```text
1. 非 array 单样本/多样本运行。
2. array part 写出。
3. merge_outputs 合并。
4. raw.json 为空列表时处理 0 样本。
5. compileall 语法检查。
```

## 5. 服务器问题与修复

### 5.1 conda activate 与 set -u

问题：

```text
/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/etc/conda/activate.d/activate-binutils_linux-64.sh:
ADDR2LINE: unbound variable
```

原因：

```text
sbatch 中 set -u 与 conda activate.d 脚本不兼容。
```

修复：

```text
sbatch 使用 set -eo pipefail，不启用 nounset。
```

### 5.2 merge 找不到 Ligand package

问题：

```text
ModuleNotFoundError: No module named 'Ligand'
```

原因：

```text
手动 merge 时没有 cd 到 /home/penghongen/My_Project/Pocket_Plus。
Ligand 是仓库内源码包，不是 site-packages 中安装好的包。
```

修复：

```text
新增 Ligand/sbatch/merge_rcsb_ligand_outputs.sh。
文档中明确 merge 前需要 cd 到仓库根目录或设置 PYTHONPATH。
```

### 5.3 merge summary 原始计数错误

问题：

```text
merge 后 raw_json_entries / unique_pdb_ids_in_raw_json 曾显示 0。
```

原因：

```text
merge_outputs.py 仅从合并后的 rows 反推部分字段。
```

修复：

```text
merge_outputs.py 读取 reports/parts/validation_summary_part_*_of_n.json 聚合 raw/matched/missing 等 summary 字段。
```

## 6. 大规模失败诊断与修复

### 6.1 第一轮：糖类 branch scheme

现象：

```text
RCSB_INSTANCE_MATCH_FAILED = 44313
其中 98.6% 是 NAG/MAN/BMA/FUC/GAL 等糖类或支链糖类。
```

原因：

```text
第一版只查 _pdbx_nonpoly_scheme。
糖链 monomer 通常在 _pdbx_branch_scheme。
```

修复：

```text
parse_branch_scheme()
BRANCH_PDB_ASYM_PDB_SEQ
BRANCH_PDB_ASYM_AUTH_SEQ
```

验证样本：

```text
6HUG / NAG / chain F / res 1
```

### 6.2 第二轮：mol2 元素推断

现象：

```text
BEF 等无机 ligand validation failed。
```

原因：

```text
mol2 atom type BE 被旧逻辑误判为 B。
```

修复：

```text
mol2_parser.py 使用完整周期表推断元素。
```

验证样本：

```text
6AP1 / BEF
```

### 6.3 第三轮：atom_site fallback

现象：

```text
剩余 1.4% 非糖类 RCSB_INSTANCE_MATCH_FAILED。
```

原因：

```text
一些 HETATM/修饰残基不在 nonpoly/branch scheme，但在 _atom_site 中可定位。
```

修复：

```text
atom_site_groups_for_comp()
match_atom_site_ligand()
ATOM_SITE_COORD
ATOM_SITE_AUTH_ASYM_AUTH_SEQ
ATOM_SITE_AUTH_ASYM_LABEL_SEQ
ATOM_SITE_LABEL_ASYM_LABEL_SEQ
ATOM_SITE_LABEL_ASYM_AUTH_SEQ
```

代表样本：

```text
6AP1 / ACE
6VMI / Y5P
6VMI / P5P
7FGI / GTA
8CEP / KBE-DPP-UAL-MYN
9IF4 / S0R
```

### 6.4 第四轮：branch/atom_site 序号过滤过宽

现象：

```text
VALIDATION_FAILED = 38718
大量 NAG/MAN/FRU 等糖类重原子数不一致。
```

原因：

```text
同一个 label_asym_id 下多个相同 CCD monomer 被一起提取。
```

修复：

```text
extract_rcsb_atom_site_ligand() 只用 atom row 自己的 auth_seq_id/label_seq_id 与目标 pdb_seq_num/res_id 相交。
不再把目标 row 的序号塞进每个 atom row 的候选集合。
```

### 6.5 第五轮：altLoc 选择过窄

现象：

```text
BCL/CLF/HE2/FRU 等 ligand CIF 提取为 0 个重原子。
```

原因：

```text
部分 instance 只有 label_alt_id=B，旧逻辑只接受空 alt/A/1。
```

修复：

```text
preferred_alt_ids()
优先空 alt + A/1。
若只有 B/C/...，选择排序后的第一个可用 altLoc。
```

代表样本：

```text
7Z6Q / BCL
8DBY / CLF
8XGG / HE2
8UVU / FRU
```

### 6.6 第六轮：branch auth_seq_num 不能用于 CIF instance 提取

现象：

```text
VALIDATION_FAILED = 227
其中 220 条来自 branch 糖链坐标错配。
```

原因：

```text
_pdbx_branch_scheme.auth_seq_num 可能与 pdb_seq_num 顺序相反。
extract 阶段若把 auth_seq_num 放进目标序号集合，会选到相邻 monomer。
```

修复：

```text
extract_rcsb_atom_site_ligand() 目标序号集合只保留 ligand.res_id 与 match.row.pdb_seq_num。
```

代表样本：

```text
8AA3 / FRU
8G3R / MAN
8G6U / MAN
```

### 6.7 第七轮：native mol2 missing 的 readiness 字段

现象：

```text
RCSB_NATIVE_MOL2_MISSING = 156
但 NOT_READY_NO_VALID_NATIVE_MOL2_PAIR 只统计到 validation failed。
```

原因：

```text
native mol2 下载失败时，row.status 已更新，但 DockEM/EMERALD-ID/PocketXMol readiness 字段未填。
```

修复：

```text
worker.py 在 mol2.error_message 或 mol2 空分子时，调用 evaluate_tool_readiness(None, has_smiles, False)。
```

### 6.8 第八轮：ATOM_SITE_COORD 二次提取漂移

现象：

```text
9L5S 的 6 条记录 match_method=ATOM_SITE_COORD，但 validation 距离仍为 37-53 A。
```

原因：

```text
匹配阶段用 Make_Data 坐标唯一化找到了正确 _atom_site group；
extract 阶段又用 synthetic row 的 label_asym/seq 重新提取，重复实例中可能漂回另一个 group。
```

修复：

```text
当 match_method == ATOM_SITE_COORD 时，extract 阶段也直接用 Make_Data 坐标唯一化返回同一个 _atom_site group。
```

结果：

```text
9L5S 的 6 条 validation failed 消失。
```

## 7. 最终服务器验收

最终 merge summary：

```text
class4_candidate_count = 190256
PASS_HIGH = 190093
RCSB_NATIVE_MOL2_MISSING = 156
VALIDATION_FAILED = 7
RCSB_INSTANCE_MATCH_FAILED = 0
SKIPPED_NON_SMALL_MOLECULE = 192839
RAW_JSON_PDB_NOT_IN_PARSED_ROOT = 648
```

工具 readiness：

```text
DockEM:
  FORMAT_OK = 16222
  FORMAT_OK_WITH_WARNING = 173871
  NOT_READY_NO_VALID_NATIVE_MOL2_PAIR = 163

EMERALD-ID:
  FORMAT_OK_FOR_LIBRARY_ENTRY = 190093
  NOT_READY_NO_VALID_NATIVE_MOL2_PAIR = 163

PocketXMol:
  FORMAT_OK_SMILES_AND_STRUCTURE = 190093
  NOT_READY_NO_VALID_SMILES_OR_STRUCTURE = 163
```

成功率：

```text
全量 class-4 口径:
190093 / 190256 = 99.914%

排除 RCSB ModelServer native mol2 外部缺失口径:
190093 / (190256 - 156) = 99.996%
```

## 8. 最终残留失败

### 8.1 RCSB native mol2 缺失

数量：

```text
156
```

典型错误：

```text
HTTP 404: Error: Could not find source file for 'pdb-bcif/{pdb_id}'.
```

集中 PDB：

```text
6IXA
9DZU
8D68
8XNH
8T0Q
8XN9
8V2I
7ZK7
7FEN
8UIK
8Q4F
7QVG
6EK0
7M5R
7AA6
7M5Q
```

解释：按“只接受 RCSB native mol2”的规则，这些是外部源不可用，不算映射流程失败。

### 8.2 validation hard cases

数量：

```text
7
```

具体样本：

```text
6JLU / CLA / chain 18 / res 311
  Make_Data 重原子数为 5，RCSB CIF/native mol2 为 46。

7V68 / IXO / chain R / res 501
7V68 / 2CU / chain R / res 502
9O7S / 1KP / chain E / res 201
9O7S / 1KP / chain F / res 201
9O7S / 1KP / chain G / res 201
9O7S / 1KP / chain H / res 201
  重原子数一致，但 Make_Data vs RCSB CIF 坐标超过严格阈值。
```

## 9. 结论

RCSB 单源短期 enrichment 路线已验收通过。

关键结论：

```text
1. Make_Data ligand -> RCSB ligand instance 映射已打通，最终 RCSB_INSTANCE_MATCH_FAILED = 0。
2. RCSB native mol2 可获得条目中，PASS_HIGH = 99.996%。
3. 即使把 RCSB native mol2 外部缺失纳入总分母，PASS_HIGH = 99.914%。
4. 剩余失败可解释、可审计，不构成路线失败。
```

本阶段不做：

```text
1. 不接入 Q-BioLiP。
2. 不处理 ligand_class_ids 2/3。
3. 不用 PubChem/ChEBI/OpenBabel/RDKit 生成 fallback mol2。
4. 不运行真实 docking，只做 DockEM/EMERALD-ID/PocketXMol 输入格式初筛。
```

## 10. 执行字段审计补充

本节解释执行记录中反复出现的枚举和统计字段。完整字段索引以 `Ligand/notes_of_dataset.md` 为准；这里强调这些值在多轮运行中如何产生、如何解读。

### 10.1 match_method

`match_method` 是 Make_Data candidate 与 RCSB ligand instance 的对齐证据来源。

```text
PDB_STRAND_PDB_SEQ:
  来自 _pdbx_nonpoly_scheme。
  用 ccd_id + pdb_strand_id + pdb_seq_num + insertion_code 唯一匹配。
  普通非聚合物小分子的首选路径。

PDB_STRAND_AUTH_SEQ:
  来自 _pdbx_nonpoly_scheme。
  与 PDB_STRAND_PDB_SEQ 类似，但 residue number 使用 auth_seq_num。
  表示 Make_Data res_id 更接近作者编号。

BRANCH_PDB_ASYM_PDB_SEQ:
  来自 _pdbx_branch_scheme。
  用 ccd_id + pdb_asym_id + pdb_seq_num 唯一匹配。
  主要修复 NAG/MAN/BMA/FUC/GAL 等糖类或支链糖类。

BRANCH_PDB_ASYM_AUTH_SEQ:
  来自 _pdbx_branch_scheme。
  与 BRANCH_PDB_ASYM_PDB_SEQ 类似，但 residue number 使用 auth_seq_num。

ATOM_SITE_COORD:
  来自 _atom_site。
  先按 Make_Data candidate_coords_{id} 与 atom_site 分组做坐标最近邻唯一匹配。
  用于 scheme 表缺失或不够唯一的特殊 ligand。

ATOM_SITE_AUTH_ASYM_AUTH_SEQ:
  来自 _atom_site。
  用 auth_asym_id + auth_seq_id 唯一匹配。

ATOM_SITE_AUTH_ASYM_LABEL_SEQ:
  来自 _atom_site。
  用 auth_asym_id + label_seq_id 唯一匹配。

ATOM_SITE_LABEL_ASYM_LABEL_SEQ:
  来自 _atom_site。
  用 label_asym_id + label_seq_id 唯一匹配。

ATOM_SITE_LABEL_ASYM_AUTH_SEQ:
  来自 _atom_site。
  用 label_asym_id + auth_seq_id 唯一匹配。

FAILED:
  未找到唯一 RCSB ligand instance。
  最终验收中 class-4 RCSB_INSTANCE_MATCH_FAILED = 0，因此正式结果不应再有 class-4 对齐失败。
```

本轮关键执行经验：

```text
1. 只读 _pdbx_nonpoly_scheme 会遗漏大量糖/支链糖类。
2. 加入 _pdbx_branch_scheme 后，NAG/MAN/BMA/FUC/GAL 等主失败簇被解决。
3. 剩余 scheme 难例需要 _atom_site fallback；其中坐标唯一化比纯 auth/label 编号更强。
4. ATOM_SITE_COORD 命中后，提取阶段必须复用同一个 atom_site group，不能重新用宽松编号条件提取。
```

### 10.2 validation_summary.json

summary 字段分三类：

```text
输入过滤口径:
  raw_json_entries
  unique_pdb_ids_in_raw_json
  matched_parsed_pdb_count
  missing_parsed_pdb_count

ligand 处理口径:
  pdb_with_class4_ligands
  total_ligand_records_from_labels
  class4_candidate_count
  skipped_non_small_molecule_count
  status_counts

docking 格式初筛口径:
  dockem_input_status_counts
  emerald_id_input_status_counts
  pocketxmol_input_status_counts
```

逐项解释：

```text
raw_json_entries:
  raw.json 原始条目数。array merge 后取各 part summary 的最大值，因为每个 part 看到的是同一个 raw.json。

unique_pdb_ids_in_raw_json:
  raw.json 中唯一 PDB 数，按统一大写去重。

matched_parsed_pdb_count:
  raw.json 过滤后，在 parsed_pdb 根目录中实际找到并参与处理的 PDB 数。
  merge 时对各 part 求和。

missing_parsed_pdb_count:
  raw.json 中存在但 parsed_pdb 缺失的 PDB 数。
  这些记录写为 RAW_JSON_PDB_NOT_IN_PARSED_ROOT。

pdb_with_class4_ligands:
  至少有一个 ligand_class_ids == 4 candidate 的 PDB 数。

total_ligand_records_from_labels:
  从 labels.npz 读出的全部 ligand 记录数，包含 class-4 和非 class-4。

class4_candidate_count:
  本轮真正尝试 RCSB enrichment 的 small molecule candidate 数。

skipped_non_small_molecule_count:
  ligand_class_ids != 4 的跳过数量。它不是失败，而是本阶段明确不处理的范围。

status_counts:
  主验收口径。PASS_HIGH、RCSB_NATIVE_MOL2_MISSING、VALIDATION_FAILED 等都来自这里。

dockem_input_status_counts / emerald_id_input_status_counts / pocketxmol_input_status_counts:
  只描述工具输入格式是否具备基础文件条件，不等价于真实 docking 成功率。

output_root:
  本次运行输出根目录。

created_at:
  summary 写出时间。

array_index / array_count:
  array part summary 中记录当前分片编号和总分片数。

merged_from_array_count:
  merge 后总 summary 中记录合并了多少个 array part。
```

最终验收时需要优先看：

```text
1. status_counts.PASS_HIGH
2. status_counts.RCSB_INSTANCE_MATCH_FAILED
3. status_counts.RCSB_NATIVE_MOL2_MISSING
4. status_counts.VALIDATION_FAILED
5. 三个 *_input_status_counts 的 NOT_READY 数量是否等于严格失败数量
```
