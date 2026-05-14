# RCSB ligand enrichment 自动化实施计划

## 背景/目标

当前 `Make_Data` 已经为 `/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/` 生成每个 PDB 的 ligand candidate 与 label 信息，但没有保存小分子的 SMILES 和 docking 可用的 ligand mol2 文件。

前期探索已证明：**对本地样例中 `ligand_class_ids == 4` 的 small molecule，RCSB 单源路线可以高置信补充 SMILES 与 RCSB 原生 ligand mol2**。探索结果已记录在：

```text
Ligand/explore.md
```

本计划目标是把探索脚本升级为服务器可运行的正式自动化管线：

```text
/home/penghongen/My_Project/Data/raw.json
-> 过滤 parsed_pdb 中已有 PDB
-> 读取 Make_Data labels/candidates
-> 只处理 ligand_class_ids == 4
-> 下载 RCSB full CIF / chemcomp descriptor / native ligand mol2
-> 建立 Make_Data ligand <-> RCSB ligand instance 的对应关系
-> 执行重原子、元素组成、坐标一致性校验
-> 输出 CSV + JSONL mapping 与 summary
```

正式产物统一写入：

```text
/storage/penghongen/CIF_Ligand
```

## 不修改的部分

1. 不修改 `Make_Data/process_and_label.py` 的既有解析和标签生成逻辑。
2. 不修改已有 `DATA_v2_raw4/parsed_pdb/{pdb_id}` 下的 `.npz` 文件。
3. 第一版不处理 `ligand_class_ids == 2/3` 的 peptide/nucleic ligand。
4. 第一版不接入 Q-BioLiP、PubChem、ChEBI 或 DrugBank fallback。
5. 第一版不允许 SDF/mmCIF 转 mol2 作为成功结果，只接受 RCSB 原生 native ligand mol2。
6. 第一版不运行 DockEM、EMERALD-ID 或 PocketXMol，只做输入文件合规性检查。

## 设计决策

### RCSB 单源

本管线只依赖 RCSB：

| 数据                                         | 来源                                          |
| ------------------------------------------ | ------------------------------------------- |
| full structure CIF                         | `https://files.rcsb.org/download/{PDB}.cif` |
| SMILES / SMILES\_stereo / InChI / InChIKey | RCSB Data API `core/chemcomp/{CCD}`         |
| ligand native mol2                         | RCSB ModelServer ligand endpoint            |

### 按 PDB 并行

使用 `joblib.Parallel` 按 PDB 并行：

```text
每个 worker 处理 1 个 PDB
每个 PDB 内多个 ligand 共享 full CIF 与 mmCIF 解析结果
每个 worker 返回该 PDB 的多行 ligand mapping 和 skipped 统计
```

原因：

```text
减少 full CIF 重复下载
减少同一 PDB 的 mmCIF 重复解析
便于服务器上用 --n-jobs 对应 sbatch CPU 数
```

### 文件复用

默认复用已有下载文件：

```text
文件存在且大小 > 0，则跳过下载
仍然重新执行 mapping 与校验
```

提供：

```text
--force-download
```

用于覆盖下载。

### 匹配主键

每个输出条目对应一个 Make\_Data ligand candidate：

```text
pdb_id + candidate_id
```

RCSB 对齐使用：

```text
Make_Data resname  -> _pdbx_nonpoly_scheme.mon_id / pdb_mon_id / auth_mon_id
Make_Data chain_id -> _pdbx_nonpoly_scheme.pdb_strand_id
Make_Data res_id   -> _pdbx_nonpoly_scheme.pdb_seq_num
Make_Data insertion_code -> _pdbx_nonpoly_scheme.pdb_ins_code
```

匹配成功后保存：

```text
label_asym_id = _pdbx_nonpoly_scheme.asym_id
pdb_seq_num   = _pdbx_nonpoly_scheme.pdb_seq_num
```

用于 RCSB ModelServer ligand mol2 下载。

> \[!IMPORTANT]
> 探索中发现 RCSB ModelServer 参数名为 `auth_seq_id`，但对测试样例应传入 `_pdbx_nonpoly_scheme.pdb_seq_num`，即 RCSB 页面可见的 PDB residue number。正式代码中必须明确注释该约定。

## Proposed Changes

## 改动文件汇总

| 文件                                                 | 类型        | 作用                                                           |
| -------------------------------------------------- | --------- | ------------------------------------------------------------ |
| `Ligand/explore.md`                                | \[MODIFY] | 写入 RCSB 单源路线可行性的实验结论与证据                                      |
| `Ligand/implement_plan.md`                         | \[MODIFY] | 写入本实施计划                                                      |
| `Ligand/rcsb_enrichment/__init__.py`               | \[NEW]    | Python package 标记                                            |
| `Ligand/rcsb_enrichment/config.py`                 | \[NEW]    | 默认路径、RCSB URL、阈值和状态码常量                                       |
| `Ligand/rcsb_enrichment/io_utils.py`               | \[NEW]    | raw.json、npz、CSV/JSONL、下载缓存读写                                |
| `Ligand/rcsb_enrichment/rcsb_client.py`            | \[NEW]    | RCSB full CIF、chemcomp、ligand mol2 下载客户端                     |
| `Ligand/rcsb_enrichment/mmcif_mapping.py`          | \[NEW]    | 从 RCSB full CIF 中解析 ligand instance 并对齐 Make\_Data candidate |
| `Ligand/rcsb_enrichment/mol2_parser.py`            | \[NEW]    | 轻量解析 native mol2 的 atom/bond/charge 信息                       |
| `Ligand/rcsb_enrichment/validation.py`             | \[NEW]    | 重原子、元素组成、坐标、工具输入合规性校验                                        |
| `Ligand/rcsb_enrichment/worker.py`                 | \[NEW]    | 单 PDB 处理函数，供 joblib 调用                                       |
| `Ligand/rcsb_enrichment/run_enrichment.py`         | \[NEW]    | 统一 CLI 入口                                                    |
| `Ligand/sbatch/rcsb_ligand_enrichment_raw4.sbatch` | \[NEW]    | 服务器 sbatch 示例                                                |

### #### \[NEW] `Ligand/rcsb_enrichment/config.py`

定义默认路径、URL 模板和校验阈值。

新增常量：

| 常量                       | 类型      | 默认值                                                        | 含义                                                                   |
| ------------------------ | ------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| `DEFAULT_RAW_JSON`       | `str`   | `/home/penghongen/My_Project/Data/raw.json`                | :comment[EMDB-PDB 外层过滤文件]{#comment-1778761277948 text="允许它为空，表示不过滤"} |
| `DEFAULT_PARSED_ROOT`    | `str`   | `/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb` | Make\_Data parsed\_pdb 根目录                                           |
| `DEFAULT_OUTPUT_ROOT`    | `str`   | `/storage/penghongen/CIF_Ligand`                           | 所有下载与 mapping 输出根目录                                                  |
| `TARGET_LIGAND_CLASS_ID` | `int`   | `4`                                                        | 只处理 small molecule                                                   |
| `COORD_MEDIAN_THRESHOLD` | `float` | `0.05`                                                     | 坐标最近邻 median 阈值，单位 Å                                                 |
| `COORD_MAX_THRESHOLD`    | `float` | `0.20`                                                     | 坐标最近邻 max 阈值，单位 Å                                                    |
| `RCSB_FULL_CIF_URL`      | `str`   | `https://files.rcsb.org/download/{pdb_id}.cif`             | full CIF 下载模板                                                        |
| `RCSB_CHEMCOMP_URL`      | `str`   | `https://data.rcsb.org/rest/v1/core/chemcomp/{ccd_id}`     | chemcomp JSON 下载模板                                                   |
| `RCSB_LIGAND_MOL2_URL`   | `str`   | RCSB ModelServer ligand endpoint                           | native mol2 下载模板                                                     |

新增状态码常量：

| 状态码                               | 含义                                            |
| --------------------------------- | --------------------------------------------- |
| `PASS_HIGH`                       | RCSB SMILES 和 native mol2 获取成功，且全部校验通过        |
| `SKIPPED_NON_SMALL_MOLECULE`      | 非 class-4 ligand，仅统计跳过                        |
| `RAW_JSON_PDB_NOT_IN_PARSED_ROOT` | raw.json 中 PDB 在 parsed\_pdb 中不存在             |
| `PDB_LEVEL_FAILED`                | PDB 级读取或 full CIF 下载失败                        |
| `RCSB_INSTANCE_MATCH_FAILED`      | Make\_Data ligand 无法唯一匹配 RCSB ligand instance |
| `RCSB_CHEMCOMP_DOWNLOAD_FAILED`   | chemcomp JSON 下载失败                            |
| `RCSB_SMILES_MISSING`             | RCSB chemcomp 中缺少 SMILES                      |
| `RCSB_NATIVE_MOL2_MISSING`        | RCSB 原生 mol2 下载失败或返回空分子                       |
| `VALIDATION_FAILED`               | 文件存在但重原子、元素或坐标校验失败                            |

### #### \[NEW] `Ligand/rcsb_enrichment/io_utils.py`

负责输入输出和缓存文件管理。

新增函数：

```python
def load_raw_mapping(raw_json_path: Path) -> list[dict[str, str]]:
    """读取 raw.json，返回原始 List[Dict[emdb_id, pdb_id]]。"""
```

参数：

| 参数              | 类型     | 含义                                             |
| --------------- | ------ | ---------------------------------------------- |
| `raw_json_path` | `Path` | `/home/penghongen/My_Project/Data/raw.json` 路径 |

返回：

```text
list[dict[str, str]]
```

不变量：

```text
每个 dict 应只有一个 key-value；key 为 emdb_id，value 为 pdb_id。
```

```python
def resolve_pdb_samples(raw_mapping: list[dict[str, str]], parsed_root: Path) -> list[SampleRef]:
    """按 raw.json 外层过滤 parsed_pdb 中已有样本，PDB ID 统一大写匹配。"""
```

返回数据结构 `SampleRef`：

| 字段             | 类型     | 含义                                |
| -------------- | ------ | --------------------------------- |
| `emdb_id`      | `str`  | raw.json 中的 EMDB ID，如 `emd_63092` |
| `pdb_id`       | `str`  | 统一小写保存的 PDB ID，如 `9lhb`           |
| `pdb_id_upper` | `str`  | 统一大写 PDB ID，如 `9LHB`              |
| `parsed_dir`   | `Path` | `parsed_pdb/{pdb_id}` 实际目录        |

```python
def load_make_data_ligands(parsed_dir: Path) -> list[MakeDataLigand]:
    """读取 labels.npz 与 candidates.npz，返回该 PDB 中 labels 记录的 ligand instance。"""
```

返回数据结构 `MakeDataLigand`：

| 字段                      | 类型           | 含义                                                             |
| ----------------------- | ------------ | -------------------------------------------------------------- |
| `pdb_id`                | `str`        | PDB ID，小写                                                      |
| `candidate_id`          | `int`        | Make\_Data candidate ID                                        |
| `ligand_class_id`       | `int`        | `labels.npz` 中的 ligand class ID                                |
| `ccd_id`                | `str`        | `candidates.npz/resnames[candidate_id]`，大写                     |
| `chain_id`              | `str`        | `candidates.npz/chain_ids[candidate_id]`                       |
| `res_id`                | `int`        | `candidates.npz/res_ids[candidate_id]`                         |
| `insertion_code`        | `str`        | `candidates.npz/insertion_codes[candidate_id]`，`.`/`?` 归一为空字符串 |
| `make_data_heavy_atoms` | `int`        | `candidates.npz/n_heavy_atoms[candidate_id]`                   |
| `make_data_coords`      | `np.ndarray` | `candidate_coords_{candidate_id}`，形状 `(N_heavy, 3)`            |

```python
def write_mapping_outputs(rows: list[MappingRow], output_root: Path) -> None:
    """同时写出 mapping CSV、JSONL、summary JSON 和 failed cases CSV。"""
```

输出文件结构：

```text
/storage/penghongen/CIF_Ligand/
  mapping/
    ligand_mapping.csv
    ligand_mapping.jsonl
    failed_cases.csv
  reports/
    validation_summary.json
```

### #### \[NEW] `Ligand/rcsb_enrichment/rcsb_client.py`

负责 RCSB 网络下载、缓存复用和有限重试。

新增类：

```python
class RCSBClient:
    def __init__(self, output_root: Path, force_download: bool, max_retries: int, retry_sleep: float) -> None: ...
```

参数：

| 参数               | 类型      | 含义                               |
| ---------------- | ------- | -------------------------------- |
| `output_root`    | `Path`  | `/storage/penghongen/CIF_Ligand` |
| `force_download` | `bool`  | 是否覆盖已有下载文件                       |
| `max_retries`    | `int`   | 下载失败最大重试次数                       |
| `retry_sleep`    | `float` | 每次失败后的 sleep 秒数                  |

方法：

```python
def download_full_cif(self, pdb_id: str) -> Path:
    """下载或复用 full CIF，保存到 rcsb_full_cif/{PDB}.cif。"""
```

```python
def download_chemcomp(self, ccd_id: str) -> Path:
    """下载或复用 RCSB chemcomp JSON，保存到 rcsb_chemcomp_cache/{CCD}.json。"""
```

```python
def download_native_mol2(
    self,
    pdb_id: str,
    label_asym_id: str,
    pdb_seq_num: str,
    ccd_id: str,
    candidate_id: int,
) -> Path:
    """下载或复用 RCSB native ligand mol2。"""
```

native mol2 命名规则：

```text
rcsb_ligand_mol2/{pdb_id}/{candidate_id}_{ccd_id}_{label_asym_id}_{pdb_seq_num}.mol2
```

### #### \[NEW] `Ligand/rcsb_enrichment/mmcif_mapping.py`

负责解析 RCSB full CIF 并建立 Make\_Data ligand 到 RCSB ligand instance 的映射。

新增函数：

```python
def parse_nonpoly_scheme(cif_dict: dict[str, list[str]]) -> list[NonpolySchemeRow]:
    """解析 _pdbx_nonpoly_scheme。"""
```

`NonpolySchemeRow` 字段：

| 字段              | 类型    | 含义                                           |
| --------------- | ----- | -------------------------------------------- |
| `asym_id`       | `str` | RCSB label asym ID，用于 ModelServer            |
| `mon_id`        | `str` | CCD ID                                       |
| `pdb_seq_num`   | `str` | RCSB 页面可见 residue number                     |
| `auth_seq_num`  | `str` | 作者 residue number                            |
| `pdb_strand_id` | `str` | RCSB/PDB chain ID，与 Make\_Data `chain_id` 对齐 |
| `pdb_ins_code`  | `str` | insertion code                               |

```python
def match_make_data_ligand_to_rcsb(
    ligand: MakeDataLigand,
    rows: list[NonpolySchemeRow],
) -> RCSBInstanceMatch:
    """用 ccd_id + chain_id + res_id + insertion_code 唯一匹配 RCSB ligand instance。"""
```

匹配优先级：

```text
1. PDB_STRAND_PDB_SEQ:
   ccd_id == mon_id/pdb_mon_id/auth_mon_id
   chain_id == pdb_strand_id
   res_id == pdb_seq_num
   insertion_code == pdb_ins_code

2. PDB_STRAND_AUTH_SEQ:
   ccd_id == mon_id/pdb_mon_id/auth_mon_id
   chain_id == pdb_strand_id
   res_id == auth_seq_num
   insertion_code == pdb_ins_code

3. FAILED:
   无唯一匹配
```

```python
def extract_rcsb_atom_site_ligand(
    cif_dict: dict[str, list[str]],
    ligand: MakeDataLigand,
    match: RCSBInstanceMatch,
) -> RCSBAtomSiteLigand:
    """从 _atom_site 中提取匹配 ligand instance 的重原子坐标、元素和 atom name。"""
```

> \[!IMPORTANT]
> `_atom_site` 回取坐标时必须只使用已匹配到的 `label_asym_id`，不能同时接受 auth chain，否则同名链/重复 ligand 可能造成坐标翻倍。

### #### \[NEW] `Ligand/rcsb_enrichment/mol2_parser.py`

负责轻量解析 native mol2，避免第一版依赖 RDKit/OpenBabel。

新增函数：

```python
def parse_mol2(mol2_path: Path) -> Mol2Info:
    """解析 TRIPOS mol2 的 ATOM/BOND section。"""
```

`Mol2Info` 字段：

| 字段                        | 类型           | 含义                         |
| ------------------------- | ------------ | -------------------------- |
| `atom_count`              | `int`        | mol2 总原子数                  |
| `heavy_atom_count`        | `int`        | 非 H 原子数                    |
| `hydrogen_count`          | `int`        | H 原子数                      |
| `bond_count`              | `int`        | BOND section 条目数           |
| `has_charge_field`        | `bool`       | 每个 atom 行是否都有可解析 charge    |
| `heavy_coords`            | `np.ndarray` | 重原子坐标，形状 `(N_heavy, 3)`    |
| `heavy_elements`          | `list[str]`  | 重原子元素列表                    |
| `raw_has_tripos_molecule` | `bool`       | 文件是否包含 `@<TRIPOS>MOLECULE` |

### #### \[NEW] `Ligand/rcsb_enrichment/validation.py`

负责多角度校验和工具输入合规性判断。

新增函数：

```python
def validate_ligand_pair(
    make_data_ligand: MakeDataLigand,
    rcsb_atom_site: RCSBAtomSiteLigand,
    mol2_info: Mol2Info,
    descriptors: ChemCompDescriptors,
    thresholds: ValidationThresholds,
) -> ValidationMetrics:
    """执行重原子数、元素组成和坐标一致性校验。"""
```

`ValidationMetrics` 字段：

| 字段                      | 类型          | 含义                                    |                                   |
| ----------------------- | ----------- | ------------------------------------- | --------------------------------- |
| `make_data_heavy_atoms` | `int`       | Make\_Data candidate 重原子数             |                                   |
| `rcsb_cif_heavy_atoms`  | `int`       | RCSB full CIF 同一 ligand instance 重原子数 |                                   |
| `mol2_heavy_atoms`      | `int`       | native mol2 重原子数                      |                                   |
| `make_cif_coord_median` | \`float     | None\`                                | Make\_Data vs RCSB CIF 最近邻 median |
| `make_cif_coord_max`    | \`float     | None\`                                | Make\_Data vs RCSB CIF 最近邻 max    |
| `cif_mol2_coord_median` | \`float     | None\`                                | RCSB CIF vs mol2 同元素最近邻 median    |
| `cif_mol2_coord_max`    | \`float     | None\`                                | RCSB CIF vs mol2 同元素最近邻 max       |
| `coord_status`          | `str`       | `PASS` 或 `FAIL`                       |                                   |
| `validation_errors`     | `list[str]` | 失败原因列表                                |                                   |

新增函数：

```python
def evaluate_tool_readiness(mol2_info: Mol2Info, has_smiles: bool) -> ToolReadiness:
    """检查 DockEM / EMERALD-ID / PocketXMol 的输入格式可用性。"""
```

输出状态：

| 字段                        | 含义                                                                             |
| ------------------------- | ------------------------------------------------------------------------------ |
| `dockem_input_status`     | `FORMAT_OK` / `FORMAT_OK_WITH_WARNING` / `NOT_READY_NO_VALID_NATIVE_MOL2_PAIR` |
| `emerald_id_input_status` | `FORMAT_OK_FOR_LIBRARY_ENTRY` / `NOT_READY_NO_VALID_NATIVE_MOL2_PAIR`          |
| `pocketxmol_input_status` | `FORMAT_OK_SMILES_AND_STRUCTURE` / `NOT_READY_NO_VALID_SMILES_OR_STRUCTURE`    |
| `docking_ready_warning`   | 如 `MOL2_HAS_NO_HYDROGEN;MOL2_HAS_NO_BONDS`                                     |

### #### \[NEW] `Ligand/rcsb_enrichment/worker.py`

负责单 PDB 的完整处理。

新增函数：

```python
def process_one_pdb(
    sample: SampleRef,
    output_root: Path,
    force_download: bool,
    max_retries: int,
    retry_sleep: float,
) -> PDBProcessResult:
    """处理一个 PDB，返回该 PDB 的所有 MappingRow 和统计。"""
```

处理顺序：

```text
1. 下载或复用 full CIF
2. 解析 full CIF
3. 读取 Make_Data labels/candidates
4. 遍历 labels 中的 ligand candidate
5. 非 class-4 写 SKIPPED_NON_SMALL_MOLECULE
6. class-4 执行 RCSB instance 匹配
7. 下载或复用 chemcomp JSON
8. 下载或复用 native mol2
9. 执行 validation
10. 构造 MappingRow
```

### #### \[NEW] `Ligand/rcsb_enrichment/run_enrichment.py`

统一 CLI 入口。

CLI 参数：

| 参数                 | 类型           | 默认值                                                        | 含义                         |                     |
| ------------------ | ------------ | ---------------------------------------------------------- | -------------------------- | ------------------- |
| `--raw-json`       | `Path`       | `/home/penghongen/My_Project/Data/raw.json`                | EMDB-PDB 外层过滤              |                     |
| `--parsed-root`    | `Path`       | `/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb` | Make\_Data parsed\_pdb 根目录 |                     |
| `--output-root`    | `Path`       | `/storage/penghongen/CIF_Ligand`                           | 输出根目录                      |                     |
| `--n-jobs`         | `int`        | `16`                                                       | joblib PDB 并行数             |                     |
| `--max-retries`    | `int`        | `3`                                                        | RCSB 下载重试次数                |                     |
| `--retry-sleep`    | `float`      | `2.0`                                                      | 重试 sleep 秒数                |                     |
| `--force-download` | `store_true` | `False`                                                    | 覆盖已有下载                     |                     |
| `--limit`          | \`int        | None\`                                                     | `None`                     | debug 时只处理前 N 个 PDB |
| `--pdb-id`         | \`list\[str] | None\`                                                     | `None`                     | debug 时只处理指定 PDB    |

运行结束必须 print 统计摘要：

```text
raw_json_entries
unique_pdb_ids_in_raw_json
matched_parsed_pdb_count
pdb_with_class4_ligands
class4_candidate_count
PASS_HIGH count
status counts
skipped non-small-molecule count
dockem_input_status counts
emerald_id_input_status counts
pocketxmol_input_status counts
output_root
```

### #### \[NEW] :comment[Ligand/sbatch/rcsb\_ligand\_enrichment\_raw4.sbatch]{#comment-1778761521514 text=" --array 设置成5，并且做调整：__NEWLINE____HASH__SBATCH --cpus-per-task=8"}

新增服务器运行脚本，参考 `sbatch/CPU模板/cpu_split.sbatch`。

建议内容：

```bash
#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/Pocket_Plus/Ligand/logs/rcsb_ligand_enrichment_%j.out
#SBATCH -e /home/penghongen/My_Project/Pocket_Plus/Ligand/logs/rcsb_ligand_enrichment_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J RCSB_Lig
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu
cd /home/penghongen/My_Project/Pocket_Plus

python -m Ligand.rcsb_enrichment.run_enrichment \
  --raw-json /home/penghongen/My_Project/Data/raw.json \
  --parsed-root /home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb \
  --output-root /storage/penghongen/CIF_Ligand \
  --n-jobs 16 \
  --max-retries 3 \
  --retry-sleep 2.0
```

> \[!NOTE]
> 是否使用锁文件机制可在实现阶段沿用 CPU 模板；第一版 sbatch 示例可以先给出直接运行版本，便于批处理和日志验收。

## 输出数据结构说明

### `/storage/penghongen/CIF_Ligand/rcsb_full_cif/{PDB}.cif`

RCSB full structure CIF。

```text
{PDB}.cif  # 原始 RCSB full CIF，用于 _pdbx_nonpoly_scheme 和 _atom_site 解析
```

### `/storage/penghongen/CIF_Ligand/rcsb_chemcomp_cache/{CCD}.json`

RCSB chemical component descriptor 缓存。

关键字段：

```text
rcsb_chem_comp_descriptor.SMILES         # str, RCSB 非手性/常规 SMILES
rcsb_chem_comp_descriptor.SMILES_stereo  # str, RCSB stereo SMILES
rcsb_chem_comp_descriptor.InChI          # str, InChI
rcsb_chem_comp_descriptor.InChIKey       # str, InChIKey
rcsb_chem_comp_info.atom_count_heavy     # int, CCD 层级重原子数
chem_comp.formula                        # str, 化学式
chem_comp.formula_weight                 # float, 分子量
```

### `/storage/penghongen/CIF_Ligand/rcsb_ligand_mol2/{pdb_id}/{candidate_id}_{ccd_id}_{label_asym_id}_{pdb_seq_num}.mol2`

RCSB native ligand mol2。

```text
candidate_id   # Make_Data candidate ID
ccd_id         # ligand CCD ID
label_asym_id  # RCSB _pdbx_nonpoly_scheme.asym_id
pdb_seq_num    # RCSB _pdbx_nonpoly_scheme.pdb_seq_num
```

### `/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv`

一行对应一个 Make\_Data ligand candidate。

字段说明：

```text
emdb_id                                             # str, raw.json 中的 EMDB ID，如 emd_63092
pdb_id                                             # str, 小写 PDB ID，如 9lhb
pdb_id_upper                                       # str, 大写 PDB ID，如 9LHB
candidate_id                                       # int, Make_Data candidate ID
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
full_cif_path                                      # str, 下载后的 full CIF 路径
chemcomp_json_path                                 # str, chemcomp JSON 路径
native_mol2_path                                   # str, native mol2 路径
make_data_heavy_atoms                              # int, Make_Data candidate 重原子数
rcsb_cif_heavy_atoms                               # int, RCSB full CIF 同一 ligand instance 重原子数
mol2_total_atoms                                   # int, native mol2 总原子数
mol2_heavy_atoms                                   # int, native mol2 重原子数
mol2_hydrogen_count                                # int, native mol2 氢原子数
mol2_bond_count                                    # int, native mol2 bond 条目数
mol2_has_charge_field                              # bool, mol2 atom 行是否包含可解析 charge
make_cif_coord_median                              # float, Make_Data vs RCSB CIF 最近邻距离 median，单位 Å
make_cif_coord_max                                 # float, Make_Data vs RCSB CIF 最近邻距离 max，单位 Å
cif_mol2_coord_median                              # float, RCSB CIF vs mol2 最近邻距离 median，单位 Å
cif_mol2_coord_max                                 # float, RCSB CIF vs mol2 最近邻距离 max，单位 Å
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
```

### `/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.jsonl`

与 CSV 一致，但一行一个 JSON object，允许保留嵌套字段。

建议额外字段：

```text
validation_detail                                  # dict, 结构化校验指标
download_attempts                                  # dict, full_cif/chemcomp/mol2 各自尝试次数
rcsb_nonpoly_scheme_row                            # dict, 匹配到的 _pdbx_nonpoly_scheme 原始字段
```

### `/storage/penghongen/CIF_Ligand/mapping/failed_cases.csv`

从 `ligand_mapping.csv` 中筛出：

```text
status != PASS_HIGH
status != SKIPPED_NON_SMALL_MOLECULE
```

字段与 `ligand_mapping.csv` 相同。

### `/storage/penghongen/CIF_Ligand/reports/validation_summary.json`

统计摘要。

字段说明：

```text
raw_json_entries                                   # int, raw.json 原始条目数
unique_pdb_ids_in_raw_json                         # int, raw.json 中唯一 PDB 数
matched_parsed_pdb_count                           # int, raw.json 与 parsed_pdb 求交集后的 PDB 数
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
```

## :comment[Verification Plan]{#comment-1778761778321 text="最后，把本次做的事以目录的形式写入 C:__BSLASH__Users__BSLASH__15919__BSLASH__OneDrive__BSLASH__My_Project__BSLASH__Pocket_Plus__BSLASH__Ligand__BSLASH__notes_of_dataset.md 。里面包含两部分：__NEWLINE__(1). 文件级：服务器上的文件路径与这个文件路径下每个文件的大致说明。__NEWLINE__(2). 字段级：把你写的每个条目的意义与归属都写入这里。"}

### 1. 本地小样本回归

使用本地已验证样例模拟正式入口：

```bash
python -m Ligand.rcsb_enrichment.run_enrichment \
  --raw-json <本地临时 raw.json> \
  --parsed-root "C:\Users\15919\Desktop\服务器上的部分数据\home-penghongen-My_Project-Data-DATA_v2_raw4-parsed_pdb" \
  --output-root "C:\Users\15919\Desktop\测试\CIF_Ligand_formal" \
  --pdb-id 3j7a 5bki 5gmk 5i68 \
  --n-jobs 4
```

关键断言：

```text
3j7a class-4 PASS_HIGH == 1
5bki class-4 PASS_HIGH == 4
5gmk class-4 PASS_HIGH == 1
5i68 class-4 PASS_HIGH == 1
失败数 == 0
```

### 2. 本地全样例回归

使用本地样例目录全量运行：

```bash
python -m Ligand.rcsb_enrichment.run_enrichment \
  --raw-json <本地临时 raw.json> \
  --parsed-root "C:\Users\15919\Desktop\服务器上的部分数据\home-penghongen-My_Project-Data-DATA_v2_raw4-parsed_pdb" \
  --output-root "C:\Users\15919\Desktop\测试\CIF_Ligand_formal" \
  --n-jobs 4
```

预期：

```text
class4_candidate_count == 58
PASS_HIGH == 58
VALIDATION_FAILED == 0
RCSB_NATIVE_MOL2_MISSING == 0
```

### 3. :comment[服务器 dry-run / limit 验证]{#comment-1778761819488 text="这步略去吧"}

服务器上先跑小批量：

```bash
python -m Ligand.rcsb_enrichment.run_enrichment \
  --raw-json /home/penghongen/My_Project/Data/raw.json \
  --parsed-root /home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb \
  --output-root /storage/penghongen/CIF_Ligand \
  --n-jobs 16 \
  --limit 20
```

验收：

```text
程序正常 print summary
mapping/ligand_mapping.csv 存在
mapping/ligand_mapping.jsonl 存在
reports/validation_summary.json 存在
下载目录存在且文件非空
失败记录均有明确 status 和 error_message
```

### 4. sbatch 验证

提交：

```bash
sbatch Ligand/sbatch/rcsb_ligand_enrichment_raw4.sbatch
```

验收：

```text
stdout 最后包含统计摘要
stderr 无 Python traceback
/storage/penghongen/CIF_Ligand/mapping/ligand_mapping.csv 正常生成
/storage/penghongen/CIF_Ligand/reports/validation_summary.json 正常生成
```

### 5. 断点续跑验证

重复运行同一命令，不加 `--force-download`：

```text
应复用已有 full_cif / chemcomp / mol2 文件
应重新生成 mapping 和 summary
不应因为文件已存在报错
```

再使用：

```text
--force-download
```

确认会覆盖下载缓存。

## User Review Required

当前已确认的用户决策：

```text
1. 输出根目录固定为 /storage/penghongen/CIF_Ligand。
2. raw.json 作为外层过滤，只处理 raw.json 中出现且 parsed_pdb 存在的 PDB。
3. 只处理 ligand_class_ids == 4。
4. 一行 mapping 对应一个 Make_Data candidate ligand instance。
5. 同时保存 CSV + JSONL。
6. 允许有限重试；失败仍写入 mapping。
7. 程序结束必须 print 统计摘要。
8. 按 PDB 用 joblib 并行。
9. 默认复用已有下载文件，--force-download 才覆盖。
10. 坐标阈值沿用 median <= 0.05 Å, max <= 0.20 Å。
11. 新增可运行 sbatch 示例。
12. explore.md 先写实验结论，正式代码完成后再追加正式程序说明。
```

如果后续需要把 `ligand_class_ids == 2/3` 纳入处理，应单独制定 peptide/nucleic ligand enrichment 方案，不在本计划第一版中混入。