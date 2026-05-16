# 实现 RCSB ligand enrichment 自动化管线

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

完成后，用户可以在服务器上用一个 sbatch array 作业，把 `/home/penghongen/My_Project/Data/raw.json` 指定且已存在于 `/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb` 的 PDB 样本批量接入 RCSB。程序会为 Make_Data 中 `ligand_class_ids == 4` 的小分子 ligand 下载 RCSB full CIF、RCSB chemical component descriptor、RCSB 原生 native mol2，建立 `Make_Data candidate_id <-> RCSB ligand instance` 的稳定映射，并输出 CSV/JSONL mapping、失败表、统计报告和数据集说明文档。

## Progress

- [x] (2026-05-14 17:00 CST) 读取用户修改后的 `Ligand/implement_plan.md`、`execplan` 规范、Python 注释/写作技能，并确认 `joblib` 等本地依赖可用。
- [x] (2026-05-14 17:04 CST) 建立本 ExecPlan 文件，记录已确认的实现范围、并行策略和输出格式。
- [x] (2026-05-14 17:20 CST) 实现 `Ligand/rcsb_enrichment/` 模块化程序，包括下载、映射、校验、主入口和 merge 工具。
- [x] (2026-05-14 17:23 CST) 新增 sbatch array 示例和 `Ligand/notes_of_dataset.md` 文件级/字段级说明。
- [x] (2026-05-14 20:51 CST) 运行本地四样本非 array 回归，7 个 class-4 ligand 全部 `PASS_HIGH`。
- [x] (2026-05-14 20:52 CST) 运行本地全样例模拟 array+merge，合并后 58 个 class-4 ligand 全部 `PASS_HIGH`。
- [x] (2026-05-14 20:53 CST) 验证 raw.json 内容为 `[]` 时处理 0 个样本。
- [x] (2026-05-14 20:55 CST) 完成最终编译检查、输出文件检查和 ExecPlan 回顾更新。

## Surprises & Discoveries

- Observation: 本地 `Pocket_Plus_windows` 环境已经包含 `numpy`、`scipy`、`requests`、`Bio`、`joblib`。
  Evidence: 依赖检查命令输出这些模块均为 `OK`。
- Observation: 工作树存在与本任务无关的删除和新增文件。
  Evidence: `git status --short` 显示 `.agents/rules/*.md` 删除、`CLAUDE.md` 等未跟踪文件。本任务只修改 `Ligand/`。
- Observation: Windows PowerShell 写出的临时 raw.json 可能带 UTF-8 BOM。
  Evidence: 空列表 raw.json 验证首次触发 `JSONDecodeError: Unexpected UTF-8 BOM`; 已将 `load_raw_mapping` 改为 `utf-8-sig` 读取。
- Observation: 服务器 conda 环境激活脚本不兼容 sbatch 中的 `set -u`。
  Evidence: `/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/etc/conda/activate.d/activate-binutils_linux-64.sh` 报 `ADDR2LINE: unbound variable`。已将 `Ligand/sbatch/rcsb_ligand_enrichment_raw4.sbatch` 的 `set -euo pipefail` 改为 `set -eo pipefail`。
- Observation: 手动执行 merge 动态脚本时，如果不在仓库根目录运行，Python 找不到 `Ligand` package。
  Evidence: `/tmp/penghongen/slurm_255197/conda_env/bin/python` 报 `ModuleNotFoundError: No module named 'Ligand'`。已新增 `Ligand/sbatch/merge_rcsb_ligand_outputs.sh`，并在文档中明确 merge 前需要 `cd /home/penghongen/My_Project/Pocket_Plus` 或设置 `PYTHONPATH`。
- Observation: 大规模失败表中 `RCSB_INSTANCE_MATCH_FAILED` 的 98.6% 是糖类/支链糖残基。
  Evidence: `failed_cases.csv` 中该状态 44313 行，`NAG/MAN/BMA/FUC/GAL/...` 等 sugar-like CCD 共 43687 行；抽样 `6HUG/NAG/F/1` 位于 `_pdbx_branch_scheme`，不在 `_pdbx_nonpoly_scheme`。已增加 branch scheme 匹配。
- Observation: `BEF/ALF/MOO` 等无机簇 validation failed 的一类原因是 mol2 元素推断不完整。
  Evidence: `6AP1/BEF` mol2 atom type 为 `BE`，旧逻辑误推为 `B`；已将 mol2 元素推断改为完整周期表。
- Observation: merge summary 的 raw/matched 统计不能只从合并 rows 反推。
  Evidence: 用户合并日志中 `raw_json_entries=0` 且 `matched_parsed_pdb_count=11037`，但 array part 日志显示 raw_json_entries 为 11037、missing 为 648。已让 `merge_outputs.py` 读取 part summary 聚合 raw/matched/missing。
- Observation: 剩余 1.4% 非糖类 `RCSB_INSTANCE_MATCH_FAILED` 多数可以通过 `_atom_site` 直接匹配。
  Evidence: Windows 本地复测全部 626 个非糖失败行加 200 个糖类样本，新增逻辑匹配 825/826；其中 559 行走 `ATOM_SITE_AUTH_ASYM_AUTH_SEQ`，266 行走 `BRANCH_PDB_ASYM_PDB_SEQ`。抽样 `6AP1/ACE`、`6VMI/Y5P`、`7FGI/GTA`、`8CEP/KBE`、`9IF4/S0R` 均可下载 RCSB native mol2。

## Decision Log

- Decision: 正式输出根目录固定为 `/storage/penghongen/CIF_Ligand`。
  Rationale: 用户明确要求所有产物统一放在该目录，目录内再分子文件夹。
  Date/Author: 2026-05-14 / User
- Decision: 只处理 `ligand_class_ids == 4`，其它 ligand 只写跳过统计。
  Rationale: 前期 RCSB 单源强验证覆盖的是 small molecule；peptide/nucleic ligand 需要后续单独方案。
  Date/Author: 2026-05-14 / User + Codex
- Decision: array 模式写 part 文件，非 array 模式写总文件，并提供合并程序。
  Rationale: 避免多个 sbatch array task 并发写同一 CSV/JSONL 导致文件损坏；后续读取兼容总表和 parts 表。
  Date/Author: 2026-05-14 / User + Codex
- Decision: array task 共享下载缓存，下载写入使用临时文件再原子替换。
  Rationale: 允许断点续跑和跨任务缓存复用，同时避免并发写坏半成品。
  Date/Author: 2026-05-14 / User
- Decision: `--raw-json` 缺省或空字符串表示不过滤；如果 raw.json 文件内容为 `[]`，则处理 0 个样本。
  Rationale: 区分“用户未要求过滤”和“用户显式给出空过滤集”。
  Date/Author: 2026-05-14 / User + Codex
- Decision: sbatch 不启用 `set -u`。
  Rationale: conda activate.d 脚本可能引用未定义变量，启用 nounset 会让环境激活失败；保留 `set -e` 和 `pipefail` 已能覆盖主要失败场景。
  Date/Author: 2026-05-14 / User + Codex
- Decision: merge 命令必须从仓库根目录运行，或显式设置 `PYTHONPATH`。
  Rationale: 当前 `Ligand` 是仓库内源码包，不是 site-packages 里安装好的 package；`python -m Ligand...` 依赖当前工作目录或 `PYTHONPATH` 能找到仓库根目录。
  Date/Author: 2026-05-15 / User + Codex
- Decision: RCSB instance 匹配同时支持 `_pdbx_nonpoly_scheme` 和 `_pdbx_branch_scheme`。
  Rationale: Make_Data class-4 中包含大量糖类 HETATM 残基，RCSB 将糖链 monomer 组织在 branch scheme 中；不支持 branch 会系统性漏配。
  Date/Author: 2026-05-15 / Codex
- Decision: 在 scheme 表匹配失败后增加 `_atom_site` 兜底匹配。
  Rationale: 一些特殊 HETATM/修饰残基不出现在 scheme 表中，但 `_atom_site` 仍有完整 comp/chain/seq/坐标；优先用 Make_Data 坐标唯一化，避免重复实例中标识符可匹配但坐标属于另一个实例；无唯一坐标命中时再按 auth/label 标识分优先级匹配。
  Date/Author: 2026-05-15 / Codex

## Outcomes & Retrospective

实现已完成。新增 `Ligand/rcsb_enrichment/` 模块化程序、array sbatch 示例和 `Ligand/notes_of_dataset.md`。本地验证覆盖了非 array 四样本、array 分片合并全样例、空 raw.json 三条路径。四样本得到 7/7 `PASS_HIGH`；全样例 array merge 得到 58/58 `PASS_HIGH`；空 raw.json 正确处理 0 个样本。

本次实现保留了前期探索的核心结论，并把它升级成可服务器运行的正式管线。仍未做的事是：不接入 Q-BioLiP，不处理 class 2/3，不运行真实 docking，只做 DockEM/EMERALD-ID/PocketXMol 的输入格式初筛。

## Context and Orientation

现有 Make_Data 数据集说明在 `Make_Data/notes_of_dataset.md`。`parsed_pdb/{pdb_id}` 下有 `labels.npz` 和 `candidates.npz`。`labels.npz` 中 `ligand_candidate_ids` 与 `ligand_class_ids` 同序，`ligand_class_ids == 4` 表示 small molecule。`candidates.npz` 中保存每个 candidate 的 `resnames`、`chain_ids`、`res_ids`、`insertion_codes`、`n_heavy_atoms` 和 `candidate_coords_{candidate_id}`。

前期探索脚本在桌面测试目录中证明，RCSB 单源可通过 full CIF 的 `_pdbx_nonpoly_scheme` 将 Make_Data 的 `resname/chain_id/res_id/insertion_code` 对齐到 RCSB ligand instance。关键是 Make_Data 的 `res_id` 对应 `_pdbx_nonpoly_scheme.pdb_seq_num`；随后用 `_pdbx_nonpoly_scheme.asym_id` 作为 `label_asym_id` 下载 RCSB native mol2。

## Plan of Work

先在 `Ligand/rcsb_enrichment/` 新建 package。`config.py` 保存路径、URL、阈值和状态码。`models.py` 保存 dataclass，避免模块间循环导入。`io_utils.py` 读取 raw.json、解析 Make_Data npz、写 CSV/JSONL/summary，并兼容读取总表或 parts 表。`rcsb_client.py` 用 `requests` 下载 RCSB full CIF、chemcomp JSON 和 native mol2，支持重试、缓存复用、临时文件原子替换。`mmcif_mapping.py` 解析 full CIF 的 `_pdbx_nonpoly_scheme` 和 `_atom_site`。`mol2_parser.py` 轻量解析 TRIPOS mol2。`validation.py` 完成重原子、元素组成、坐标和工具输入合规性检查。`worker.py` 处理单个 PDB。`run_enrichment.py` 作为统一 CLI，用 joblib 按 PDB 并行。`merge_outputs.py` 合并 array parts，生成总表、失败表和 summary。

之后新增 `Ligand/sbatch/rcsb_ligand_enrichment_raw4.sbatch`，使用 `#SBATCH --array=0-4`、`#SBATCH --cpus-per-task=8`，每个 array task 传入 `--array-index`、`--array-count` 和 `--n-jobs 8`。

最后新增 `Ligand/notes_of_dataset.md`，按文件级和字段级说明 `/storage/penghongen/CIF_Ligand` 的目录、CSV/JSONL 字段、summary 字段和 part 文件读取规则。

## Concrete Steps

在仓库根目录 `C:\Users\15919\OneDrive\My_Project\Pocket_Plus` 执行本地验证：

    & 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m Ligand.rcsb_enrichment.run_enrichment --raw-json "" --parsed-root "C:\Users\15919\Desktop\服务器上的部分数据\home-penghongen-My_Project-Data-DATA_v2_raw4-parsed_pdb" --output-root "C:\Users\15919\Desktop\测试\CIF_Ligand_formal" --pdb-id 3j7a 5bki 5gmk 5i68 --n-jobs 4 --force-download

预期输出 summary 中 `PASS_HIGH` 为 7，失败数为 0。

模拟 array 验证：

    & 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m Ligand.rcsb_enrichment.run_enrichment --raw-json "" --parsed-root "C:\Users\15919\Desktop\服务器上的部分数据\home-penghongen-My_Project-Data-DATA_v2_raw4-parsed_pdb" --output-root "C:\Users\15919\Desktop\测试\CIF_Ligand_array" --n-jobs 2 --array-index 0 --array-count 2
    & 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m Ligand.rcsb_enrichment.run_enrichment --raw-json "" --parsed-root "C:\Users\15919\Desktop\服务器上的部分数据\home-penghongen-My_Project-Data-DATA_v2_raw4-parsed_pdb" --output-root "C:\Users\15919\Desktop\测试\CIF_Ligand_array" --n-jobs 2 --array-index 1 --array-count 2
    & 'C:\Users\15919\miniconda\envs\Pocket_Plus_windows\python.exe' -m Ligand.rcsb_enrichment.merge_outputs --output-root "C:\Users\15919\Desktop\测试\CIF_Ligand_array" --array-count 2

预期合并后 `class4_candidate_count` 为 58，`PASS_HIGH` 为 58。

## Validation and Acceptance

验收条件是：非 array 本地四样本运行生成总表并得到 7 个 `PASS_HIGH`；模拟 array 两个 part 文件能被 merge 成总表，并得到 58 个 `PASS_HIGH`；`Ligand/notes_of_dataset.md` 清楚记录文件级和字段级说明；sbatch 文件中 array 数为 5、每 task CPU 为 8、`--n-jobs` 为 8。

## Idempotence and Recovery

下载默认复用非空目标文件。`--force-download` 才覆盖缓存。每次下载先写入 `target.tmp.{pid}`，成功后用 `Path.replace` 原子替换目标文件。array 模式每个 task 写自己的 part 文件，不并发写总表。失败 ligand 仍写入 mapping，带明确 `status` 和 `error_message`，后续可以按失败表重试或审计。

## Artifacts and Notes

最终新增或修改的主要文件位于 `Ligand/`。输出数据位于 `/storage/penghongen/CIF_Ligand` 或本地测试指定的 `--output-root`。

## Interfaces and Dependencies

依赖 Python 包：`numpy`、`scipy`、`requests`、`Bio`、`joblib`。不依赖 RDKit/OpenBabel。RCSB 网络服务包括 full CIF 下载、Data API chemcomp endpoint、ModelServer ligand endpoint。
## 2026-05-15 Post-Run Diagnostics

服务器第二轮完整运行后，`RCSB_INSTANCE_MATCH_FAILED` 从 44313 降到 13，说明 `_pdbx_branch_scheme` 与 `_atom_site` 兜底匹配基本解决了 instance 对齐问题。新的主要失败为 `VALIDATION_FAILED=38718`，其中 38652 条是 `_atom_site` 提取阶段的序号过滤过宽：branch 糖链或重复 CCD 链中，同一个 `label_asym_id` 下多个相同 CCD monomer 被一起提取，导致 RCSB CIF 重原子数远大于 Make_Data/native mol2。已修复为只用 atom row 自身的 `auth_seq_id/label_seq_id` 与目标 ligand 序号集合相交，不再把目标 row 序号塞进每一行 atom 的候选集合。

剩余少量 `BCL/CLF/HE2/FRU` 等失败来自 altLoc 选择过窄：这些 ligand instance 只有 `label_alt_id=B`，旧逻辑只接受空 alt、`A` 或 `1`，因此 RCSB CIF 被提取为 0 个重原子。已修复为每个 `_atom_site` instance 选择单个可用 altLoc：优先空 alt、`A`、`1`，否则接受排序后的第一个可用 altLoc。

本地 Windows 复测代表样本 `6HUG/NAG`、`6HUG/MAN`、`6VMI/Y5P`、`6VMI/P5P`、`7Z6Q/BCL`、`8DBY/CLF`、`8XGG/HE2`、`8UVU/FRU`，修复后 RCSB CIF 重原子数均与 Make_Data/native mol2 对齐，`validate_ligand_pair` 返回 `errors=[]`。

## 2026-05-15 Third Server Run Diagnostics

服务器第三轮完整运行后，`PASS_HIGH=189873 / 190256`，`RCSB_INSTANCE_MATCH_FAILED=0`，剩余失败为 `RCSB_NATIVE_MOL2_MISSING=156` 与 `VALIDATION_FAILED=227`。`RCSB_NATIVE_MOL2_MISSING` 集中在 16 个 PDB，错误来自 RCSB ModelServer 返回 `Could not find source file for 'pdb-bcif/...'`，在“只接受 RCSB native mol2”的规则下应保留为外部源缺失。

`VALIDATION_FAILED` 中 220 条来自 branch 糖链的序号语义问题：`_pdbx_branch_scheme.auth_seq_num` 不能用于 `_atom_site` instance 提取；实际应使用 Make_Data `res_id` / `_pdbx_branch_scheme.pdb_seq_num`，否则会在 glycan chain 中选到相邻 monomer。已修复 `extract_rcsb_atom_site_ligand`，目标序号集合只保留 `ligand.res_id` 与 `match.row.pdb_seq_num`。本地复测 `8AA3/FRU`、`8G3R/MAN`、`8G6U/MAN`、`6HUG/NAG`、`8DBY/CLF`、`7Z6Q/BCL`，RCSB CIF/native mol2 均重新对齐，`validate_ligand_pair` 返回 `errors=[]`。

预计修复后剩余真正 validation hard cases 为 7 条：`6JLU/CLA` 的 Make_Data 只保留 5 个重原子而 RCSB/native mol2 为 46 个重原子；`7V68/IXO`、`7V68/2CU`、`9O7S/1KP` 四条链的重原子数一致但 Make_Data 与 RCSB CIF 坐标偏差超过当前阈值。

## 2026-05-15 Fourth Server Run Diagnostics

服务器第四轮完整运行后，`PASS_HIGH=190087 / 190256`，`RCSB_INSTANCE_MATCH_FAILED=0`，`VALIDATION_FAILED=13`，`RCSB_NATIVE_MOL2_MISSING=156`。失败表显示 `RCSB_NATIVE_MOL2_MISSING` 行的工具 readiness 字段为空，导致 summary 中 `NOT_READY_NO_VALID_NATIVE_MOL2_PAIR=13` 只统计到 validation failed，漏掉 156 个 native mol2 缺失条目。已修复 `worker.py`：native mol2 下载失败或返回空 mol2 时，同步填充 DockEM/EMERALD-ID/PocketXMol 的 NOT_READY 状态。

13 个 `VALIDATION_FAILED` 中，6 个来自 `9L5S` 的 `_atom_site` 兜底匹配，ID 精确匹配到 `auth_asym=8` 但坐标偏差 37-53 Å，说明重复实例中标识符匹配不足以唯一化。已调整 `match_atom_site_ligand`：先用 Make_Data 坐标唯一化匹配，只有坐标没有唯一命中时才退回 auth/label 精确匹配。预计这 6 条可被救回。

剩余 7 条更像真实 hard cases：`6JLU/CLA` 为 Make_Data 局部切片重原子数 5 vs RCSB/native mol2 46；`7V68/IXO`、`7V68/2CU`、`9O7S/1KP` 四条链为重原子数一致但坐标偏差超过当前 PASS_HIGH 阈值。

## 2026-05-16 Fifth Server Run Diagnostics

第五轮失败表 `Ligand/logs/failed_cases_5.csv` 显示 `RCSB_NATIVE_MOL2_MISSING=156` 与 `VALIDATION_FAILED=13`，并且工具 readiness 已正确把 169 条失败计入 `NOT_READY_NO_VALID_NATIVE_MOL2_PAIR` / `NOT_READY_NO_VALID_SMILES_OR_STRUCTURE`。这说明统计字段修复生效。

13 条 validation failed 中仍有 6 条 `9L5S/P5P/Y5P`，其 `match_method=ATOM_SITE_COORD`，但校验距离仍为 37-53 Å。诊断为二次提取不一致：匹配阶段用 Make_Data 坐标唯一化找到了正确 `_atom_site` group，但 `extract_rcsb_atom_site_ligand` 又用合成 row 的 label_asym/seq 重新提取，可能在重复实例中漂回另一个 group。已修复为：当 `match_method == ATOM_SITE_COORD` 时，extract 阶段也直接用 Make_Data 坐标唯一化返回同一个 `_atom_site` group。

如果该修复生效，预计 `9L5S` 的 6 条会消失，最后稳定残留为 7 条 hard cases 与 156 条 RCSB native mol2 source 缺失。

## 2026-05-16 Final Server Run Diagnostics

第六轮失败表 `Ligand/logs/failed_cases_6.csv` 与 merge summary 显示最终状态：

```text
class4_candidate_count = 190256
PASS_HIGH = 190093
RCSB_NATIVE_MOL2_MISSING = 156
VALIDATION_FAILED = 7
RCSB_INSTANCE_MATCH_FAILED = 0
```

DockEM/EMERALD-ID/PocketXMol readiness 统计均为 `190093` 条可用、`163` 条不可用，正好等于 `156` 条 native mol2 缺失加 `7` 条 validation hard cases。说明最终工具输入状态统计与状态表一致。

最终 7 条 validation hard cases 为：`6JLU/CLA` 一条 Make_Data 重原子数 5 vs RCSB/native mol2 46；`7V68/IXO`、`7V68/2CU`、`9O7S/1KP` 四条链为重原子数一致但坐标超过严格阈值。`9L5S` 的 6 条 `ATOM_SITE_COORD` 二次提取漂移已消失，说明坐标兜底提取修复生效。

结论：RCSB 单源短期 enrichment 路线达成验收。若把 RCSB ModelServer native mol2 缺失视为外部源不可用，在 RCSB native mol2 可获得条目中，`PASS_HIGH = 190093 / (190256 - 156) = 99.996%`。即使把 native mol2 缺失也计入总分母，`PASS_HIGH = 190093 / 190256 = 99.914%`。
