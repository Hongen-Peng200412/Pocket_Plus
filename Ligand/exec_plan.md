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
