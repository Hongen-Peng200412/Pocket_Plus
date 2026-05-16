# Ligand Enrichment 文档入口

本目录保存 Make_Data class-4 small molecule ligand 的 RCSB 单源 enrichment 程序、运行脚本、日志和说明文档。

## 推荐阅读顺序

1. `Ligand/notes_of_dataset.md`
   最终数据集说明与字段索引。后续写数据读取、质检、训练或 docking 准备脚本时，优先查这个文件；其中包含 `match_method`、`status`、工具输入状态和 `validation_summary.json` 的详细字段字典。

2. `Ligand/doc/explore.md`
   探索过程说明。解释为什么短期选择 RCSB 单源，而不是直接从本地 CIF 恢复 SMILES/mol2，或立即切换到 Q-BioLiP。

3. `Ligand/doc/implement_plan.md`
   正式实现计划。记录程序模块、CLI、并行策略、输出格式和 sbatch 设计。

4. `Ligand/doc/exec_plan.md`
   实际执行记录。记录服务器多轮运行、失败诊断、bug 修复、最终验收结果，以及执行过程中关键枚举/统计字段的审计口径。

## 最终结论

RCSB 单源短期 enrichment 路线已验收通过。

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

## 程序入口

批量 enrichment：

```bash
python -m Ligand.rcsb_enrichment.run_enrichment
```

array part 合并：

```bash
python -m Ligand.rcsb_enrichment.merge_outputs \
  --output-root /storage/penghongen/CIF_Ligand \
  --array-count 5
```

推荐服务器提交脚本：

```text
Ligand/sbatch/rcsb_ligand_enrichment_raw4.sbatch
Ligand/sbatch/merge_rcsb_ligand_outputs.sh
```

## 正式输出根目录

```text
/storage/penghongen/CIF_Ligand
```

主要产物：

```text
mapping/ligand_mapping.csv
mapping/ligand_mapping.jsonl
mapping/failed_cases.csv
reports/validation_summary.json
rcsb_full_cif/
rcsb_chemcomp_cache/
rcsb_ligand_mol2/
```

## 旧文档归档

旧版 `notes_of_dataset.md` 已归档：

```text
Ligand/doc/archive/notes_of_dataset_legacy_2026-05-16.md
```
