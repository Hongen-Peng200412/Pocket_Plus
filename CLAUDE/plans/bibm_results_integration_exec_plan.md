# 整合 BIBM 多模型评估结果

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

## Purpose / Big Picture

本任务要把服务器上零散的 Pocket Plus、Phenix 和 Emap2lig 逐样本评估产物整合成一个可审计的 Markdown 结果总表。完成后，用户可以打开 `D:\OneDrive\结果.md`，直接看到蛋白测试集、核酸测试集、补充细则、以及“限制在 Emap2lig 成功样本上”的公平对比表，并能看到 Emap2lig 失败样本清单。

## Progress

- [x] (2026-06-16 00:00Z) 通过 grill-me 收敛最终口径：展示新名 `unet_raw/unet_diff/unet_full/emb_unet`，读取旧名远端目录 `unet_c1/unet_c2/unet_base/emb_unet`。
- [x] (2026-06-16 00:00Z) 确认主表指标列：`Dice`、`PR-AUC`、60% 与 30% 的 coverage F1、one-to-one F1、top3/top4/top5。
- [x] (2026-06-16 00:00Z) 确认 Emap2lig 行语义：`Emap2lig 原始结果` 为全量 110/40，失败样本按 empty prediction metrics 纳入；`Emap2lig 去除失败样本后` 为 `status == "ok"` 子集。
- [x] (2026-06-16 00:00Z) 只读探针确认远端 `/home/penghongen/My_Project/EVAL_OUT/infer_out` 与 `/home/penghongen/My_Project/EVAL_OUT/emap2lig_find_official` 存在。
- [x] (2026-06-16 00:00Z) 发现 Phenix 当前只有 protein_110 与 protein_40 产物，没有 nucleic_40 评估产物。
- [x] (2026-06-16 09:37Z) 补齐 Phenix nucleic_40 结果；`/home/penghongen/run_cmd_297517.sh` 最终完成了 nucleic_40 的 diff-map 生成与 fixed-test 评估，stardard/strict 两套 `best_summary.json` 均已落盘。
- [x] (2026-06-16 04:00Z) 已把 `src/inference/baseline/generate_phenix_diff_maps.py` 改为 joblib 默认 8 worker 并行；本地与服务器环境均通过 `python -m py_compile`，远端文件已覆盖到 `/home/penghongen/My_Project/Pocket_Plus/src/inference/baseline/generate_phenix_diff_maps.py`。
- [x] (2026-06-16 09:37Z) 297517 的目标产物已经完整落盘；虽然 `squeue` 在一段时间内仍显示 `R`，但 `try_lock_297517` 已恢复，日志也已打印 `[run_baseline_two_stage] all baselines done.` 与 `[run_cmd_297517] done`。
- [x] (2026-06-16 05:38Z) 并行版 Phenix diff-map 生成阶段完成：nucleic_40 的 stardard/strict 40 个样本均生成齐全，`generate_summary.json` 记录本轮生成 74、跳过 306、失败 0。
- [x] (2026-06-16 09:37Z) baseline cache 与 fixed-test 最终自然完成；`phenix_real_space_diff_map_stardard_nucleic_40` 与 `phenix_real_space_diff_map_strict_nucleic_40` 的 cache 都补齐到 40/40，核酸 fixed-test 摘要已生成。
- [x] (2026-06-16 00:00Z) 编写可复跑脚本：`scripts/create_bibm_results_snapshot.py` 负责创建快照，`scripts/collect_bibm_results.py` 负责生成 `D:\OneDrive\结果.md` 与审计 JSON；两者均已通过 `python -m py_compile`。
- [x] (2026-06-16 00:00Z) 从服务器抓取最新逐样本结果快照到 `tmp/bibm_results_snapshot.json`；使用远端写 JSON 后 base64 传回，避免中文转码损坏。
- [x] (2026-06-16 00:00Z) 运行汇总脚本并验证表格、失败样本清单、样本计数和数值格式；`D:\OneDrive\结果.md` 已生成，核酸 Phenix 目前以 `NA` 和缺失样本数暴露。
- [x] (2026-06-16 10:54Z) 基于补齐后的 Phenix 核酸结果重新抓取快照、重算审计 JSON，并刷新 `D:\OneDrive\结果.md`；主表与对比表中的核酸 `phenix(standard/strict)` 行已从 `NA` 更新为真实指标。
- [x] (2026-06-16 10:54Z) 处理 Windows 控制台对 `D:\OneDrive\结果.md` 中文路径的转码问题：先写到 `tmp/bibm_results_output.md`，再由 PowerShell 复制到最终路径，确保最终文件内容与新快照一致。

## Surprises & Discoveries

- Observation: 旧的 `collect_eval_results.py` 对 Phenix 核酸结果路径使用了 `phenix_real_space_diff_map_{system}_40/stage2_threshold_component_policy`，但远端交叉样本名后确认该目录是 `protein_40` calibration 集，不是 `nucleic_40`。
  Evidence: 远端只读脚本显示该目录 40 个样本与 `protein_40.json` 重合 40 个，与 `nucleic_40.json` 重合 0 个。

- Observation: DL 固定测试输出根目录中同时存在 `best_summary.json` 和 `per_sample_best_metrics.json`，其中 `best_summary.json` 已含 `per_sample`，但 `per_sample_best_metrics.json` 保留显式 `sample_name`，更适合按 Emap2lig 成功样本重聚合。
  Evidence: `/home/penghongen/My_Project/EVAL_OUT/infer_out/emb_unet_stardard/per_sample_best_metrics.json` 长度为 110，单项含 `sample_name`、`metrics`、`error`。

- Observation: SSH/PowerShell 文本通道会破坏远端 JSON 中的中文字段。
  Evidence: 直接把 stdout 写入本地时，`核酸测试集` 字段附近出现未闭合字符串和控制字符；改为远端写 UTF-8 JSON 文件再用 `base64 -w 0` 传回后，`collect_bibm_results.py` 可正常读取。

- Observation: 297517 当前仍在执行前一次串行 Phenix nucleic_40 补跑，日志中能看到新的 `[ok]` diff map 生成记录，但 `try_lock_297517` 还没有恢复。
  Evidence: `squeue -j 297517` 显示作业仍为 RUNNING，`/home/penghongen/try_lock_297517` 不存在，stdout tail 包含 `7pbp`、`8iaz`、`7v2m` 的 stardard/strict `[ok]` 记录。

- Observation: Phenix diff-map 生成阶段可用 joblib 并行完成，但后续 baseline cache 使用 `loky` 后端时出现 worker-stopped warning，并在 stardard nucleic_40 cache 38/40 处长时间无进展。
  Evidence: 最新 cache 目录 `/home/penghongen/My_Project/EVAL_OUT/phenix_/infer_cache/baseline/phenix_real_space_diff_map_stardard_nucleic_40` 只有 38 个 `.npz`，缺 `8p03` 与 `8j1z`；日志停在 `[build_baseline_cache] ... cache_backend=loky` 后的 joblib warning。

- Observation: baseline cache 虽然在 38/40 阶段长时间停顿，但最终没有真正卡死；两个尾部样本只是异常慢，继续等待后自然补齐到 40/40。
  Evidence: 最终 `phenix_real_space_diff_map_stardard_nucleic_40` 与 `phenix_real_space_diff_map_strict_nucleic_40` 目录下都存在 40 个 `.npz`，并且两套 `best_summary.json` 都已生成。

- Observation: Windows 控制台会把传给 Python 的中文输出路径 `D:\OneDrive\结果.md` 转码成错误字符，导致汇总脚本把 Markdown 写到错误文件名。
  Evidence: 首次直接运行 `collect_bibm_results.py --output-md D:\OneDrive\结果.md` 时，脚本输出的落盘路径显示为乱码；改为先写 `tmp/bibm_results_output.md` 再用 PowerShell `Copy-Item` 到目标路径后，`D:\OneDrive\结果.md` 内容与最新快照一致。

## Decision Log

- Decision: 最终表格展示新模型名，条目说明保留旧目录名。
  Rationale: 用户确认服务器仍使用旧名，但论文/结果总表应使用 `unet_raw/unet_diff/unet_full/emb_unet`。
  Date/Author: 2026-06-16 / Codex

- Decision: Emap2lig 对比专用表只放 `Emap2lig 去除失败样本后`，不重复 `Emap2lig 原始结果`。
  Rationale: 对比专用表定义为所有模型都限制在 Emap2lig 成功样本上。
  Date/Author: 2026-06-16 / Codex

- Decision: 遇到模型缺失某些 Emap2lig 成功样本时，不把缺失样本当 0，而是在补充表中记录缺失期望样本数。
  Rationale: 用户确认主表按已有样本聚合，缺失通过样本计数和说明暴露。
  Date/Author: 2026-06-16 / Codex

- Decision: Phenix diff map 生成脚本默认使用 joblib `loky` backend 和 8 worker，但保留 `--n_jobs` 与 `--backend` CLI 参数。
  Rationale: 用户确认 8 线程并行已实测可用；保留 CLI 参数便于后续在资源紧张或调试时降并行度。
  Date/Author: 2026-06-16 / Codex

- Decision: Windows 侧最终结果文件继续落到 `D:\OneDrive\结果.md`，但汇总脚本调用时先写入 ASCII 临时路径，再由 PowerShell 复制到中文目标路径。
  Rationale: 这能绕过 PowerShell/控制台到 Python 的中文参数转码问题，不需要改动汇总脚本本身的数据逻辑。
  Date/Author: 2026-06-16 / Codex

## Outcomes & Retrospective

任务已经完成：`D:\OneDrive\结果.md` 现在包含 8 个结果/补充表、2 个 Emap2lig 失败样本清单和条目说明。蛋白与核酸两个数据集都已经填入 `unet_raw / unet_diff / unet_full / emb_unet / phenix / Emap2lig` 的完整结果，其中核酸 `phenix(standard)` 与 `phenix(strict)` 已不再是 `NA`。

最终审计结果显示，DL 条目在全量口径和 Emap2lig 成功样本子集口径下都没有缺失期望样本。Emap2lig 蛋白为 57 成功 / 53 失败，核酸为 17 成功 / 23 失败。Phenix 核酸 fixed-test 的关键结果已经纳入总表：standard 为 `Dice 0.0599 / PR-AUC 0.0378`，strict 为 `Dice 0.0398 / PR-AUC 0.0206`。

这次工作的主要教训有两个：一是 Phenix nucleic_40 的真实结果必须来自 extra fixed-test 输出目录，不能误用 protein_40 calibration 目录；二是 Windows 终端到 Python 的中文路径参数并不可靠，最终交付文件应用 ASCII 临时路径加 PowerShell 复制的方式落盘。

## Context and Orientation

本地仓库为 `D:\OneDrive\My_Project\Pocket_Plus`。主要远端结果目录为 `/home/penghongen/My_Project/EVAL_OUT`。DL 模型结果在 `/home/penghongen/My_Project/EVAL_OUT/infer_out`，Emap2lig 结果在 `/home/penghongen/My_Project/EVAL_OUT/emap2lig_find_official`，Phenix baseline 结果在 `/home/penghongen/My_Project/EVAL_OUT/phenix_/infer_out/baseline`。

“standard” 在配置和目录中拼作 `stardard`，最终 Markdown 展示写 `standard`。“strict” 保持不变。蛋白测试集为 `protein_110.json`，核酸测试集为 `nucleic_40.json`。

## Plan of Work

先用远端只读脚本收集所有逐样本 JSON、样本元信息和来源路径，写成本地快照 JSON。然后用 `scripts/collect_bibm_results.py` 读取快照，重算全量与 Emap2lig 成功子集指标，生成 Markdown 总表与审计 JSON。

如果 Phenix nucleic_40 缺失，则通过 `/home/penghongen/try_lock_297517` 运行 `generate_phenix_diff_maps.py` 与 `run_baseline_two_stage.py`，只补 `phenix_real_space_diff_map` 的 `nucleic_40` extra fixed-test，不重新跑已有 DL 或 Emap2lig。产物齐全后，再抓一次最新快照并刷新最终 Markdown。

在 Windows 本地落盘最终 Markdown 时，若目标路径包含中文文件名，则先输出到 ASCII 临时路径，再由 PowerShell 复制到目标路径，避免控制台参数转码破坏文件名。

## Concrete Steps

1. 远端只读确认目录结构和关键 JSON 可读。

       PowerShell from D:\OneDrive\My_Project\Pocket_Plus:
       $env:CODEX_SSH_PASSWORD = (Get-Content $env:USERPROFILE\.ssh\pocket_plus_sshpass.txt -Raw).Trim()
       & .\与服务器交互\other\Invoke-PasswordSsh.ps1 -HostName 10.102.33.220 -Port 10022 -UserName penghongen -Command "hostname"

2. 如需补 Phenix nucleic_40，写入 `/home/penghongen/run_cmd_297517.sh`，删除 `/home/penghongen/try_lock_297517` 触发作业，并监控日志。

3. 抓取远端结果快照到 `tmp/bibm_results_snapshot.json`。

4. 运行本地汇总脚本；若 `D:\OneDrive\结果.md` 直接作为命令行参数出现转码问题，则先写到 ASCII 临时路径，再复制到最终路径：

       python scripts/collect_bibm_results.py --snapshot-json tmp/bibm_results_snapshot.json --output-md tmp/bibm_results_output.md --audit-json tmp/bibm_results_audit.json
       Copy-Item -LiteralPath tmp\bibm_results_output.md -Destination D:\OneDrive\结果.md -Force

## Validation and Acceptance

`D:\OneDrive\结果.md` 必须包含 8 个结果表：前 4 个为全量口径，后 4 个为 Emap2lig 成功样本子集口径。每个主结果表必须包含 12 个主指标列；每个补充细则表必须包含 voxel precision/recall/F1/IoU、候选数、coverage/one-to-one precision/recall 和样本计数。

失败样本清单必须按蛋白和核酸分开，列出 `sample_name`、`PDB ID`、`EMDB ID`、`status`、`error`。所有小数比例保留 4 位，平均候选数保留 2 位，缺失写 `NA`。本次完成态还要求核酸 `phenix(standard/strict)` 在主表和补充表中都为真实数值，而不是 `NA`。

## Idempotence and Recovery

汇总脚本是幂等的，重复运行会覆盖审计 JSON，并覆盖临时 Markdown。远端 Phenix 补跑使用现有脚本的 skip-existing 行为，已有 protein_40/protein_110 产物不会重算。不要触碰 `kill_lock_297517`，除非用户明确授权终止任务；不要删除 `after_lock_297517`。

如果最终目标路径是 `D:\OneDrive\结果.md`，优先采用 ASCII 临时文件加 `Copy-Item` 的方式落盘；这一步也是幂等的。

## Artifacts and Notes

远端最终完成的核酸 Phenix 结果路径为：

- `/home/penghongen/My_Project/EVAL_OUT/phenix_/infer_out/baseline/phenix_real_space_diff_map_stardard_nucleic_40/best_summary.json`
- `/home/penghongen/My_Project/EVAL_OUT/phenix_/infer_out/baseline/phenix_real_space_diff_map_strict_nucleic_40/best_summary.json`

DL 逐样本结果示例路径为 `/home/penghongen/My_Project/EVAL_OUT/infer_out/emb_unet_stardard/6z1r/summary.json`。

## Interfaces and Dependencies

新增脚本只使用 Python 标准库：`argparse`、`json`、`datetime`、`pathlib`、`statistics`、`typing`。输入快照 JSON 中每个模型条目需要包含 `rows`，每行至少有 `sample_name` 和 `metrics`；Emap2lig 行还包含 `status` 和 `error`。

Revision note 2026-06-16: 初版计划根据 grill-me 口径、本地源码和远端只读探针创建。
Revision note 2026-06-16: 根据 Phenix nucleic_40 补跑完成情况、最新快照刷新结果，以及 Windows 中文路径落盘绕行方案，将计划更新为完成态。
