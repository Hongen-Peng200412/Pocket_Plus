# Handoff: occurrence-centric held-out 测试完成

Date: 2026-09-03

## Current State

Job `367411` 的第 2 次执行已经成功结束正式命令，A100 资源随后由用户释放。Job `367332` 仍在 A800 上执行 pdb-centric-v1 held-out probability；2026-09-03 11:36 为 88/179。

## Completed

- occurrence-centric 使用既有 `F1_semantic.json` 与 `F1_basic.json`，完成 `/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json` 的固定评估。
- held-out 清单、179 个 probability、179 个 `F1_blobs`、179 个逐 PDB evaluation NPZ 与 179 行 JSONL 的 PDB 集合完全相同。
- 汇总指标文件为 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950/artifacts/unet_c1/held_out_test_0/evaluation/f1_blobs_basic_macro_selected.metrics.json`，SHA-256 为 `ecc51a10e1e9aa91bbbc9102c3d9eedf5b80245c21ca62a623de9778fa2f77dc`。
- 逐 PDB 文件 `f1_blobs_basic_macro_selected.jsonl` 的 SHA-256 为 `9304f9f59a3e9240229afb4c8f56455a1735e3daa27c17bcd83697ebe77b8de3`；全部汇总浮点值均为有限值。
- semantic micro/macro F1 为 `0.5483965335871361/0.3810521577659803`，micro/macro PRAUC 为 `0.4979719127280558/0.3995855713715538`。
- coverage 阈值 0.3 的 micro/macro F1 为 `0.5607615110425321/0.4605255323274474`，micro/macro PRAUC 为 `0.41850753522245415/0.45608615319770157`。
- one-to-one 阈值 0.3 的 micro/macro F1 为 `0.5481986368062317/0.4551295232491723`，micro/macro PRAUC 为 `0.40385531605555874/0.4495114476872045`。0.5 与 0.6 阈值的完整指标已写入执行日志和汇总 JSON。

## Decisions

- occurrence-centric calibration 产物没有重算或改写；本次只新增 held-out 目录和评估汇总。
- occurrence 不再重跑；A100 资源已经释放。
- 三模型最终比较继续采用 F1 blobs+basic；Gaussian 与 centered 不参与。

## Open Questions

- pdb-centric-v1 held-out 测试尚未完成。
- pdb-centric-v2 训练 Job `358384` 尚未自然结束，最终 checkpoint 仍未冻结。

## Next Actions

1. 继续守护 Job `367332` 到 pdb-centric-v1 的 179 个 probability、179 个 F1 blobs 和最终评估完全闭合。
2. Job `358384` 正常结束后核验最终 checkpoint，再把 Job `367332` 的正式命令改为 `pdb_centric_2` 并按锁纪律启动。
3. 三模型结果齐全后，对相同 held-out 清单汇总 semantic、coverage、one-to-one 的 F1 与 PRAUC。

## Files To Reopen

- `文档/exec_plan/2026-09-02_unet_c1三种采样模型推理与测试.md`
- `训练与运行/sh/infer/unet_c1_sampling_comparison.sh`
- `CLAUDE/memory/handoffs/2026-09-03-unet-c1-pdb-centric-v1-calibrated.md`
