# unet_c1 pdb-centric-v2 F2-F1 basic 消融完成

## 完成结论

2026-09-20，正式 pdb-centric-v2 模型内部新增的 F2-F1 basic 方法消融已经完成。三模型原始比较仍统一采用 F1-F1；本轮只在该正式模型内部并列比较 F1-F1 与 F2-F1。

正式结果根为 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts/unet_c1`。F2 semantic threshold 为 `0.23468017578125`；F2 basic 参数为 `score_threshold=0.40040671825408936`、`prefiltered_min_voxel=8`、`min_voxels=26`。`test_0` 包含 179 个 PDB；`test_1` 是按清单原顺序从相同逐 PDB事实派生的 149-PDB 子集，不是独立推理或独立统计样本。

## 运行证据

- Job：`378693`，gnode10，A800×1、16 CPU。
- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_49db579cfe2c/Pocket_Plus`。
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/378693/Find_1_job378693_20260920T205610_a29`。
- 正式命令：

```bash
exec bash "${TASK_PROJECT_ROOT}/tmp/unet_c1_pdb_centric_v2_f2_basic_20260920/run_f2_basic.sh"
```

- 正式入口生成 745 个预期新文件。保护复核确认 1,861 个既有文件的大小与纳秒修改时间未变，466 个既有 F1 文件的内容摘要未变。
- 最终门控输出：`status=PASS`，`test_0_rows=179`，`test_1_rows=149`。
- 2026-09-20 22:25 +08:00，只删除 `after_lock_378693`；随后 Job 离开 `squeue`，`try_lock_378693` 与 `after_lock_378693` 均不存在，A800 已安全释放。没有触碰其他任务。

## 主要比较

| 数据集 | 方法 | semantic macro F1 | one-to-one macro F1@0.3 | semantic macro PRAUC |
| --- | --- | ---: | ---: | ---: |
| `test_0` | F1-F1 | 0.394437186 | 0.478132517 | 0.401618043 |
| `test_0` | F2-F1 | 0.399982798 | 0.472451580 | 0.401618043 |
| `test_1` | F1-F1 | 0.404699220 | 0.486865928 | 0.413262000 |
| `test_1` | F2-F1 | 0.411186862 | 0.492032689 | 0.413262000 |

两种方法复用同一 probability，因此 semantic PRAUC 相同是预期结果。完整的语义与三档实例 micro/macro P、R、F1、PRAUC、top-K、固定 occurrence/分辨率/CC 分层和 Spearman 描述性分析见本地 `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\pdb_centric_v2\结果.md`。

## 本地文档与继续边界

- `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\pdb_centric_v2\说明.md`，SHA-256：`b13ef70a7cfed1c192e4339c6bee85339bac70a19a83c19b930379142361ca8e`。
- `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\unet_c1系列\pdb_centric_v2\结果.md`，SHA-256：`a33c85fec80a26762563bd0406519664ba88b56a1365d56156e71c434426d37a`。
- 服务器确定性文档来源 JSON 的 SHA-256：`f0673a7343bcc00572264556aa91faecfa5905fead92d59c3dd90cb28c59a4ad`。
- 两份文档保持 unstaged，未上传服务器，也未提交 Git。用户审阅前不要提交或上传。

完整文件摘要、门控命令、参数和计划差异见 `文档/exec_plan/2026-09-20_unet_c1_pdb-centric-v2_F2-F1-basic消融.md`；共享历史见 `文档/exec_plan/2026-09-02_unet_c1三种采样模型推理与测试.md`。
