# Emap2lig 官方 Find held-out 评估执行记录

本文记录 Emap2lig v0.3.4 官方 Find 在 Pocket Plus `held_out_06_chain` 测试集上的一次完整前向、标准 Stage1 指标评估、`test_1` 保序派生和结果文档收口。任务不执行参数校准，也不运行 Emap2lig-Build。

## 当前状态

- 2026-09-16：实现开始。Job `383986` 已在 `gnode09` 预留 A800×1、4 CPU，并稳定停在 `pre_lock_383986`；`after_lock_383986` 存在。正式代码和门控通过前不释放 `pre_lock`。
- Pocket Plus 的任务边界提交为 `13148ff`，Emap2lig 的任务边界提交为 `d2af4d0`。后续实现分别位于 `codex/emap2lig-held-out-evaluation` 分支。
- Job `383986` 第 1 次执行完成单 PDB 隔离门控。`9ter` 的官方 Find 共保存 279 个不少于 32 体素的实例；超过 100 个实例后仅产生告警，没有退出。随后同一批产物完成完整概率线性映射、279 个实例的保身份稀疏映射、逐实例交集与匹配事实计算和标准指标聚合。门控根为 `/storage/penghongen/tmp/emap2lig_official_find_li_gate_20260916/allocation_runner_job383986_20260916T092204_a1`。Job 已停回 `try_lock_383986`，`after_lock_383986` 保留。
- 09:40 关键事件：Job `383986` 第 2 次执行启动正式任务。AdaLigand allocation launch 为 `/home/penghongen/Feedback/AdaLigand/launches/383986/allocation_runner_job383986_20260916T094030_a2`；Pocket Plus release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_5570cf8a0b7a/Pocket_Plus`，Emap2lig release 为 `/home/penghongen/Feedback/Emap2lig/releases/Emap2lig_60e34f30b989/Emap2lig`。首个 `9ter` 已正常进入 512 个 ROI 的 GPU 前向，尚无异常；`after_lock_383986` 保留。

## 冻结范围

- Emap2lig 版本：v0.3.4。
- 模型：官方 `emap2lig-find-v0.0.1.safetensors`。
- 随机种子：42；官方 detection batch size：16。
- 唯一预测行为修改：保留超过 100 个官方 blob 后不再调用 `sys.exit(1)`；其余官方预处理、Li 阈值、连通域、少于 32 体素过滤和输出行为不变。
- `test_0`：179 个 PDB，执行一次完整官方 Find。
- `test_1`：149 个 PDB，是 `test_0` 的保序子集；只从父测试集逐 PDB 评估事实派生，不重复模型前向。

## 正式产物

```text
/storage/penghongen/AdaLigand_stage1_inference/Emap2lig/official_find_li/
├── held_out_test_0/
└── held_out_test_1/
```

旧实验根 `/home/penghongen/My_Project/EVAL_OUT/emap2lig_find_official` 只读保留，不复用或覆盖。

## 正式运行命令

Job `383986` 的正式命令冻结为：

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/infer/emap2lig_official_find_li.sh"
```

动态命令在执行该行前把 `TASK_PROJECT_ROOT` 固定为上述 Pocket Plus release，并把 `EMAP2LIG_PROJECT_ROOT` 固定为上述 Emap2lig release。核心文件 SHA-256：

- Pocket Plus 适配器：`dfc6a6ae903b9a8f919a341e3f910ed696ccdd84e803c265bc64b2e97bcd61e6`
- Pocket Plus 正式 shell：`5bfafe963ed51e4dde110fdae28713105883b82bbcd26a7d7d54a2d56fb28eff`
- Emap2lig `main.py`：`a65cc21d35603174e4cc22a51de4eb8713455674192558b02f60435cb61d44b6`
- Emap2lig held-out runner：`7f95e3cd0f78cf2a0652ac42034c652674966ca9e6008c5abecf352dff3ecd28`

本节不混入单元测试、只读门控或临时核查命令。产物哈希在最终验收后补充。

## 门控与测试

- Pocket Plus 服务器定向测试：`1 passed`。
- Emap2lig Python 3.10 环境不含 pytest；同一环境直接执行 AST 与清单解析门控，结果为 `EMAP_GATE_OK`。
- Job `383986` 第 1 次执行是单 PDB 隔离门控，不是正式运行命令。该门控完整覆盖 `9ter` 官方 Find、`>100 blob` 行为、网格映射、实例事实保存和现有 Stage1 聚合函数。

## 计划与实现差异

- 资源差异：原计划沿用 Job `368455` 的 H100×2、64 CPU；用户改为 Job `383986` 的 A800×1、4 CPU。官方 Find 仍仅使用 GPU 0；CPU 评估并发上限相应从 8 调整为 4，不增加双卡分片。
