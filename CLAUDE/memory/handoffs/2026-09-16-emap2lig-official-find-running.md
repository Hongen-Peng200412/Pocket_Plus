# Handoff: Emap2lig 官方 Find held-out 评估运行中

Date: 2026-09-16

## Current State

Job `383986` 正在 `gnode09` 使用 A800×1、4 CPU 和旧 Emap2lig release 执行 179-PDB `test_0`。截至 12:29，25 个 PDB 已写出 `status=ok`，`30yu` 已完整保存 1,878 个实例，日志正在处理 `9nnc`。`after_lock_383986` 与原动态命令保留，`kill_lock_383986` 不存在。结果等价的局部裁剪优化已经完成门控并冻结为新 release，但未经用户明确授权不得停止当前动态命令或切换 release。

## Completed

- Emap2lig 实现提交为 `6d793a5`；唯一预测行为修改是超过 100 个 blob 后不再退出。
- Emap2lig 上一轮历史已按双线规则重建；本轮局部裁剪实现端点为 `aff2c5f`，学习端点和 `Learn/CUMULATIVE` 为 `4e20f19`，两个端点的 tree 哈希同为 `96258424077f788639488cc711c7eca9ff38c6cd`。
- 新实现不再为每个实例建立并扫描完整体积掩码，而是使用 `prop.bbox` 与局部 `prop.image` 复现原 `crop_mrcs()` 的 48³、对称补齐、边界截断、尾部补零和原点计算。
- 完整 `9ter` 门控核对 776 个原始连通域和 279 个保留实例；497 个小连通域、4 个长实例、2 个边界实例及超过 100 个实例的场景均被实际覆盖。NPZ 全字段与 MRC 科学字段逐项严格一致，实例后处理由 70.714508 秒降至 8.065864 秒，提升 8.767134 倍。
- 门控根为 `/storage/penghongen/tmp/emap2lig_local_blob_crop_gate_20260916`；`comparison.json` SHA-256 为 `cc751494082af56db42c52b4eecfc815ea4775930eb72de044d56af694907317`。
- 新 Emap2lig release 为 `/home/penghongen/Feedback/Emap2lig/releases/Emap2lig_eb7a32a0ee70/Emap2lig`，内容 SHA-256 为 `eb7a32a0ee702eff52e85734c7279b4bc7ad6c05db2a57192573e3ccaad5a5b8`；它尚未进入 Job `383986` 的动态命令。
- Pocket Plus 真实实现端点 `ac028bc` 与 Learn 端点 `3966fde` 整树等价；`Learn/CUMULATIVE` 已推进到 `3966fde`。
- 服务器定向测试通过。Job `383986` 第 1 次隔离门控完整保留 `9ter` 的 279 个官方 blob，并成功完成概率映射、实例映射和标准指标聚合。
- 正式 Pocket Plus release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_5570cf8a0b7a/Pocket_Plus`。
- 正式 Emap2lig release 为 `/home/penghongen/Feedback/Emap2lig/releases/Emap2lig_60e34f30b989/Emap2lig`。
- 正式 launch 为 `/home/penghongen/Feedback/AdaLigand/launches/383986/allocation_runner_job383986_20260916T094030_a2`。

## Decisions

- `test_0` 只执行一次官方 Find；`test_1` 从 `test_0` 的逐 PDB 评估事实保序派生。
- 每个官方不少于 32 体素的 blob 均为 `selected=True`、`prauc_eligible=True`；分数为原生 ligand probability 的实例内均值。
- 概率采用线性映射，实例采用最近体素映射并保留官方编号，允许不同实例映射后重叠。
- 正式评估使用 4 个 CPU worker；不增加双卡分片。
- 串行局部裁剪对 279 个实例只需约 8 秒，已经不再明显不足；因此不为了把 8.767 倍继续推到 10 倍以上而增加进程池和并行写盘复杂度。

## Next Actions

1. 等待用户明确授权使用 `kill_lock_383986`。授权前不改动态命令、不创建或删除任何锁，也不用 `scancel`。
2. 授权后先只读确定当前正在写出的 PDB。`30yu` 已完成，不能再按旧状态把它当作未完成目录移动；只把实际未完成 PDB 的 `find_blobs` 移到带时间标识的备份目录，并复用全部 `status=ok` PDB。
3. 把动态命令中的 Emap2lig 根切换到 `Emap2lig_eb7a32a0ee70`，保留 detection batch size 16，并用项目锁协议重新启动。
4. Find 完成后核验 179 个官方状态，再等待 4-worker CPU 评估与 149-PDB `test_1` 派生。
5. 执行全量只读验收，生成 AdaLigand `收口の结果/Stage1/Emap2lig/` 下三份文档；最终让 Job 停回 `try_lock_383986` 并继续保留 `after_lock_383986`。

## Files To Reopen

- `文档/exec_plan/2026-09-16_Emap2lig官方Find_held_out评估.md`
- `src/inference/baseline/emap2lig_find.py`
- `训练与运行/sh/infer/emap2lig_official_find_li.sh`
- `CLAUDE/memory/handoffs/2026-09-16-emap2lig-official-find-running.md`
- `C:\Users\15919\Desktop\Emap2lig\src\emap2lig\main.py`
- `C:\Users\15919\Desktop\Emap2lig\复现Find\README.md`
