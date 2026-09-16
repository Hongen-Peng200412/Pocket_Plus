# Emap2lig 官方 Find held-out 评估执行记录

本文记录 Emap2lig v0.3.4 官方 Find 在 Pocket Plus `held_out_06_chain` 测试集上的一次完整前向、标准 Stage1 指标评估、`test_1` 保序派生和结果文档收口。任务不执行参数校准，也不运行 Emap2lig-Build。

## 当前状态

- 2026-09-16：实现开始。Job `383986` 已在 `gnode09` 预留 A800×1、4 CPU，并稳定停在 `pre_lock_383986`；`after_lock_383986` 存在。正式代码和门控通过前不释放 `pre_lock`。
- Pocket Plus 的任务边界提交为 `13148ff`，Emap2lig 的任务边界提交为 `d2af4d0`。后续实现分别位于 `codex/emap2lig-held-out-evaluation` 分支。
- Job `383986` 第 1 次执行完成单 PDB 隔离门控。`9ter` 的官方 Find 共保存 279 个不少于 32 体素的实例；超过 100 个实例后仅产生告警，没有退出。随后同一批产物完成完整概率线性映射、279 个实例的保身份稀疏映射、逐实例交集与匹配事实计算和标准指标聚合。门控根为 `/storage/penghongen/tmp/emap2lig_official_find_li_gate_20260916/allocation_runner_job383986_20260916T092204_a1`。Job 已停回 `try_lock_383986`，`after_lock_383986` 保留。
- 09:40 关键事件：Job `383986` 第 2 次执行启动正式任务。AdaLigand allocation launch 为 `/home/penghongen/Feedback/AdaLigand/launches/383986/allocation_runner_job383986_20260916T094030_a2`；Pocket Plus release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_5570cf8a0b7a/Pocket_Plus`，Emap2lig release 为 `/home/penghongen/Feedback/Emap2lig/releases/Emap2lig_60e34f30b989/Emap2lig`。首个 `9ter` 已正常进入 512 个 ROI 的 GPU 前向，尚无异常；`after_lock_383986` 保留。
- 12:18 关键事件：Emap2lig 实例裁剪的结果等价优化通过完整 `9ter` 门控。原实现和串行局部实现都从同一份 `unified.mrc` 与 `ligand_mask.mrc` 读取 776 个原始连通域，均过滤 497 个少于 32 体素的连通域，并按相同顺序保存 279 个实例。全部 `blob_N.npz` 字段和数组，以及全部 `mask_N.mrc` 数组、原点、体素尺寸和数据类型均逐项使用 `np.array_equal()` 核验一致。原实现与优化实现的实例后处理墙钟时间分别为 70.714508 秒和 8.065864 秒，串行提升 8.767134 倍；优化后的 279 个实例只需约 8 秒，因此没有引入多进程写盘。
- Emap2lig 双线历史已经收口：实现端点为 `aff2c5f`，学习端点和 `Learn/CUMULATIVE` 为 `4e20f19`，两个端点的 Git tree 哈希均为 `96258424077f788639488cc711c7eca9ff38c6cd`。新 release 为 `/home/penghongen/Feedback/Emap2lig/releases/Emap2lig_eb7a32a0ee70/Emap2lig`，manifest 内容哈希为 `eb7a32a0ee702eff52e85734c7279b4bc7ad6c05db2a57192573e3ccaad5a5b8`。
- 12:29 只读状态：旧 release 的正式任务已经完成 25 个 `status=ok` 的 PDB，其中 `30yu` 已完整保存 1,878 个实例；当前日志正在处理 `9nnc`。Job `383986` 仍为 RUNNING，`after_lock_383986` 与原动态命令保留，`kill_lock_383986` 不存在。本阶段等待用户明确授权后才允许停止当前动态命令并切换到新 release。
- 12:54—12:56 关键事件：用户授权后，先冻结旧动态命令及进度证据，再创建 `kill_lock_383986`。旧进程以退出码 137 停止，allocation 随即创建 `try_lock_383986`；Job 始终为 RUNNING，`after_lock_383986` 始终保留。停止时已有 30 个 PDB 为 `status=ok`，实际未完成样本为 `9ifc`，其不完整 `find_blobs` 含 77 对 NPZ/MRC，已移动到 `find_blobs.incomplete_before_local_crop_20260916T125538`，没有删除。
- 第 3 次执行已从 launch `/home/penghongen/Feedback/AdaLigand/launches/383986/allocation_runner_job383986_20260916T125406_a3` 使用新 Emap2lig release 启动。`9ifc` 检测到全部既有 label maps/masks 后跳过模型前向；`ligand.mrc` 与 `ligand_mask.mrc` 的 SHA-256 在重启前后分别保持 `4167353c...d0799` 与 `3b9f4443...573e`。新实现从 12:54:51 至 12:55:21 保存 1,105 个实例并写出 `status=ok`，随后进入下一 PDB 的正常 GPU 前向。接管证据位于 `/storage/penghongen/tmp/emap2lig_local_blob_crop_gate_20260916/takeover_383986/`。
- 16:56—18:10 完成正式运行与最终验收。`test_0` 的官方 Find、映射候选和逐 PDB evaluation 均为 179/179；`test_1` 按清单顺序从同一批逐 PDB事实派生 149/149，没有重复模型前向或逐 PDB产物。正式任务成功退出后 Job `383986` 已停回 `try_lock_383986`，`after_lock_383986` 继续保留。

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
- Emap2lig 局部裁剪优化版 `main.py`：`8f8f538af0c1c91426c67292e5d7cebea42f06cb16876dd87e09c7c844c95b66`
- Emap2lig held-out runner：`7f95e3cd0f78cf2a0652ac42034c652674966ca9e6008c5abecf352dff3ecd28`

本节不混入单元测试、只读门控或临时核查命令。最终机器可读结果及 SHA-256 为：

- `held_out_test_0/evaluation/emap2lig_official.metrics.json`：`19a0bca8380ea2ea708cc0eb1b1867b8d7b4c2c94e6c810bdeed6446a361d100`
- `held_out_test_0/evaluation/emap2lig_official.jsonl`：`0f0a9bae817ef57e21caf1b3b0eb986765d559cc293775483dcb273b9cb6c0b0`
- `held_out_test_1/evaluation/emap2lig_official.metrics.json`：`d76c4e3d3284b1386ee585a9176d0731715b6623fd5c807b2b402f8a7aa79668`
- `held_out_test_1/evaluation/emap2lig_official.jsonl`：`fec1084cc4bf9c86a816b1bc1b8090f813ca78f903d3d95a26586e2c760338df`
- `held_out_test_1/evaluation/emap2lig_official.provenance.json`：`55b1d5b3d654368278c5a1b1a3387e166910e5013b9c1656e4c5464935fa845d`

## 门控与测试

- Pocket Plus 服务器定向测试：`1 passed`。
- Emap2lig Python 3.10 环境不含 pytest；同一环境直接执行 AST 与清单解析门控，结果为 `EMAP_GATE_OK`。
- Job `383986` 第 1 次执行是单 PDB 隔离门控，不是正式运行命令。该门控完整覆盖 `9ter` 官方 Find、`>100 blob` 行为、网格映射、实例事实保存和现有 Stage1 聚合函数。
- 局部裁剪合成测试使用同一组 64³ 密度与掩码，同时覆盖 31 体素过滤、Z 轴长度 50 的实例和体积尾部边界实例。新旧实现的 NPZ 全字段和 MRC 科学字段逐项一致；三个既有及新增门控函数在服务器 Python 3.10 环境直接执行通过。
- 完整 `9ter` 性能门控由 CPU Job `384166` 执行。临时脚本 Job `384160` 在原实现写出 279 个实例后因 `numpy.int64` 不能直接编码为 JSON 而退出，没有进入等价比较；其产物保存在 `failed_384160_reference/`。临时计数显式转换为 Python `int` 后，Job `384166` 正常完成，科学比较没有发现差异。
- `9ter` 门控根为 `/storage/penghongen/tmp/emap2lig_local_blob_crop_gate_20260916`。`comparison.json` 的 SHA-256 为 `cc751494082af56db42c52b4eecfc815ea4775930eb72de044d56af694907317`；原实现和优化实现的 `summary.json` SHA-256 分别为 `5163568ad7070c41ce032ea51856ab6f9689040f63676100eff24582d1b64e7f` 和 `6153e105efa1808f365d6f49c43e09c09a70c5304349cf2c381a5c0db01afdcd`。门控源码快照及 `SHA256SUMS` 保存在同一根目录的 `source_snapshot/`。
- 最终只读验收报告为 `/storage/penghongen/tmp/emap2lig_local_blob_crop_gate_20260916/final_validation.json`，SHA-256 为 `e70e748321597ba8978e8aca3d4473956ee4e14fb9fd5eefb0e84881061405fa`。报告核对 63,342 个官方保留实例；117 个 PDB 超过 100 个实例，最多的 `11jb` 为 5,100 个，均未被过滤。逐实例分数、候选顺序、稀疏坐标、交集、匹配索引、汇总指标与 top-K 算术全部通过。

## 结果文档与收口

- 本地结果文档位于 `C:\Users\15919\Desktop\AdaLigand\收口の结果\Stage1\Emap2lig\`：`主要结果.md`、`说明.md`、`补充结果.md`。
- 三份文档中的六位小数指标、top-K 计数与百分比均由最终机器可读结果自动核对通过；AdaLigand 文档提交为 `e7d80db docs: record Emap2lig Stage1 results`。
- 按用户约定，三份文档尚未同步到服务器。正式产物、最终验收报告、执行记录和 handoff 已完成收口。

本轮性能门控命令与正式运行命令分开。CPU Job `384166` 的门控提交命令为：

```bash
sbatch --job-name=emap_crop_gate --partition=cpu --qos=Cpu96 --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --output=/storage/penghongen/tmp/emap2lig_local_blob_crop_gate_20260916/slurm_%j.out --error=/storage/penghongen/tmp/emap2lig_local_blob_crop_gate_20260916/slurm_%j.err /home/penghongen/My_Project/Map_Ligand/Emap2lig/tmp/emap2lig_local_blob_crop_gate_20260916/run_gate.sh
```

## 计划与实现差异

- 资源差异：原计划沿用 Job `368455` 的 H100×2、64 CPU；用户改为 Job `383986` 的 A800×1、4 CPU。官方 Find 仍仅使用 GPU 0；CPU 评估并发上限相应从 8 调整为 4，不增加双卡分片。
- 中性实现差异：串行局部裁剪在完整 `9ter` 上达到 8.767134 倍加速，低于“达到至少 10 倍即可停止优化”的充分条件，但 279 个实例的绝对墙钟时间已经降至 8.065864 秒，不再明显不足。按照“仅在串行仍明显不足时增加并行”的限制，本轮没有加入进程池。
