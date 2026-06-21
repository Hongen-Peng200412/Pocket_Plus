# Phase 6 handoff：CPC v3 配置纠偏与最终收口

更新时间：2026-06-19 12:58:00 +08:00

本文件是 `最后重构代码_v3.md` 的最终有效收口记录。此前 `phase5_最终测试与_main试跑.md` 使用了旧入口 `+experiment=CPC/CPC_main`，已被用户纠正并作废。最终 `_main` 入口必须是 `+experiment=CPC1/trunk_main`。

## 本阶段完成内容

- 补齐 `configs/experiment/CPC1/` 的 4 个实体 YAML：
  `trunk_main`、`trunk_no_real_density`、`trunk_hard_scatter`、`trunk_decoder_fusion`。
- 补齐 `configs/experiment/CPC2/` 的 4 个实体 YAML：
  `heads_trunk_main`、`heads_trunk_no_real_density`、`heads_trunk_hard_scatter`、`heads_trunk_decoder_fusion`。
- 补齐 `configs/experiment/CPC3/` 的 9 个实体 YAML：
  `refine_tversky_{73,82,91}`、`refine_tversky_bce_{73,82,91}`、`refine_tversky_rank_{73,82,91}`。
- 修正 CPC1/CPC3 内部继承为绝对 Hydra group 路径，避免服务器 compose 将 `trunk_main` 误解析成 `experiment/trunk_main`。
- 新增并收紧 `tests/test_cpc_v3_configs.py`，守护 4/4/9 文件数、顶部启动命令、`CPC1/trunk_main` 最终入口、CPC2/CPC3 阶段边界、CPC3 3x3 网格和 Hydra compose。
- `src/model/stage1_model.py` 已移除旧约束：启用 sparse refine 不再要求 atom head 构造耦合；refine 依赖 final P before/after interaction 特征。

## 服务器纪律

- 只使用用户指定资源：`/home/penghongen/try_lock_297515` 对应 Slurm job `297515`。
- 使用项目 safe sync：`与服务器交互/sync_code.ps1`；没有使用 clean sync。
- 未触碰 `kill_lock_297515`。
- 未删除 `after_lock_297515`。
- 试跑没有设置“10 分钟后退出”逻辑；观察满 10 分钟后让训练继续留在原卡运行。

## 最终全量测试

在 `try_lock_297515` 对应 job 中执行：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
python -m pytest tests -q
```

最终结果：

```text
283 passed, 8 warnings in 18.93s
```

测试中曾暴露一次真实配置漂移：

```text
hydra.errors.MissingConfigException: In 'experiment/CPC1/trunk_decoder_fusion': Could not load 'experiment/trunk_main'
```

根因是 CPC1/CPC3 内部继承使用了相对路径。已改为 `/experiment/CPC1/...` 和 `/experiment/CPC3/...`，并收紧测试，防止再次漂移。

## `_main` 真实试跑

最终 `_main` 使用的配置入口：

```text
configs/experiment/CPC1/trunk_main.yaml
```

通过 `/home/penghongen/run_cmd_297515.sh` 写入并由 `/home/penghongen/try_lock_297515` 触发的命令：

```bash
cd /home/penghongen/My_Project/Pocket_Plus
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
python -m pytest tests -q
python /home/penghongen/My_Project/Pocket_Plus/src/train.py "+experiment=CPC1/trunk_main" train.nnodes=1 train.devices=1
```

真实运行时间点：

- 全量测试开始：`2026-06-19T12:42:20+0800`。
- 全量测试通过并启动 `CPC1/trunk_main`：`2026-06-19T12:42:41+0800`。
- 最终观察时间：`2026-06-19 12:53:52 +0800`。
- 连续观察时长：约 11 分 11 秒。

观察结论：

- Slurm job `297515` 仍为 `R` 状态，节点为 `hnode01`。
- `/home/penghongen/try_lock_297515` 未重新出现，说明训练没有异常退出回到 lock 暂停态。
- `/home/penghongen/after_lock_297515` 仍存在，资源未释放。
- 日志显示训练已进入 `trainer.fit`，输出目录为 `/home/penghongen/My_Project/feedback_plus/logs/CPC1/trunk_main____job297515`。

## 日志位置

Slurm 日志：

```text
/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.out
/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.err
```

训练目录：

```text
/home/penghongen/My_Project/feedback_plus/logs/CPC1/trunk_main____job297515
```

注意：同一个 Slurm 日志中保留了早前失败记录，包括旧 `CPC/CPC_main` 的 `ModelCheckpoint` 冲突、用户手动 `kill_lock_297515` 终止旧任务、以及第一次 CPC v3 compose 失败。最终有效记录以 `[CodexCloseoutV2] 2026-06-19T12:42:20+0800` 之后的段落为准。

## 给下一位 agent 的注意事项

- 不要使用旧 `configs/experiment/CPC_old` 或旧 `+experiment=CPC/CPC_main` 做最终验收。
- 不要为了兼容旧测试或旧配置加回退逻辑；历史测试若断言旧结构，应继续改测试本身。
- `BestCheckpointAlias` 是用户接受的实现偏差：`BEST.ckpt` 由普通 callback 复制主 `ModelCheckpoint.best_model_path`，不是第二个 `ModelCheckpoint`。
- 如果需要查看训练是否还在跑，只读执行：

```bash
squeue -j 297515
ls -l /home/penghongen/*_lock_297515 /home/penghongen/run_cmd_297515.sh
tail /home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/*297515.out
```

- 不要删除 `after_lock_297515`，除非用户明确要求释放资源。
- 不要触碰 `kill_lock_297515`，除非用户明确要求终止该资源。
