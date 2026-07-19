# AdaLigand Stage1 推理生产链

`python -m src.inference.cli` 是完整图、校准、组件森林、F1/CLG 居中和 Selected-refined 的唯一生产入口。旧推理/评估/可视化编排已从本分支移除；`utils/receptor_strip.py` 仅因制图代码仍依赖而保留，不属于 Stage1 runner。

## 固定阶段

每个 producer（`Find_0`、`Find_1`、`unet_c1`）按以下顺序运行。示例参数均须替换为当前 run 的绝对路径；`--pdb-list` 是一行一个 PDB ID 的冻结清单。

```bash
# 1. calibration 100：只生成完整图概率
python -m src.inference.cli cal-probability \
  --producer Find_0 --pdb-list CAL_IDS.txt \
  --data-root /absolute/path/to/Ori_Data \
  --checkpoint /absolute/path/to/BEST.ckpt \
  --config /absolute/path/to/resolved_config.yaml \
  --device cuda --output-root /absolute/path/to/stage1_outputs

# 2. 从完整 calibration 清单冻结阈值并汇报 fitted 指标
python -m src.inference.cli freeze-thresholds \
  --producer Find_0 --pdb-list CAL_IDS.txt \
  --data-root /absolute/path/to/Ori_Data \
  --output-root /absolute/path/to/stage1_outputs

# 3. 给 calibration 补齐 components、F1_centered、CLG_centered
python -m src.inference.cli cal-produce-f1-clg [同一组 runtime 参数]

# 4–5. validation/train：每张图立即连续发布 probability→components→两个 centered role
python -m src.inference.cli val-produce-prob-f1-clg [同一组 runtime 参数]
python -m src.inference.cli train-produce-prob-f1-clg [同一组 runtime 参数]

# 6. Selector 发布 selection.npz 后，按选定局部阈值重新居中推理
python -m src.inference.cli selected-refined \
  --split train --producer Find_0 --pdb-list TRAIN_IDS.txt \
  --selection-root /absolute/path/to/selections \
  --data-root /absolute/path/to/Ori_Data \
  --checkpoint /absolute/path/to/BEST.ckpt \
  --config /absolute/path/to/resolved_config.yaml \
  --device cuda --output-root /absolute/path/to/stage1_outputs
```

完整参数以 `python -m src.inference.cli <command> --help` 为准。正式默认使用 80³ 窗口、stride 40、三次 recycle、32768 阈值分母和 `max_voxels=1023`。

## 续跑、分片与发布

任务身份是 `(producer, split, pdb_id)`。`--shard-index i --shard-count n` 按冻结清单行号取模，卡数增加或缩减时可用新的分片参数重扫；已经完成的 role 会跳过，因此允许重跑。建议先完成每个 producer 的 calibration 概率和阈值冻结；随后每个 worker 消费自己的 validation、calibration 未完成份额，再消费 train。调度层不写死 GPU 数量。

每个 PDB 只有一个 `_RUNNING` 目录作为互斥租约。下游仅在不存在 `_RUNNING/_BLOB_EXCEED`，且所需 `status/<role>/_COMPLETE` 全部存在时读取。`_BLOB_EXCEED` 表示 `t_F1` 层 eligible component 超过 200，是可识别终态而不是失败重跑信号。payload 先原子写入，再发布 role 完成标志。

`Selected_Refined_Centered` 只重跑并匹配 Selector 已选定的来源节点：局部新增的无关组件不会成为新节点，也不会改写完整图组件森林。其 NPZ 仍保存来源节点对应关系、BOX 几何、A/P/V 和各类概率。
