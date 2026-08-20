# Stage1 V3 推理模块结构概览

本文说明正式入口如何把 V3 Dataset、Stage1 wrapper、CPU worker 和单 GPU 连接起来。阅读代码时先看 `src/inference/README.md`, 再按下列顺序阅读。

## 推荐阅读顺序

1. `src/inference/artifacts.py`: 路径、状态和正式文件字段。
2. `src/inference/full_map.py`: 80³ 滑窗与异步融合。
3. `src/inference/blobs.py`: F1/F3 单阈值 26-连通区域。
4. `src/inference/centered.py`: 80³ centered 前向、48³ V-centered 切块和 A/P/V 字段。
5. `src/inference/scoring.py`: 基本分数与 Find A 原子高斯分数。
6. `src/inference/evaluation.py`: 阈值冻结、参数选择和 micro/macro 指标。
7. `src/inference/checkpoint.py`: 训练快照和 wrapper 恢复。
8. `src/inference/pipeline.py`: 三个生产阶段的直接编排。
9. `src/inference/cli.py`、`configs/inference/stage1_v3.yaml` 和 `训练与运行/sh/infer/stage1_v3.sh`: 正式命令。

## 调用关系

~~~text
cli.main
└── Stage1InferencePipeline
    ├── load_stage1_wrapper
    ├── Stage1Dataset + Stage1BatchCollator
    ├── infer_full_map
    │   ├── CPU window materialization
    │   ├── H2D + wrapper.forward_voxel_probability
    │   ├── D2H
    │   └── ordered CPU fusion
    ├── publish_probability_blobs
    │   └── scipy.ndimage.label, 26-neighborhood
    ├── run_centered_inference
    │   ├── CPU Stage1Dataset.materialize_request
    │   ├── voxel-only forward for F1
    │   ├── full wrapper forward for F3
    │   └── CPU pack + atomic NPZ publish
    └── evaluate_stage1
        ├── score_centered_candidates
        ├── occurrence overlap
        └── micro/macro/top-K reports
~~~

## 并行边界

- 单 GPU 只由 `Stage1InferencePipeline` 所在线程提交 CUDA 工作。
- 完整图内部的 CPU 物化、H2D、GPU 前向、D2H 和融合通过有界队列重叠。
- PDB k 的 probability 发布后, blobs 任务进入 CPU 进程池; GPU 随即处理 PDB k+1。
- centered 阶段由 CPU 线程准备连续候选 batch, GPU 保持原候选顺序执行, CPU 写出线程处理已完成 batch。
- 原子发布和 `_COMPLETE` 由 `artifacts.py` 统一拥有; 计算模块不自行发明路径或状态文件。

## 依赖方向

`artifacts.py` 不导入其他推理模块。`full_map.py`、`blobs.py`、`centered.py` 和 `scoring.py` 只依赖数组、PyTorch、SciPy 与 Dataset 公共字段。`evaluation.py` 依赖 artifacts、blobs 和 scoring 的稳定入口。`pipeline.py` 可以依赖全部阶段模块, 但阶段模块不得反向导入 pipeline。

`ops/stage1_inference/` 只保存性能基准和一次性对照命令, 不定义科学字段。测试可以导入正式模块; 正式模块不得导入测试、ops 或 Matcher。

## 有意跳过的代码

- `src/model/` 内的训练损失、候选 C 稀疏 refine 与调度器不在本轮修改范围; centered 完整前向只读取其既有稳定输出。
- `ops/stage1_data_preparation/` 已冻结 split 与 BOX pool, 本轮只消费其产物。
- Matcher `matcher/data/` 只用于确认 F3 字段, 不在 Pocket Plus 分支修改。
- Git 中的 Selector、CLG、forest、Li 和旧 Fα 实现只作为历史事实, 不恢复入口或兼容类型。
