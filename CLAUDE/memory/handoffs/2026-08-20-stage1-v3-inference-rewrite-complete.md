# Handoff: Stage1 V3 推理重写本地收口

Date: 2026-08-20

## Current State

Stage1 V3 推理已在本地完成代码、契约、测试、三类独立审查和双线 Git 收口。活动入口是 `python -m src.inference.cli`，由 `训练与运行/sh/infer/stage1_v3.sh` 调用；未提交服务器推理任务，也未使用真实最终 checkpoint 生产正式结果。

实现分支的科学代码端点是 `codex/stage1-inference-v3@bdecd9802fdef4dcc28401411b35f31a72e5a3cc`。学习线的科学代码等价节点是 `c62de259f48da4f04962d4b68658a76a6afbafbe`，两端 Git tree 均为 `226b56e7ed34b44545af6236afba4653caac32b9`。本 handoff 与项目记忆作为不参与运行的学习记录追加在该节点之后。

## Completed

- 删除旧 Selector、组件森林、CLG、Li、多 F-alpha 推理入口和对应测试，只通过 Git 历史读取。
- `unet_c1`、`Find_0`、`Find_1`、`Find_2` 共用 F1 basic 与 F3 centered 流程；Matcher 首先消费 F3。
- 完整图概率不乘受体 hardmask；stride 和保存开关由配置显式提供。
- 实现概率、26 邻域 blobs、centered、Find Gaussian 三阶段校准、micro/PDB 等权 macro 评估和原子发布。
- 概率压缩、CPU blobs、centered CPU 整理与相邻 PDB 的 GPU 前向采用有界重叠；正式 F3 不为 score/selected 二次完整解压压缩。
- calibration 最终按冻结 `min_voxels` 重新生成正式 centered；低于阈值的候选只保留在 blobs。
- calibration JSON 与 `_COMPLETE` 共同绑定运行身份；概率角色不跨 checkpoint 复用。
- Python 3.10、Hydra dataclass 转换、BF16 到 NumPy、共享 mmap LRU、CUDA 目标 stream、blob float32 排序等边界已修复。
- 本机测试为 29 passed，其中包含 RTX CUDA 的 125 窗口真实 Conv3d 流水；compileall、CLI、OmegaConf、Bash 语法和 Git 差异检查通过。
- 布局/Git、注释/Docstring、科学逻辑三类审查均在两轮全面审查后完成窄口径批准。

## Decisions

- 所有 producer 都有 F1 basic 与 F3 centered；F1 不保存 `voxel_final`，F3 保存。
- Find F3 使用 A 原子 Gaussian 分数；U-Net F3 与所有 F1 使用来源 blob 平均概率。
- `_BLOB_EXCEED` 只记录严格超量事实，不跳过 PDB。
- `A_feat_L0` 在模型边界由 49 维 receptor token 与 `is_backbone` 拼成 float32 50 维，不改写上游 NPZ。
- 正式命令总是为当前 checkpoint 和推理配置重新计算 probability，因为 PDB 角色 `_COMPLETE` 不带生产身份。

## Open Questions

- 最终使用哪些 `unet_c1` 和 Find checkpoint 进行实战，仍由用户确认。
- 真实 checkpoint 的显存、BOX/s、PDB/s、GPU active ratio 和串行/流水线对比尚未测量。
- Matcher 对最终 F3 scientific fields 的首轮实战验收尚未开始。

## Next Actions

1. 用户确认 checkpoint 与服务器运行授权后，先用短 calibration 清单执行 `unet_c1` 和一个 Find 的真实 smoke。
2. 用 `ops/stage1_inference_benchmark/benchmark_gpu_utilization.py` 对同一科学配置分别记录 serial 与 pipeline，并由第二次运行引用第一次 `summary.json`。
3. 核对五类 NPZ、角色 `_COMPLETE`、calibration identity 和逐 PDB 评估字段后，再扩大到 validation/train。
4. 真实 F3 通过验收后，交给 Matcher 首先消费 `F3_centered.npz`。

## Files To Reopen

- `talk/refactor/stage1_v3_inference.md`
- `src/inference/README.md`
- `src/inference/workflow.py`
- `configs/inference/stage1_v3.yaml`
- `训练与运行/sh/infer/README.md`
- `ops/stage1_inference_benchmark/README.md`
- `C:/Users/15919/Desktop/AdaLigand/文档/规划文档/BOX-level数据契约.md`
- `C:/Users/15919/Desktop/AdaLigand/文档/exec_plan/Stage1_V3推理重写实施记录.md`
