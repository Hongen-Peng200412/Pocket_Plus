# Stage1 V3 推理配置

`stage1_v3.yaml` 是四个 producer 共用的科学与并行配置。producer、checkpoint、训练最终配置、PDB 清单、数据划分和输出根目录由 `src.inference.cli` 的命令参数显式传入。

配置中的 YAML anchor 只复用 F1/F3 完全相同的 centered batch、CPU 线程数、预取深度和精度；F1/F3 的前向模式、保存字段、阈值目标和评分规则仍在各自角色下明确写出。

字段读取链路：

```text
stage1_v3.yaml
→ src.inference.cli.main
→ src.inference.pipeline
→ full_map / centered / calibration / evaluation
```

`stride_zyx` 和 `save_voxel_final`、`save_dense48` 没有 Python 默认值。修改推理策略时直接修改本文件并依靠 release/launch 保存本次真实配置；不要在 shell 中建立另一套同名科学参数。
