# Stage1 V3 推理测试

本目录验证 `src/inference/` 的正式科学和执行契约。`test_stage1_v3.py` 运行 CPU 科学契约测试，`test_stage1_cuda.py` 在可用 GPU 上运行真实 CUDA 异步 smoke；旧 Selector、组件森林、CLG、Li 和旧七种 Fα 的专项测试已经删除，历史行为只通过 Git 查阅。

`test_stage1_v3.py` 覆盖以下边界：

- 无填充 80³ 滑窗与归一化 Gaussian 权重；
- 26 邻域连通区域稳定排序，以及偏斜区域的合法 centered BOX；
- 正式 float32 blob 均值的排序 tie-break；
- centered offsets、`centered_box_index`、50 维 `A_feat_L0` 和 float16 特征；
- H100 BF16 输出进入 NumPy 前的显式 float32 转换；
- 5 Å Find Gaussian 分数；
- calibration 与正式 Find Gaussian 的逐位数值同源；
- 语义 micro/macro、双向覆盖、一对一匹配下标和 top-K 获胜 rank/index；
- Python 3.10 CLI、真正的 `ResolvedStage1Crop` Dataset 构造和重复 PDB 拒绝；
- 科学概率 NPZ 与性能 JSON 的字段隔离、checkpoint 路径一致性和原子完成标记。

运行命令：

```powershell
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest -q tests/inference/test_stage1_v3.py
```

测试使用 CPU 构造最小数组和临时目录，不需要正式 checkpoint。真实 checkpoint、CUDA 显存和 GPU 利用率属于 `ops/stage1_inference_benchmark/` 的实战验证。

CUDA smoke 单独运行：

```powershell
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest -q tests/inference/test_stage1_cuda.py
```

本目录与 `tests/datasets/test_stage1_dataset.py` 应在同一次 CPU 回归中运行。已经内联到 `produce_centered_role` 发布事务的 `selected` 字段由正式发布测试覆盖，不保留只测试薄包装的单独用例。实际测试次数、GPU 型号和运行结果记录在 AdaLigand `文档/exec_plan/Stage1_V3推理重写实施记录.md`。
