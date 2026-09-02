# Stage1 V3 推理测试

本目录验证 `src/inference/` 的正式科学和执行契约。`test_stage1_v3.py` 运行 CPU 科学契约测试，`test_calibration_parallel.py` 验证 basic/Gaussian 外层并发与串行结果等价，`test_stage1_cuda.py` 在可用 GPU 上运行真实 CUDA 异步 smoke；旧 Selector、组件森林、CLG、Li、固定 `F1_basic/F3_centered` 和 `calibrate/run` 流程的专项测试已经删除，历史行为只通过 Git 查阅。

`test_stage1_v3.py` 覆盖以下边界：

- 无填充 80³ 滑窗、归一化 Gaussian 权重、跨 PDB session 复用和下一 PDB 首批预取；
- 26 邻域连通区域稳定排序，以及偏斜区域的合法 centered BOX；
- 正式 float32 blob 均值的排序 tie-break；
- centered offsets、`centered_box_index`、同一 batch 内逐请求并发物化、所有 producer 固定保存的 auxiliary 与三张 48³ 数组、50 维 `A_feat_L0` 和 float16 特征；
- 无法被单个 80³ BOX 完整容纳的 blob 仍执行前向，只归档框内真实体素，并单独保存完整来源体素数与可容纳标志；
- 合成 CPU BF16 输出进入 NumPy 前的显式 float32 转换；
- 5 Å Find Gaussian 纳入端点与 10 Å Find A 表保留端点；
- calibration 与正式 Find Gaussian 的逐位数值同源；
- basic 与 Gaussian 在参数搜索前固定同一个预过滤门槛，且该值不限制最终 `min_voxels` 搜索列表；
- 不均衡 PDB 规模下的语义 macro 阈值选择，以及 evaluate 同时发布三类 micro/macro F-beta、semantic PRAUC、coverage PRAUC 与 one-to-one PRAUC；候选 PRAUC 另验证实际 float32 分数、最终分数阈值无关性和固定体素门槛；semantic PRAUC 以相邻 float32 端点证明 1024 个阈值与训练 TorchMetrics 一致；
- Python 3.10 CLI、真正的 `ResolvedStage1Crop` Dataset 构造和重复 PDB 拒绝；
- 动态 `F{alpha}` 路径标签、probability/centered 正式阶段和固定种子 3407 的随机分片；
- evaluate 显式结果名、全候选不做二次打分，以及全候选与参数过滤结果并存；
- centered 全模型前向、score-only 只替换两个选择字段，以及来源 blob 数严格大于 1000 时的默认跳过和显式提示模式；
- 科学概率 NPZ 与性能 JSON 的字段隔离和原子完成标记。

`test_calibration_parallel.py` 另外覆盖：

- basic 实际分数扫描、Gaussian 粗搜、细搜和最小体素数搜索使用同一个 PDB 等权 macro 三项目标；
- 大体积与小体积 PDB 对阈值意见相反时，basic 和 Gaussian 都不退回 micro 选择；
- 数学上并列的 basic macro 目标不被浮点差量累计破坏，仍保留先遇到的高分阈值；

- `workers=1` 与 `workers>1` 的 basic 完整选择 JSON 逐字段相等，并用同步屏障证明最终 `min_voxels` 目标确实由多个线程重叠计算；
- `workers=1` 与 `workers>1` 的 Gaussian 粗搜、细搜、每组实际分数扫描和最终体素门槛结果逐字段相等；
- 用线程事件分别阻塞粗搜、细搜和最终体素门槛的首个参数任务，并让全部目标值并列，证明三个阶段乱序完成时仍按配置原顺序保留首项；
- `run_tune_stage()` 以两个 worker 并行读取临时正式 NPZ，保持 PDB 清单顺序，完成 basic 字段改名、Gaussian A 原子字段读取和两类 JSON 发布。

运行命令：

```powershell
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest -q tests/inference/test_stage1_v3.py tests/inference/test_calibration_parallel.py
```

测试使用 CPU 构造最小数组和临时目录，不需要正式 checkpoint。真实 checkpoint、CUDA 显存和 GPU 利用率属于 `ops/stage1_inference_benchmark/` 的实战验证。

CUDA smoke 单独运行，覆盖 probability 与 centered 的真实 CUDA 前向、异步 D2H 和 CPU 收口：

```powershell
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest -q tests/inference/test_stage1_cuda.py
```

完整 CPU 回归同时运行 `tests/inference/test_stage1_v3.py`、`tests/inference/test_calibration_parallel.py` 与 `tests/datasets/test_stage1_dataset.py`。`run_centered_stage()` 的首次发布和 score-only 更新，以及 `run_tune_stage()` 的 basic/Gaussian 并行加载、字段转换与 JSON 发布，都由正式入口测试覆盖，不保留只测试薄包装的单独用例。实际测试次数、GPU 型号和运行结果记录在 AdaLigand 的对应执行记录。
