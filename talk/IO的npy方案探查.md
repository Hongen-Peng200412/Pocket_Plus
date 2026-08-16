# Stage1 完整体数组迁出 NPZ 的 NPY 方案探查

本文记录一次只在服务器临时目录中进行的 CPU Dataset 实验，用于回答以下问题：把 Stage1 Find 每个 PDB 的四个完整体数组从压缩 NPZ 移到独立 NPY，并在读取 80³ BOX 时使用内存映射（memory map，简称 mmap），能否减少完整图解压、进程内存和 Dataset 等待时间。

当前结论是：**NPY+mmap 明显优于现有 NPZ 整图物化方式，值得进入正式实现讨论；但本实验不能单独证明 H200 训练的数据等待已经完全消失。**在 16 核 CPU、非 PDB 分组请求、16 workers、prefetch=4 的同条件比较中，Dataset 吞吐从 3.34 BOX/s 提高到 8.80 BOX/s，即 2.63×；worker 进程 RSS 峰值合计从 179.09 GiB 降到 30.67 GiB。若允许 NPZ 与 mmap 分别选择各自的最佳参数，本次观测到的最佳吞吐分别为 3.99 和 8.80 BOX/s，即 2.20×。

必须始终保留两个真实规模边界：正式训练集超过 1 万个 PDB，validation 是 200 个 PDB。本次只转换并计时 100 个合法 PDB，因此足以比较文件格式与单请求物化代价，但不能直接代表 1 万以上 PDB 的 worker-local 缓存命中率，也不能代替 200-PDB validation 的独立观察。

## 候选文件契约

每个 PDB 仍保留原有四个 NPZ 文件和相对路径，只把下表中的完整体数组移到同目录 NPY。NPZ 中未列出的字段继续保留，字段值不改变。

| 原文件与字段 | 新增 NPY | 数组契约 | 迁移后原 NPZ |
| --- | --- | --- | --- |
| `exp.npz:grid` | `exp.npy` | `float32 (1,D,H,W)`，实验密度 | 保留 `grid` 之外的字段 |
| `sim.npz:grid` | `sim.npy` | `float32 (1,D,H,W)`，模拟密度 | 保留 `grid` 之外的字段 |
| `ligand_dist.npz:distance` | `ligand_dist.npy` | `float16 (1,D,H,W)`，最近配体原子距离 | 保留 `distance` 之外的空间契约字段 |
| `ligand_area.npz:union_mask` | `union_mask.npy` | `bool (1,D,H,W)`，全部 occurrence 配体区域并集 | 保留 `union_mask` 之外的字段 |

训练读取 NPY 时使用 `np.load(path, mmap_mode="r")`，必须先在 mmap 数组上裁出 80³ BOX，之后才允许复制、转换 dtype 或构造 Tensor。不能先把完整 NPY 转成普通 `ndarray` 或 Tensor。

临时转换的核心操作是：

```python
with np.load(npz_path, allow_pickle=False) as source:
    np.save(npy_path, source[large_key], allow_pickle=False)
    remaining = {key: source[key] for key in source.files if key != large_key}
np.savez_compressed(rewritten_npz_path, **remaining)
```

正式迁移时，`rewritten_npz_path` 最终替换同一 PDB 的原 NPZ；本次探查没有替换正式数据，而是直接在 `/storage/penghongen/tmp` 写出目标格式，旧基线继续读取正式 NPZ。

## “冷缓存”和“热缓存”的准确含义

本任务至少存在三层不同缓存，不能统称为一个缓存：

1. **Dataset worker 内的 `_ByteLruCache`**：每个 worker 进程独立保存最近读取的 PDB 资产。只有同一 PDB 的后续请求再次落到同一 worker，并且此前资产尚未淘汰时，才会命中。
2. **计算节点的 Linux 文件页缓存**：操作系统保留最近访问的文件页。不同 worker 重新打开同一文件时，也可能复用这些页；NPY+mmap 主要依靠这一层只调入目标 BOX 对应的页。
3. **Lustre 服务端缓存**：存储服务器可能保留近期数据。本次作业没有权限精确清空或控制这一层。

本文中的“冷缓存建议”是：计时前对相关文件调用 `posix_fadvise(..., POSIX_FADV_DONTNEED)`，请求计算节点丢弃本地文件页。该调用只是建议，不能保证 Lustre 服务端也处于冷状态，因此不能称为绝对冷缓存。

“热缓存”是：文件刚被前一轮访问后，不执行上述丢弃建议就重复运行。当前配置使用 `persistent_workers=false`，每轮会重建 Dataset worker，所以热缓存主要表示 Linux/Lustre 文件页更可能已经驻留，**不表示上一轮 worker 的 `_ByteLruCache` 仍然存在**。

最初的一次预实验把同一 PDB 的请求连续排列，并比较了冷、热两种状态。用户确认正式请求不按 PDB 分组后，该预实验的性能数字不再作为最终证据；它只用于检查转换等价性和建立缓存术语。最终参数结论全部来自后续的非 PDB 分组测试。

## 临时实验边界

临时根目录为：

```text
/storage/penghongen/tmp/stage1_npz_npy_io_20260816_Lt1PEk
```

实验遵守以下边界：

- 正式数据根 `/storage/penghongen/AdaLigand/Ori_Data` 只读。
- 服务器 Pocket Plus 主仓库只读；临时 Python 脚本不被正式包或训练入口调用。
- Slurm 作业只申请 CPU，不申请 GPU；主要计时作业均为 16 核 CPU。
- Dataset 使用服务器现有 `Stage1Dataset`、Find_1 的完整密度通道构造、监督构造和 `collate_fn`，不只计时四次数组切片。
- `batch_size=8`，每个 PDB 固定 4 个 BOX，共 400 BOX。
- 100 个 PDB 的 400 个请求使用 seed 3407 全局打散，不按 PDB 分组。
- 每种参数均使用相同 PDB、BOX 起点、请求顺序和随机种子。
- worker 内缓存上限为 32 GiB；参数扫描只改变 `num_workers`、`prefetch_factor` 和 NPZ/NPY 读取方式。

最初从整个 `density/` 随机选出的 100 个目录中，有 10 个 PDB 的 `exp` 与 `sim` shape、`voxel_size` 或 `origin` 不一致，真实 Dataset 按既有契约拒绝读取。实验没有绕过该检查，而是排除这 10 个目录，再补充 10 个合法 PDB，最终计时集合恰为 100 个合法 PDB。

转换验证包括：

- 100 个 PDB 的四个迁出数组均核对 dtype 与 shape；
- 每个迁出数组至少核对一个相同 80³ BOX，共核对 400 个 BOX；
- 前 5 个 PDB 的 NPZ 剩余字段集合和值不变；
- 4 个完整 Find_1 Dataset 样本逐字段、逐 Tensor 精确相等。

## 磁盘占用

100 个合法 PDB 的结果如下。GiB 按 $2^{30}$ 字节计算。

| 状态 | 文件集合 | 占用 |
| --- | --- | ---: |
| 当前格式 | 四份原 NPZ | 42.16 GiB |
| 目标格式 | 四份去除大字段的 NPZ + 四份 NPY | 46.36 GiB |
| 正式替换后的净增加 | 目标格式减当前格式 | 4.21 GiB（+9.98%） |

若正式替换，旧 NPZ 中的大字段会被移除，因此不是在当前格式上再永久增加 46.36 GiB，而是净增加约 4.21 GiB。本次模拟不允许修改正式数据，所以临时目录额外保存完整目标格式；再加上被排除 PDB 的临时转换文件，临时目录实际约占 50.40 GiB（`du -sh` 显示 51G）。

这 100 个 PDB 不是按体积分层抽样，不能把 4.21 GiB 机械乘 100 当作 1 万 PDB 的精确正式增量。正式迁移前仍需对全部 1 万以上 PDB 只读统计原 NPZ 压缩大小和四个数组原始字节数。

## 非 PDB 分组请求与缓存命中上限

打散后的 400 个请求具有以下特征：

- 399 个相邻请求对中，只有 3 对属于同一 PDB；
- 每个 8-BOX 物理 batch 平均包含 7.74 个不同 PDB；
- 按 DataLoader 把 batch 轮转给 worker 的顺序估算，同一 worker 曾经见过该 PDB 的请求比例为：8 workers 时 16.5%，16 workers 时 8.5%，24 workers 时 5.0%。

这个比例仍只是“可能命中”的上限：它没有扣除 LRU 淘汰。正式训练包含 1 万以上 PDB，PDB 重现距离比本次 100-PDB 实验大约再增加两个数量级，因此正式训练的 worker-local 命中率可能更低。validation 只有 200 个 PDB，工作集与本次实验属于同一数量级，缓存复用可能明显高于训练，但仍取决于 validation 请求顺序和 batch→worker 分配，必须单独观察。

## workers 与 prefetch 参数扫描

`prefetch_factor` 表示每个 worker 预先准备的 batch 数。所有下表结果都来自同一 16 核 CPU allocation、相同 400 个非分组请求和冷缓存建议。worker RSS 是主进程轮询各 worker 的 RSS 后求和的峰值；共享页可能被多次计数，因此它适合比较相对变化，不等于节点唯一物理内存占用。

| workers | prefetch | NPZ BOX/s | mmap BOX/s | 同参数加速 | NPZ worker RSS | mmap worker RSS | mmap RSS 减少 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 2.19 | 3.87 | 1.77× | 163.87 GiB | 18.63 GiB | 88.6% |
| 16 | 1 | 2.95 | 7.88 | 2.67× | 181.16 GiB | 30.83 GiB | 83.0% |
| 16 | 2 | 3.99 | 5.36 | 1.34× | 179.01 GiB | 25.27 GiB | 85.9% |
| 16 | 4 | 3.34 | 8.80 | 2.63× | 179.09 GiB | 30.67 GiB | 82.9% |
| 24 | 4 | 3.34 | 5.77 | 1.73× | 188.78 GiB | 33.57 GiB | 82.2% |

参数结论是：

- 8 workers 并发不足，NPZ 和 mmap 都慢。
- 16 workers 是当前 16 核 allocation 的有效区域。mmap 在 prefetch=1 和 prefetch=4 都超过同参数 NPZ 的 2.5×；本次最高吞吐是 16 workers、prefetch=4 的 8.80 BOX/s。
- prefetch=2 的 mmap 低点表明单次 Lustre 测量存在波动，三个点不呈单调关系。不能仅根据一次 400-BOX 测量断言 prefetch=2 必然较差，但 prefetch=4 至少没有阻止当前最高吞吐。
- 24 workers 超过 16 核 CPU 数后，NPZ 没有提高，mmap 从 8.80 降到 5.77 BOX/s；首 batch 等待也显著变长。24 workers 不推荐用于同样 16 核 allocation。
- 用户在 24-worker 趋势明确后叫停 32 workers。32-worker mmap 单边结果恰在叫停消息到达前完成，但对应 NPZ 正在运行时作业已取消；该数字没有同配置对照，不纳入表格、推荐或结论。

若两种格式分别选择本次各自最高吞吐，NPZ 最佳为 16 workers、prefetch=2 的 3.99 BOX/s，mmap 最佳为 16 workers、prefetch=4 的 8.80 BOX/s，最佳对最佳为 2.20×。因此可以说同一常用配置下格式替换超过 2.5×，但不能说对“分别调优后的最佳数据管线”已经证明 2.5×。

## 是否应关闭完整体数组缓存

额外实验固定 16 workers、prefetch=4，并保持 receptor 表与标签等小资产缓存不变，只禁止 `exp`、`sim`、`union_mask` 和 `distance` 进入 worker-local LRU。

| 格式 | 体数组缓存 | BOX/s | worker RSS 峰值合计 |
| --- | --- | ---: | ---: |
| mmap | 保留现有 LRU 行为 | 8.80 | 30.67 GiB |
| mmap | 禁止四个体数组进入 LRU | 5.11 | 24.08 GiB |
| NPZ | 保留现有 LRU 行为 | 3.34 | 179.09 GiB |
| NPZ | 禁止四个体数组进入 LRU | 2.78 | 38.99 GiB |

在本次 100-PDB 请求中，完全关闭体数组缓存没有加速：mmap 吞吐下降约 42%，NPZ 下降约 17%。mmap 的现有缓存保存的是 memmap 映射对象和已打开文件关系，不会像 NPZ 那样立即把完整体数组全部物化到匿名内存；保留它可以减少重复文件打开和元数据读取。

但该结论不能直接扩大到 1 万以上训练 PDB。100-PDB 实验的 worker-local 重现率仍可能显著高估正式训练。正式实现的最小风险边界应是：先复用现有 LRU 行为，不新增复杂句柄缓存；在真实 1 万+ PDB 训练请求轨迹和 200-PDB validation 中分别记录体数组 cache hit、miss、eviction 与 worker RSS，再决定是否缩小或关闭训练体数组缓存。训练与 validation 不应共用一个未经测量的缓存结论。

## 当前建议与尚未证明的内容

当前证据支持以下最小实现方向：

1. 把 `exp.npz:grid`、`sim.npz:grid`、`ligand_dist.npz:distance` 和 `ligand_area.npz:union_mask` 移到四个独立 NPY；原 NPZ 只移除对应字段，其余字段和值保持不变。
2. Dataset 使用 `mmap_mode="r"` 打开完整体数组，先裁 80³ BOX，再复制为后续可写数组或 Tensor。
3. 第一版复用现有 `_ByteLruCache`，不立即设计 HDF5、Zarr、共享内存、跨 worker IPC 或新的句柄缓存。
4. 在同样 16 核 CPU 条件下，下一次端到端候选参数优先使用 16 workers、prefetch=4；不能据此直接改写不同 CPU 配额的正式 H200 参数。
5. validation 的 200 个 PDB 必须单独计时和观察缓存命中，不能由 1 万+ PDB 的训练结论代替。

8.80 BOX/s 等价于平均约 0.91 秒准备一个 8-BOX batch。只有当真实 H200 的模型前向、反向和优化器步骤耗时不短于数据供应，且训练日志中的 batch wait 已经降到可忽略范围时，才能说 I/O 瓶颈被消除。本次 CPU-only Dataset 实验没有运行 GPU 模型，因此尚未证明这一终点。

## 临时证据

- 临时数据与脚本：`/storage/penghongen/tmp/stage1_npz_npy_io_20260816_Lt1PEk`
- `valid_benchmark_report.json`：100 个合法 PDB 的转换空间、等价验证和早期连续请求预实验。
- `shuffled_scan_report.json`：非 PDB 分组 workers/prefetch 扫描；因用户叫停 32 workers，文件状态保留为 `benchmarking`，有效结论只使用 8、16、24 workers 的成对结果。
- `volume_cache_probe_report.json`：只关闭四个体数组 LRU 的隔离实验。
- Slurm `343529`：合法集合与转换验证，正常完成。
- Slurm `343531`：非分组参数扫描；32-worker NPZ 子轮由用户叫停，作业取消。
- Slurm `343538`：体数组缓存隔离实验，正常完成。

这些都是一次性临时证据，不属于正式训练入口，也不应被主仓库代码导入。
