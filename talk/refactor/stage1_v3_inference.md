# Stage1 V3 推理重写实施规格与文件清单

本文规定从 `Learn/CUMULATIVE@3cbae636f444fb6630eb505d210d7ce0586b4f76` 开始的 Stage1 V3 推理重写。实现分支为 `codex/stage1-inference-v3`。旧推理代码只通过 Git 读取, 不恢复兼容层。

## 目标与范围

本轮交付一条可由单个 YAML 和单个 shell 入口提交的 Stage1 推理、校准与评估主线:

1. 对 `unet_c1`、`Find_0`、`Find_1`、`Find_2` 生成完整图配体概率。
2. 分别按语义 micro-F1 和 micro-F3 冻结体素阈值, 生成完整 `F1_blobs.npz` 与 `F3_blobs.npz`。
3. 生成轻量 `F1_basic.npz` 和供 Matcher 使用的 `F3_centered.npz`。
4. 基本模式按 $f_1$ 选择概率均值阈值与最小体素数。完整版的 Find 使用 A 原子高斯分数, U-Net 使用退化的概率均值分数, 两者都按 $f_2$ 选择参数。
5. 保存 micro、PDB 等权 macro、逐 PDB 和逐候选评估事实。
6. 在完整图、blobs 和 centered 三个阶段重叠 CPU 与 GPU 工作, 并提供真实 CUDA 利用率基准。

本轮不实现 Selector、CLG、组件森林、Li 阈值、旧七种 Fα centered、Selected、Matcher manifest 或 Matcher 标签生成。

### 2026-08-20 可读性重做边界

本次重做以 `Learn/CUMULATIVE@08f30f28439925b272a11e8e3d1682b5076229ba` 为代码基线，不改变上一轮已经冻结的科学公式、字段、并行顺序或命令范围。重做只允许以下动作：

1. 调整现有函数在模块中的顺序，使冷端复用工具位于唯一分隔线之前，正式主流程位于分隔线之后。
2. 删除只做一次字段转发、路径转发或写后重复读取的薄包装；局部并发回调若直接内联会破坏线程边界，则保留在唯一调用函数内部。
3. 细化公开入口、科学数组和文件读写的 Docstring，逐项写明字段、dtype、shape、坐标系和 offsets 对齐；普通赋值不增加解释性包装。
4. calibration 只保存 checkpoint 的规范化绝对路径。正式代码不计算 checkpoint、配置或代码摘要，不建立 `_valid*` 身份校验层。同一 checkpoint 的不同 F3/F2 目标或其他科学配置由调用者选择不同 `output_root` 版本目录。

本次明确不新建推理子包，不把 `Dataset`、collator、wrapper、blobs 和保存开关搬进新的顶层参数包装函数，也不为目录移动制造转发层。用户对本次历史整理给出显式例外：正式实现先留在 `08f30f2` 工作区，不在审查过程中追加提交；代码、文档、测试和三轮独立审查全部稳定后，再把改动融入并重建原 Stage1 推理历史，不追加“可读性修复”式学习提交。该例外只改变提交时机，不允许丢弃或混入任务开始前的用户修改。

## 代码布局

正式代码只建立一个 `src/inference/` 包, 避免重新形成 `artifacts → component_lineage → inference → selector` 的多层转发。

| 文件 | 唯一主要职责 | 跨模块入口 |
| --- | --- | --- |
| `src/inference/artifacts.py` | 定义正式路径、NPZ/JSON/JSONL 字段组和原子发布 | `Stage1ArtifactPaths`、`load_stage1_npz`、`publish_stage1_json`、`publish_stage1_jsonl`、`publish_stage1_artifact` |
| `src/inference/checkpoint.py` | 从训练 run 快照和 resolved config 恢复完整 wrapper | `load_stage1_wrapper` |
| `src/inference/full_map.py` | 生成窗口、执行异步 CUDA 前向并按固定顺序融合完整概率图 | `FullMapResult`、`window_starts_zyx`、`gaussian_window_weight`、`infer_full_map` |
| `src/inference/blobs.py` | 从一个冻结阈值提取并排序全部 26-连通区域 | `extract_probability_blobs`、`publish_probability_blobs` |
| `src/inference/centered.py` | 解析 80³/48³ 几何，批量重新前向并构造 centered 字段 | `infer_centered_boxes`、`pack_centered_entries` |
| `src/inference/scoring.py` | 计算基本分数与可选 A 原子高斯分数 | `build_gaussian_distance_table`、`sum_gaussian_atom_terms`、`score_centered_candidates` |
| `src/inference/evaluation.py` | 计算逐候选对齐事实与 micro/macro 指标 | `PdbEvaluation`、`load_occurrence_voxels`、`evaluate_centered_pdb`、`aggregate_stage1_metrics` |
| `src/inference/calibration.py` | 冻结 F1/F3 语义阈值和 centered 三阶段参数 | `calibrate_semantic_thresholds`、`tune_centered_selection` |
| `src/inference/pipeline.py` | 执行单 PDB 概率与已冻结 centered 发布事务 | `produce_probability_map`、`produce_centered_role` |
| `src/inference/workflow.py` | 直接编排跨 PDB 有界流水、calibration 和冻结参数运行 | `run_calibration_workflow`、`run_frozen_workflow` |
| `src/inference/cli.py` | 解析 YAML 与 `calibrate`/`run` 子命令 | `main` |

`workflow.py` 不建立 producer callback 工厂。它直接调用单 PDB pipeline、blobs、calibration 和 evaluation 入口。一次字段映射、一次转发和只在一个位置使用的短逻辑全部内联。

### 主代理逐函数自查

下表记录 2026-08-20 可读性重做完成后的本人自查。检查顺序按模块文件名排列, 便于与 AST 清单逐项核对; 面向读者的推荐阅读顺序仍以 `src/inference/README.md` 为准。跨模块入口和独立科学变换即使只有一个生产调用位置也保留, 因为调用者需要单独测试其字段或数值契约; 单次字段映射、一次转发和不承载线程边界的短逻辑已经内联。局部函数只在必须提交给线程池、保持原子发布事务或惰性读取大型文件时保留, 不提升为携带 Dataset、collator、wrapper 和保存开关的顶层包装。

| 顺序 | 类或函数 | 布局与嵌套判断 | 注释与契约自查结果 |
| --- | --- | --- | --- |
| 1 | `Stage1ArtifactPaths` | 保留数据类, 集中一个 PDB 的固定路径命名 | 类字段逐项说明根目录、producer、数据划分和 PDB 标识 |
| 2 | `Stage1ArtifactPaths.__post_init__` | 数据类初始化钩子, 不拆分 | 说明只规范化类型与大小写, 不检查目录存在性 |
| 3 | `Stage1ArtifactPaths.pdb_root` | 多处复用的路径属性 | Docstring 给出精确目录模板 |
| 4 | `Stage1ArtifactPaths.artifact` | 多处复用的五类科学 NPZ 路径入口 | Docstring 点名可接受的五类角色 |
| 5 | `Stage1ArtifactPaths.complete` | 多处复用的完成标记路径入口 | Docstring 给出 `status/<role>/_COMPLETE` 模板 |
| 6 | `load_stage1_npz` | 多处复用的按需字段读取边界, 位于冷端工具区 | 说明 `fields=None` 与显式字段序列的差异及返回映射 |
| 7 | `publish_stage1_json` | 多处复用的 JSON 原子发布边界, 位于冷端工具区 | 说明 UTF-8、末尾换行、同目录临时文件和 `os.replace` |
| 8 | `publish_stage1_jsonl` | 评估 JSONL 的独立正式文件边界 | 说明逐 PDB 顺序、单行 JSON、末尾换行和原子替换 |
| 9 | `publish_stage1_artifact` | NPZ 与可选完成标记组成不可分割发布事务, 位于唯一分隔线后 | 逐项说明输入数组、object dtype 禁止条件、NPZ 替换顺序和 `_COMPLETE` 字段 |
| 10 | `extract_probability_blobs` | 独立 26 邻域科学变换, 可单测且不与发布耦合 | 逐字段说明稀疏坐标、offsets、排序键、BOX 起点和阈值语义 |
| 11 | `publish_probability_blobs` | blobs 的正式发布入口, 直接组合内存概率或显式 None 读取路径 | 说明四个输入、两种概率来源、返回字段与落盘 NPZ 一致 |
| 12 | `CenteredCalibrationFacts` | 校准期间替代大型 centered 数组的小型只读事实类 | 逐字段说明候选轴、原子 offsets、距离截断和非 Find 空表 |
| 13 | `_f_beta_from_counts` | 多处复用的 TP/FP/FN 数值原语, 放在首次调用前 | 说明零分母时返回 0.0 |
| 14 | `_f_beta_from_precision_recall_counts` | 多处复用的异分母 precision/recall 数值原语, 放在首次调用前 | 说明两组计数及零分母语义 |
| 15 | `_gaussian_terms` | 粗搜和细搜共同复用的逐 PDB Gaussian 项 | 说明 PDB 键、tau 单位、正负项顺序与候选轴对齐 |
| 16 | `_scan_actual_score_thresholds` | 来源均值模式的增量阈值搜索, 独立于公开校准入口 | 说明输入事实、实际分数轴、最小体素数、返回目标和并列规则 |
| 17 | `_scan_actual_score_thresholds.augment` | 最大匹配增量更新必须递归访问当前闭包状态, 保留局部嵌套 | 说明 `seen_gt` 掩码、`matched_gt` 原位更新和布尔返回语义 |
| 18 | `_selection_objective` | Gaussian 多阶段搜索复用超过两次的目标函数 | 说明候选分数对齐、两个包含端点的阈值和三项 micro F-beta 返回值 |
| 19 | `calibrate_semantic_thresholds` | 完整语义阈值的正式入口, 位于唯一分隔线后 | 逐字段说明 JSON 摘要和独立扫描 NPZ, 并注释直方图、反向累积和并列选择 |
| 20 | `tune_centered_selection` | centered 三阶段冻结的正式入口, 与语义阈值按业务顺序相邻 | 逐字段说明两种评分模式、三阶段参数、目标函数和返回结构 |
| 21 | `pack_centered_entries` | 线程池与测试共同需要的独立 ragged 数组组装边界 | 逐字段说明共同、稀疏、A/P、48³ 字段及所有 offsets 对齐 |
| 22 | `infer_centered_boxes` | centered 单 PDB 主流程, 位于唯一分隔线后, 不再拆 Dataset/wrapper 包装 | 说明输入开关、三阶段并行、坐标系、BF16 转换和性能返回字段 |
| 23 | `infer_centered_boxes.materialize_batch` | `ThreadPoolExecutor.submit` 必需回调, 保留在 Dataset 使用位置 | 说明来源 blob 编号、80³ 请求、锁页 batch 和顺序保持 |
| 24 | `infer_centered_boxes.arrange_batch` | D2H event 后的单线程整理回调, 闭包直接读取 blobs 与保存开关 | 说明四个输入、候选列表返回、BF16 到 NumPy 和字段条件 |
| 25 | `load_stage1_wrapper` | checkpoint 生命周期的唯一公开入口 | 说明四个显式参数、快照路径临时切换、strict 恢复和两个返回对象 |
| 26 | `main` | 唯一薄 CLI 入口, 不抽参数工厂或身份对象 | 说明两个子命令、版本目录、checkpoint 路径和显式 calibration 来源 |
| 27 | `PdbEvaluation` | 一个 PDB 的候选轴与 occurrence 轴事实类 | 逐字段说明交集、语义、coverage、一对一 offsets 和 top-K 下标 |
| 28 | `load_occurrence_voxels` | schema-v3 ligand_area 文件读取边界 | 逐字段说明 `grid_shape_zyx`、`mask_<id>`、排序后 occurrence 标识和稀疏坐标返回 |
| 29 | `evaluate_centered_pdb` | 单 PDB 科学评估入口, 位于唯一分隔线后 | 说明全部输入、selected 的约束范围、线性体素编号和完整返回事实 |
| 30 | `aggregate_stage1_metrics` | 单 PDB 与全数据划分共同使用的汇总入口 | 逐项说明固定字段、动态阈值字段、micro/macro 分母和空分母语义 |
| 31 | `FullMapResult` | 完整图概率、几何和计时的只读返回类 | 逐字段说明 ZYX 概率形状、世界 XYZ 几何和等待计时 |
| 32 | `window_starts_zyx` | 独立无 padding 几何规则, 可单测 | 说明三轴长度、显式 stride、字典序和末窗贴边规则 |
| 33 | `gaussian_window_weight` | 独立滑窗融合数值变换, 可单测 | 说明规范化坐标、sigma 含义和 float32 三维权重 |
| 34 | `infer_full_map` | 单 PDB 完整图主流程, 位于唯一分隔线后 | 说明全部执行参数、三段并行、确定性累加和 `FullMapResult` |
| 35 | `infer_full_map.materialize_batch` | CPU 物化线程池必需回调, 保留在请求使用位置 | 说明窗口起点轴、锁页 batch 和返回顺序 |
| 36 | `infer_full_map.fuse_batch` | D2H event 后的唯一有序融合回调, 闭包原位更新两个累计图 | 说明概率形状、event、窗口起点对齐和原位副作用 |
| 37 | `produce_probability_map` | 单 PDB 概率正式发布入口 | 说明配置字段、三个科学数组、异步 Future 和四份发布文件 |
| 38 | `produce_probability_map.publish` | publisher 线程池需要的局部原子事务, 不提升为参数搬运函数 | 说明使用外层冻结载荷并依次发布几何、性能、NPZ 和完成标记 |
| 39 | `produce_centered_role` | 单 PDB centered 正式发布入口 | 说明全部输入、blob_limit 事实、selection 两种状态和返回 Future |
| 40 | `produce_centered_role.publish` | 打包、评分、检查和原子发布必须处于同一 publisher 事务 | 说明等待外层 Future、首次写 score/selected、发布顺序和数组返回 |
| 41 | `build_gaussian_distance_table` | 校准与正式评分共同使用的距离表科学变换 | 逐字段说明体素/A offsets、XYZ/ZYX 换轴、5 Å 截断和返回字段 |
| 42 | `sum_gaussian_atom_terms` | 校准与正式评分共同使用的唯一 float64 归约实现 | 说明距离、概率、tau、正负项公式和最终 float32 规范化 |
| 43 | `score_centered_candidates` | centered 候选分数的唯一公开入口 | 说明两种模式、三个 Gaussian 参数、候选轴和最终公式 |
| 44 | `evaluate_and_publish_role` | calibration 与冻结运行共同复用的评估发布入口 | 说明七个输入、三个文件、缺失完成标记失败和返回指标映射 |
| 45 | `run_calibration_workflow` | calibration 跨 PDB 主流程, 不再增加 runner 或工厂层 | 说明十个输入、概率与 centered 队列、最终重新生成和最后完成标记 |
| 46 | `run_calibration_workflow.probability_and_target` | 语义扫描只需一次的惰性生成器, 避免全 calibration 完整图常驻 | 说明逐 PDB 概率/真实并集形状、顺序和内存边界 |
| 47 | `run_calibration_workflow.centered_items` | 每个角色只需一次的按需字段生成器, 避免解压大型特征 | 说明两种评分模式读取字段、PDB 键和排除的 V/P/48³ 数组 |
| 48 | `run_frozen_workflow` | validation/train 跨 PDB 主流程, 不再增加 runner 或身份框架 | 说明十二个输入、checkpoint 路径比较、两段有界队列、五类产物和指标返回 |

## 显式配置

命令行显式提供 `producer`、checkpoint、训练 resolved config、数据划分、PDB 清单、`output_root` 和模型代码来源。`run` 另外显式提供 calibration JSON。`configs/inference/stage1_v3.yaml` 保存以下行为参数, Python 不为这些参数设置默认值:

- 顶层 `device`、`blob_workers`、`publish_workers`、`pending_probability_pdbs` 和 `pending_centered_pdbs`；
- `window.stride_zyx`、`gaussian_sigma`、`batch_size`、`workers`、`prefetch_batches`、`pending_fusion_batches` 和 `precision`；
- `centered_common` 中的 batch、workers、预取深度、CPU 整理队列和精度；
- `calibration.semantic_denominator` 和 Gaussian 第二阶段乘数；
- `evaluation.coverage_thresholds` 与 `topk_values`；
- `roles.F1_basic/F3_centered` 中的角色名、`min_voxel_values`、`objective_beta`、producer 到 `score_mode` 的映射、Gaussian 粗网格、`blob_limit` 和 centered 保存开关。

正式配置显式写 `stride_zyx=[50,50,50]`、`gaussian_sigma=0.5`。同一 workflow 依次运行 F1 basic 与 F3 centered：F1 角色关闭 `save_voxel_final` 和 `save_dense48`, F3 角色打开两项保存开关。

## 执行流水线

### 完整图

CPU 线程提前物化固定顺序的 80³ 请求并组 batch。主线程把页锁定 batch 非阻塞传到 GPU, 执行 `wrapper.forward_voxel_probability()` 和 sigmoid, 再把 float32 概率异步复制到专用页锁定 CPU buffer。唯一融合线程等待 CUDA event 后按窗口提交顺序执行 float32 Gaussian 累加。

完整图概率不乘 hardmask。窗口沿每轴从 0 开始, 最后一个窗口强制以 `length-80` 结束, 因而无 padding 且覆盖边界。

### 连通区域

完整图离开 GPU 后，概率 NPZ 压缩与 F1/F3 连通区域提取同时进入 CPU 线程池；blobs 直接读取同一个内存数组，GPU 主线程立即开始下一个 PDB。每个阈值下的全部区域均落盘；排序键为平均来源概率降序，再按区域最小完整图 C-order 线性索引升序。

`fits_centered_box` 只说明区域能否完整放入 80³ BOX。它不删除区域。`min_voxels` 也不在本阶段生效。

### centered

F1 使用 voxel-only centered 前向, 保存来源概率、重算概率和共同几何, 不保存 A/P、`voxel_final` 或 48³ 数组。

F3 使用完整 wrapper 前向。所有模型保存 `voxel_final` 和三张 V-centered 48³ 数组。Find 额外保存 A/P 表; U-Net 不出现 A/P 字段。

CPU 线程提前物化 centered batch。主线程执行 H2D 与 GPU 前向。完成的 batch 非阻塞复制到 CPU 后交给整理线程。一个 PDB 的字段齐备后，发布线程先写入冻结 score/selected，再执行第一次正式 NPZ 压缩，同时 GPU 开始下一个 PDB。calibration 搜索用临时候选不建立 `_COMPLETE`；最终 `min_voxels` 冻结后重新生成正式候选集合并发布完成标记。

## 科学公式

F1/F3 体素阈值分别最大化 calibration 全体 PDB 的语义 micro-F1 与 micro-F3。阈值扫描使用概率直方图精确累计 TP、FP、FN。

基本分数为:

$$
S_j=\operatorname{source\_probability\_mean}_j
$$

Find 完整分数为:

$$
S_j=\operatorname{source\_probability\_mean}_j+\lambda_+G_j^+-\lambda_-G_j^-
$$

$$
G_j^+=\sum_i w_{ji}p_i,\qquad G_j^-=\sum_i w_{ji}(1-p_i)
$$

$$
w_{ji}=\exp\left(-\frac{d_{ji}^2}{2\tau^2}\right)\mathbf{1}[d_{ji}\le r]
$$

候选编号 $j$ 索引当前 centered 文件的候选轴。原子编号 $i$ 索引候选 $j$ 的 A 原子区间。$p_i$ 是 `A_probability`，$d_{ji}$ 是 A 原子到候选来源 V 体素中心的最近世界距离，单位 Å。$\lambda_+$ 与 $\lambda_-$ 分别是 Gaussian 正项和负项系数，$\tau$ 是 Gaussian 距离标准差，单位 Å。指示函数 $\mathbf{1}[d_{ji}\le r]$ 在条件成立时取 1，否则取 0；截断半径固定为 $r=5\ \mathrm{Å}$。正负项不按原子数归一化。U-Net 没有 A 表，两项均为 0。

Find 完整模式严格分三阶段：第一阶段扫描 500 组粗网格；第二阶段固定第一阶段 tau，扫描两个 lambda 的 5×5 乘数和分数下限的 15 个乘数，共 375 组；第三阶段冻结 Gaussian 参数，只扫描 `min_voxels=8..40`。基本模式和 U-Net 完整模式先按实际出现的来源分数冻结阈值，再单独扫描相同的最小体素数范围。

## 发布与恢复

- NPZ 和 JSON 先写同目录临时文件, 写入成功后使用 `os.replace` 发布; 大型 NPZ 不做重复解压重读。
- `_COMPLETE` 只在最终文件已经原子替换后建立。
- calibration 搜索可写带占位 `score/selected` 的临时候选 centered, 但不写 `_COMPLETE`。参数冻结后按最终 `min_voxels` 重新生成候选集合, 在第一次正式压缩前写入冻结值, 再发布完成标记。
- F3 的 eligible 候选数严格大于显式 `blob_limit` 时写 `status/F3_centered/_BLOB_EXCEED`, 并始终继续发布 centered; 不建立跳过 PDB 的特殊终态。
- 不建立隐藏 work 文件、不保留旧版本备份, 恢复依靠完整图、blobs 和 Git。

## 本轮逐文件状态表

### `08f30f2` 可读性重做差异

下表是本次重做的当前权威范围。所有代码文件均保留既有科学生命周期；动作只涉及函数顺序、薄包装删除、注释或直接字段写入，不新建兼容层。

| 路径 | 职责与调用者 | 当前动作 | 验证 |
| --- | --- | --- | --- |
| `src/inference/artifacts.py` | PDB 路径与原子发布；全推理主线调用 | 删除路径薄包装和写后重复读取；按依赖顺序排列发布工具 | 原子发布测试 |
| `src/inference/blobs.py` | 26 邻域区域；workflow 调用 | 把内存概率/落盘概率选择改为显式参数；补 offsets 契约 | blob 排序测试 |
| `src/inference/calibration.py` | 语义与 centered 参数冻结；workflow 调用 | 冷端数值工具移到唯一分隔线前；补小型事实字段说明 | calibration 测试 |
| `src/inference/centered.py` | centered 前向与字段打包；pipeline 调用 | 删除递归 host-copy 和内部 array 包装；保留两项线程回调 | centered/BF16/CUDA 测试 |
| `src/inference/checkpoint.py` | wrapper 恢复；CLI 调用 | 只修正文档标点 | CLI smoke |
| `src/inference/cli.py` | 命令解析与依赖构造；shell 调用 | 删除 SHA 函数与三个摘要；训练配置 `_target_` 显式读取 | CLI/Dataset 测试 |
| `src/inference/evaluation.py` | 逐 PDB 事实与跨 PDB 指标；workflow/calibration 调用 | 补全字段、选择范围、动态指标和形状说明 | 指标测试 |
| `src/inference/full_map.py` | 完整图滑窗和融合；pipeline 调用 | 补全坐标、线程和返回契约，不新增函数 | 窗口/CUDA 测试 |
| `src/inference/pipeline.py` | 单 PDB 发布事务；workflow 调用 | 内联唯一一次 `score/selected` 映射；补真实发布顺序 | 发布事务测试 |
| `src/inference/scoring.py` | source mean/Find Gaussian 分数；pipeline/calibration 调用 | 删除单调用选择包装；调整工具与主入口顺序 | Gaussian 同源测试 |
| `src/inference/workflow.py` | calibration/run 跨 PDB 编排；CLI 调用 | 身份对象缩减为 checkpoint 路径；调用显式传入概率来源 | workflow 测试 |
| `tests/inference/test_stage1_v3.py` | CPU 科学与执行契约 | 适配 checkpoint 路径；删除已被发布事务覆盖的包装函数单测 | 与 Dataset 测试合并运行 |
| `src/inference/README.md` | 产物与模块入口 | 展开逐字段表、版本目录和 calibration 显式来源 | 与代码/BOX 契约对照 |
| `configs/inference/README.md` | YAML 字段入口 | 删除摘要绑定表述；说明版本目录 | OmegaConf 解析 |
| `训练与运行/sh/infer/README.md` | 唯一 shell 入口 | 说明 checkpoint 路径、版本目录和显式 calibration | Bash 语法 |
| `tests/inference/README.md` | 测试范围 | 更新 checkpoint 路径口径 | 测试命令 |
| `ops/stage1_inference_benchmark/README.md` | GPU 利用率基准产物契约 | 展开 `summary.json` 的逐字段类型、单位与空分母语义 | 与基准脚本字段对照 |
| `talk/global/NOTE_LIST.md` | 已冻结科学决定 | 删除 `enforce_blob_limit` 旧终止语义 | 与 YAML 对照 |
| `talk/refactor/stage1_v3_inference.md` | 本次规格、逐文件范围和审查状态 | 记录用户指定的工作区提交时机例外、函数授权与三轮核查结论 | 与 Git 状态和审查报告对照 |
| `talk/stage1_v3_inference_inputs_outputs.md` | 输入输出学习索引 | 修正 48³ 偏移方向并补 offsets 对齐 | 与 BOX 契约对照 |
| `talk/stage1_v3_inference_modules.md` | 模块依赖与并行边界学习索引 | 修正评分发生在 centered 发布事务中的调用关系 | 与当前 import 和调用图对照 |
| `C:/Users/15919/Desktop/AdaLigand/文档/规划文档/BOX-level数据契约.md` | 盘上字段权威 | checkpoint 路径、显式版本目录、评估字段和 offsets 收口 | 与代码逐字段对照 |
| `C:/Users/15919/Desktop/AdaLigand/文档/exec_plan/Stage1_V3推理重写实施记录.md` | 关键事件与验证证据 | 记录本次减法边界和重新审查状态 | 关键节点更新 |
| `C:/Users/15919/Desktop/AdaLigand/文档/mapping/计划执行映射.md` | 计划/实现索引 | 记录主代理自查、三轮独立审查和双线收口状态 | 收口时更新结论 |

### 可读性重做审查结论

主代理在独立审查前逐名检查 48 个类、方法、顶层函数和局部回调，并在本文件记录每个对象的保留理由、布局与注释契约。代码维持 9 个必要局部回调，分别服务于线程池事务、惰性读取或递归增广；没有新增 Dataset、collator、wrapper 或身份包装层。

三类独立审查均完成第 3/3 轮全面核查，之后只复核已经报告的问题。布局/Git/函数审查批准跨模块入口表、阅读顺序、函数定义顺序和嵌套关系；科学逻辑审查批准 `save_dense48=false` 时不解压完整概率图的 I/O 修正；注释与文档审查在逐字段补齐 offsets、Gaussian 参数、评估指标和 benchmark 对照字段后批准。实际测试命令与通过数量记录在 AdaLigand `文档/exec_plan/Stage1_V3推理重写实施记录.md`。

### 首次重写历史范围

以下旧表以 `3cbae636f444fb6630eb505d210d7ce0586b4f76` 为共同基点，只记录首次 Stage1 V3 推理重写时的新增、删除和迁移动作，不代表 `08f30f2` 可读性重做的当前动作。

### 正式代码、配置、脚本和文档

| 路径 | 职责与调用者 | 生命周期 | 验证 | 动作与理由 |
| --- | --- | --- | --- | --- |
| `CLAUDE.md` | 仓库导航入口 | 当前文档 | 冷读 | 更新；删除 Selector/forest/CLG 旧导航 |
| `C:/Users/15919/Desktop/AdaLigand/文档/规划文档/BOX-level数据契约.md` | Stage1 V3 盘上字段唯一权威 | 当前契约 | 与代码逐字段对照 | 重写推理章节；删除旧 forest/CLG/Selector 契约 |
| `C:/Users/15919/Desktop/AdaLigand/文档/exec_plan/Stage1_V3推理重写实施记录.md` | 记录关键实现、验证和 Git 事件 | 当前执行记录 | 冷读 | 新建；不记录逐命令流水账 |
| `C:/Users/15919/Desktop/AdaLigand/文档/mapping/计划执行映射.md` | 连接规格、契约与执行记录 | 当前映射 | 路径对照 | 更新；把旧多阈值计划标为归档 |
| `CLAUDE/memory/handoffs/2026-08-20-stage1-v3-inference-rewrite-complete.md` | 保存下一任务可直接恢复的端点、决定与开放问题 | 关键节点 handoff | 冷读 | 新建；不记录逐命令流水 |
| `CLAUDE/memory/index.json` | 指向当前项目记忆更新时间 | 项目记忆索引 | JSON 解析 | 更新至 2026-08-20 |
| `CLAUDE/memory/projects/pocket-plus.json` | 保存 Pocket Plus 项目级进展入口 | 项目记忆状态 | JSON 解析 | 追加本轮推理收口事件 |
| `configs/inference/README.md` | 解释 `stage1_v3.yaml` 的读取链和字段 | 正式新增 | 冷读 | 新建；避免配置语义依赖对话 |
| `configs/inference/stage1_v3.yaml` | 四个 producer 的科学与并行参数；CLI 读取 | 正式新增 | OmegaConf 加载 | 新建；唯一共享配置 |
| `ops/stage1_inference_benchmark/README.md` | 说明真实 GPU 采样工具 | 正式新增 | 冷读 | 新建；性能事实与科学代码隔离 |
| `ops/stage1_inference_benchmark/benchmark_gpu_utilization.py` | 启动真实命令并采样 `nvidia-smi` | 正式运维 | 本机 RTX/命令 smoke | 新建；验证 GPU/CPU 重叠 |
| `src/datasets/stage1_dataset.py` | V3 NPY/mmap 80³ 物化；推理线程调用 | 保留修改 | `test_stage1_dataset.py` | 为共享 mmap LRU 增加短临界区锁；保证缓存状态一致，命中后复用；首次并发 miss 允许重复映射 |
| `src/inference/README.md` | 推理包字段、校准和并行权威 | 正式新增 | 冷读 | 新建；提供冷读入口 |
| `src/inference/__init__.py` | 声明推理包 | 正式新增 | import smoke | 新建；不导出包装层 |
| `src/inference/artifacts.py` | 路径、NPZ/JSON/JSONL 和完成标记 | 正式新增 | `test_stage1_v3.py` | 新建；统一原子发布 |
| `src/inference/blobs.py` | 26 邻域 blobs；workflow 调用 | 正式新增 | `test_stage1_v3.py` | 新建；保留全部区域 |
| `src/inference/calibration.py` | 语义与 centered 参数冻结；workflow 调用 | 正式新增 | `test_stage1_v3.py`、规模检查 | 新建；替代旧 Gauss 调参 |
| `src/inference/centered.py` | centered GPU 前向与字段打包；pipeline 调用 | 正式新增 | `test_stage1_v3.py`、CUDA smoke | 新建；统一 F1/F3 几何 |
| `src/inference/checkpoint.py` | 恢复训练快照与 wrapper；CLI 调用 | 正式新增 | checkpoint smoke | 新建；绑定训练来源 |
| `src/inference/cli.py` | `calibrate`/`run` 子命令和身份；shell 调用 | 正式新增 | `--help`、命令 smoke | 新建；保持薄入口 |
| `src/inference/evaluation.py` | 逐候选事实和汇总指标；workflow/calibration 调用 | 正式新增 | `test_stage1_v3.py` | 新建；保存可复算事实 |
| `src/inference/full_map.py` | 80³ 滑窗与融合；pipeline 调用 | 正式新增 | `test_stage1_v3.py`、CUDA smoke | 新建；无 padding 且有序融合 |
| `src/inference/pipeline.py` | 单 PDB 发布事务；workflow 调用 | 正式新增 | `test_stage1_v3.py` | 新建；不建立 runner 类 |
| `src/inference/scoring.py` | 来源均值与 Find Gaussian；calibration/evaluation 调用 | 正式新增 | `test_stage1_v3.py` | 新建；统一正式公式 |
| `src/inference/workflow.py` | 跨 PDB 有界流水；CLI 调用 | 正式新增 | 端到端 smoke | 新建；实现 GPU/CPU 重叠 |
| `src/wrappers/voxel_point_stage1.py` | Stage1 wrapper 与 voxel-only 前向 | 保留修改 | import/训练回归 | 修正文档；删除已经退出的 Find hardmask 推理描述 |
| `src/wrappers/README.md` | wrapper 组织、阅读顺序与 V3 推理接口 | 正式新增 | 冷读 | 新建；使目录职责自包含 |
| `talk/global/NOTE_LIST.md` | 汇总用户决定和 Matcher 边界 | 当前文档 | 冷读对照 | 更新；清除旧 hardmask/角色语义 |
| `talk/refactor/stage1_v3_inference.md` | 本实施规格和逐文件记录 | 当前规划 | 路径覆盖审计 | 更新；持续回填实现差异 |
| `talk/stage1_v3_inference_inputs_outputs.md` | 输入与字段冷读概览 | 当前文档 | 与 README/契约对照 | 新建；解释每类数组 |
| `talk/stage1_v3_inference_modules.md` | 模块和依赖方向概览 | 当前文档 | 与 import 图对照 | 新建；解释调用关系 |
| `talk/代码习惯.md` | 长期代码表达习惯 | 当前文档 | 冷读 | 更新；用 F1/F3 条件字段替换已退出角色示例 |
| `tests/datasets/test_stage1_dataset.py` | Dataset mmap、裁块和共享缓存并发 | 正式测试 | pytest | 修改；覆盖加锁 LRU 的计数一致性 |
| `tests/datasets/README.md` | Dataset 测试范围、字段与运行边界 | 正式新增 | 冷读 | 新建；使测试目录自包含 |
| `tests/inference/README.md` | 推理测试目录说明 | 正式新增 | 冷读 | 新建；满足测试目录入口要求 |
| `tests/inference/test_stage1_cuda.py` | CUDA H2D/前向/D2H/融合 smoke | 正式测试 | 本机 RTX pytest | 新建；覆盖唯一 GPU owner 执行链 |
| `tests/inference/test_stage1_v3.py` | 几何、字段、评分、指标和发布 | 正式测试 | pytest | 新建；替代旧专项测试 |
| `训练与运行/sh/infer/README.md` | 正式命令、路径和提交方式 | 当前文档 | 冷读 | 重写；只说明单一入口 |
| `训练与运行/sh/infer/stage1_v3.sh` | 激活环境并调用 CLI；submit_task 调用 | 正式新增 | `bash -n` | 新建；不操作任何资源锁 |

### 退出的 Selector 与旧 Gauss 运维

| 路径 | 原职责与调用者 | 生命周期 | 验证 | 动作与理由 |
| --- | --- | --- | --- | --- |
| `configs/selector/Find_0.yaml` | 旧 Find_0 Selector 配置 | 退出 | Git 历史 | 删除；Selector 不在 Stage1 V3 |
| `configs/selector/Find_1.yaml` | 旧 Find_1 Selector 配置 | 退出 | Git 历史 | 删除；同上 |
| `configs/selector/Find_2.yaml` | 旧 Find_2 Selector 配置 | 退出 | Git 历史 | 删除；同上 |
| `configs/selector/unet_c1.yaml` | 旧 U-Net Selector 配置 | 退出 | Git 历史 | 删除；同上 |
| `ops/Gauss_Scorer/build_refinement_grid_find0.sh` | 旧细网格生成 | 退出 | Git 历史 | 删除；三阶段搜索进入正式 calibration |
| `ops/Gauss_Scorer/evaluate_find0_calibration.sh` | 旧数组调参任务 | 退出 | Git 历史 | 删除；不保留并行旧入口 |
| `ops/Gauss_Scorer/grid_find0_calibration.json` | 旧 Find 粗网格 | 退出 | Git 历史 | 删除；参数迁入共享 YAML |
| `ops/Gauss_Scorer/merge_find0_calibration.sh` | 旧调参合并 | 退出 | Git 历史 | 删除；workflow 直接冻结 |
| `ops/Gauss_Scorer/tune.py` | 旧 forest/Fα/Li scorer | 退出 | Git 历史 | 删除；避免双重科学实现 |

### 退出的旧推理 shell

| 路径 | 原职责 | 生命周期 | 验证 | 动作与理由 |
| --- | --- | --- | --- | --- |
| `训练与运行/sh/infer/Find_0_Falpha.sh` | 七种 Fα centered | 退出 | Git 历史 | 删除；只保留 F1 basic/F3 centered |
| `训练与运行/sh/infer/Find_0_Gauss.sh` | forest Gauss 回填 | 退出 | Git 历史 | 删除；Gaussian 进入正式评分 |
| `训练与运行/sh/infer/Find_0_Li.sh` | Li 角色 | 退出 | Git 历史 | 删除；不在本轮范围 |
| `训练与运行/sh/infer/Find_0_calibration_CLG.sh` | calibration CLG | 退出 | Git 历史 | 删除；CLG 退出 |
| `训练与运行/sh/infer/Find_0_calibration_F1.sh` | calibration F1 | 退出 | Git 历史 | 删除；统一入口 |
| `训练与运行/sh/infer/Find_0_calibration_probability.sh` | calibration 概率 | 退出 | Git 历史 | 删除；统一入口 |
| `训练与运行/sh/infer/Find_0_freeze_thresholds.sh` | 冻结七阈值 | 退出 | Git 历史 | 删除；新 calibration 只冻结 F1/F3 |
| `训练与运行/sh/infer/Find_0_train_CLG.sh` | train CLG | 退出 | Git 历史 | 删除；CLG 退出 |
| `训练与运行/sh/infer/Find_0_train_F1.sh` | train F1 | 退出 | Git 历史 | 删除；统一入口 |
| `训练与运行/sh/infer/Find_0_train_F1_shard_01.sh` | 一次性固定分片 | 退出 | Git 历史 | 删除；提交系统负责分片 |
| `训练与运行/sh/infer/Find_0_validation_CLG.sh` | validation CLG | 退出 | Git 历史 | 删除；CLG 退出 |
| `训练与运行/sh/infer/Find_0_validation_F1.sh` | validation F1 | 退出 | Git 历史 | 删除；统一入口 |
| `训练与运行/sh/infer/prepare_inference_pdb_lists.sh` | 旧清单转换 | 退出 | Git 历史 | 删除；V3 清单显式传入 |

### 退出的旧测试

| 路径 | 原覆盖 | 生命周期 | 替代验证 | 动作与理由 |
| --- | --- | --- | --- | --- |
| `tests/artifacts/test_stage1_io.py` | 旧 artifacts IO | 退出 | `tests/inference/test_stage1_v3.py` | 删除；旧模块已移除 |
| `tests/artifacts/test_stage1_states.py` | 旧租约与状态 | 退出 | 新原子发布测试 | 删除；不保留旧状态机 |
| `tests/component_lineage/test_stage1_clg.py` | CLG | 退出 | 无 | 删除；科学角色退出 |
| `tests/component_lineage/test_stage1_structures.py` | forest 结构 | 退出 | 无 | 删除；科学角色退出 |
| `tests/evaluation/test_stage1_calibration.py` | 旧七阈值校准 | 退出 | 新 calibration 测试 | 删除；目标已变化 |
| `tests/evaluation/test_stage1_instance_metrics.py` | 旧实例指标 | 退出 | 新 evaluation 测试 | 删除；事实字段重写 |
| `tests/inference/test_gauss_scorer.py` | forest Gauss | 退出 | 新 scoring 测试 | 删除；评分位置改变 |
| `tests/inference/test_gauss_scorer_tuning.py` | 旧两阶段调参 | 退出 | 新三阶段 calibration 测试 | 删除；加入 min 第三阶段 |
| `tests/inference/test_li_centered.py` | Li centered | 退出 | 无 | 删除；Li 退出 |
| `tests/inference/test_stage1_assembly_cli.py` | 旧多命令 CLI | 退出 | 新 CLI smoke | 删除；入口重写 |
| `tests/inference/test_stage1_centered.py` | 旧 centered schema | 退出 | `test_stage1_v3.py` | 删除；字段契约重写 |
| `tests/inference/test_stage1_checkpoint.py` | 旧 checkpoint 路径 | 退出 | 真实 checkpoint smoke | 删除；恢复逻辑简化 |
| `tests/inference/test_stage1_full_map.py` | 旧 stride/hardmask | 退出 | `test_stage1_v3.py` | 删除；科学规则变化 |
| `tests/selector/test_antichain_dp.py` | Selector antichain | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_ccln.py` | Selector CCLN | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_dataset_and_freeze.py` | Selector 数据与冻结 | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_density_munet_lite.py` | Selector 密度模型 | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_inference_schema.py` | Selector schema | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_input_fusion.py` | Selector 输入融合 | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_selector_calibration.py` | Selector 校准 | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_selector_configs.py` | Selector 配置 | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_train_sampler.py` | Selector 训练采样 | 退出 | 无 | 删除；Selector 退出 |
| `tests/selector/test_wrapper_losses.py` | Selector wrapper loss | 退出 | 无 | 删除；Selector 退出 |

### 退出的旧学习文档

| 路径 | 原职责 | 生命周期 | 验证 | 动作与理由 |
| --- | --- | --- | --- | --- |
| `talk/stage1_plus的学习注释.md` | 解释 forest、CLG、Selector 与旧角色 | 退出 | Git 历史 | 删除；内容与 V3 主线整体冲突，历史由 Git 提供 |

## 验收标准

1. CPU 回归必须覆盖窗口边界、float32 blob 稳定排序、偏斜 blob BOX、BF16 转换、centered 字段、Python 3.10 CLI、Dataset 直接构造、calibration 完成标记、Gaussian 三阶段与数值同源、micro/macro、top-K、按需字段读取和原子发布。
2. CUDA smoke 必须覆盖真实 Conv3d 的异步 H2D、GPU 前向、D2H 和有序 CPU 融合执行链。
3. Dataset 测试必须覆盖 V3 NPY/mmap、实际 80³ 数值边界与推理线程共享 LRU 的计数一致性。
4. 真实 checkpoint smoke 至少覆盖 `unet_c1` 与一个 Find；得到用户对服务器运行的授权前不得自行提交任务，也不得通过删除科学字段规避显存问题。
5. GPU 基准必须使用真实 checkpoint 记录 BOX/s、PDB/s、GPU 活跃比例和队列等待；合成 CUDA 运行只验收采样工具。
6. 冻结工作区端点后，布局/Git/函数、注释/Docstring、科学逻辑三类独立审查各执行三轮全面核查，之后只复核已经报告的问题。实际命令、通过数量、GPU 型号和审查结论写入 AdaLigand `文档/exec_plan/Stage1_V3推理重写实施记录.md`。

## Git 双线收口

双线收口必须从共同基点 `3cbae636f444fb6630eb505d210d7ce0586b4f76` 按以下顺序重建：

1. 集中删除全部旧非测试文件。
2. 提交当前规格、NOTE、README 与两份学习概览。
3. 按产物契约、完整图与 blobs、centered、评分评估、编排配置的依赖顺序提交正式代码。
4. 最后集中提交全部测试文件和旧测试删除。
5. 在实现端点与学习端点运行相同测试, 比较 Python 可执行语句、配置、字段级产物和 Git tree。
6. 等价通过后将 `Learn/CUMULATIVE` 快进到学习端点。实现分支长期保留, 不推送远端。

上一轮累计学习端点是 `Learn/CUMULATIVE@08f30f28439925b272a11e8e3d1682b5076229ba`。本次先在该端点工作区完成减法重构和验证；最终实现端点、重建后的学习端点与等价核验结果只在三轮审查完成后回填，不预写提交号。
