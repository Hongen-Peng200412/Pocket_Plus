# Stage1 V3 通用 F-alpha 推理实施规格与文件清单

本文规定从 `Learn/CUMULATIVE@916dced93b5c0f19418ddba0077f5504faa1ffd7` 开始的 Stage1 V3 推理接口重做。实现分支是 `codex/stage1-inference-falpha`。旧 `calibrate/run`、固定 F1/F3 角色和旧推理文件只通过 Git 历史读取，不建立兼容入口。

## 目标

本轮把完整推理拆成五个可以独立申请 CPU/GPU 资源并一键提交的阶段：

1. `probability`：从 checkpoint 生成完整图配体概率。
2. `blobs`：拟合或读取一个正浮点 alpha 的语义阈值，生成 `F{alpha}_blobs.npz`。
3. `centered`：从 blobs 生成 `F{alpha}_centered.npz`，或用 score-only 只更新选择字段。
4. `tune`：从 blobs 调整 basic 参数，或从 centered 调整 Gaussian 参数。
5. `evaluate`：独立评估 blobs 或 centered。

五个阶段共用 `训练与运行/sh/infer/stage1_v3.sh` 和 `configs/inference/stage1_v3.yaml`。本轮不改 Dataset、模型、训练脚本或 Matcher，也不提交服务器正式推理任务。

## 冻结契约

### F-alpha 与文件名

- `alpha` 是正浮点数，配置推荐值为 2.0。
- 路径标签使用 Python float 的最短可往返十进制；整数写成 `F2`，小数点改成 `p`，例如 `F0p5` 与 `F1p5`。
- 每个 alpha 只有 `F{alpha}_blobs.npz` 与 `F{alpha}_centered.npz` 两类 PDB 候选产物。
- `objective_beta` 独立控制 tune 的三项 F-beta 目标，配置推荐值为 2.0。

### producer 与 centered 字段

- producer 由命令显式提供，CLI 不维护 `unet_c1/Find_0/...` 白名单。
- 所有 producer 都执行完整 forward，并保存 centered 共同字段与 `voxel_final`。
- `unet_*` 不保存 A/P、辅助受体体素或 48³ 稠密数组。
- `Find_*` 另外保存 auxiliary、A/P、实验密度 48³、模拟密度 48³ 和完整图概率 48³。
- alpha 不决定模型前向模式或字段集合。

### 前向与选择

- `forward_min_voxels` 由正常 centered 命令显式提供，只控制哪些 `fits_centered_box=true` 来源 blob 进入 GPU。
- `prefiltered_min_voxel` 由 tune 命令显式提供，在全部参数尝试前固定；小于该值的候选始终未入选。
- 选择 JSON 分别保存固定 `prefiltered_min_voxel` 和搜索所得 `min_voxels`；两者共同控制 `selected`，互不限制，也不得改变 centered 候选轴和 offsets。
- 未提供选择参数的 centered 不含 `score/selected`。
- score-only 只能增加或替换 `score/selected`，其他数组逐元素保持。
- Matcher 可以忽略 `selected`，但必须保持 `centered_box_index`、`source_blob_index` 和 offsets 对齐。

### basic 与 Gaussian

- basic 分数是 `source_probability_mean`；basic tune 的事实包含全部 blobs，包括 `fits_centered_box=false` 的区域，再应用统一预过滤门槛。
- Gaussian 分数使用来源平均概率与 5 Å 内 A 原子正负项；只有 Find centered 具备所需 A 表。
- Gaussian tune 在统一预过滤后保留粗搜索、固定 tau 的细搜索和最终最小体素数三个阶段。
- tune 分别写 `F{alpha}_basic.json` 与 `F{alpha}_gaussian.json`。

### 语义阈值

- blobs 阈值来源是 `--fit-semantic`、显式浮点值或已有 `F{alpha}_semantic.json` 三者之一。
- 语义拟合使用 calibration 全集 micro F-alpha，写 `F{alpha}_semantic.json` 与 `F{alpha}_semantic_scan.npz`。
- 四种 calibration 文件相互独立，不保存 checkpoint、配置、代码摘要或哈希。

### 最小 `_BLOB_EXCEED`

- centered 读取来源 blobs 后统计 `blob_index` 长度。
- 长度严格大于全局常量 1000 时，立即写 `status/F{alpha}_centered/_BLOB_EXCEED` 并跳过当前 PDB。
- 标记只含 `pdb_id`、`centered_role`、`source_blob_count` 和 `limit`。
- 不增加覆盖、恢复、自动删除、重试或额外完成状态。
- tune/evaluate 只在原本读取清单时识别该原因并输出说明。

### 分片、复用与覆盖

- PDB 清单是顶层字符串列表 JSON。
- probability、显式阈值 blobs 和 centered 可用固定 seed 3407 打乱清单，再按 `[shard_index::shard_count]` 取 0-based 分片。
- 语义拟合、tune 与 evaluate 不分片。
- 同一 `output_root` 可以复用 probability 并逐次增加多个 alpha。
- probability、blobs、centered 的 `_COMPLETE` 默认只跳过同一阶段；`--overwrite` 只重跑当前阶段。
- 正式代码不计算哈希，不建立 `_valid*` 或 checkpoint/config/model 身份比较。
- 同一 checkpoint 的 F3 最优、F2 最优或其他科学配置由调用者使用可读目录名称区分。

### 数值与并行

- 完整图配体概率不乘受体 hardmask。
- `stride_zyx` 必须由配置显式传入，Python 不设默认值；当前配置为 `[30,30,30]`。
- probability 内部重叠 CPU 请求物化、GPU 前向、异步 D2H 与有序融合。
- centered 内部重叠 CPU 请求物化、完整 GPU forward、异步 D2H 与 CPU 字段整理。
- probability 与 centered 的 NPZ 压缩分别和下一个 PDB 的 GPU 前向重叠。
- 概率、几何、原始密度和 `A_feat_L0` 使用 float32；学习特征使用 float16。

## 代码布局

正式代码只保留 `src/inference/` 一个包，不建立新子包或顶层运行时参数对象。

| 文件 | 唯一主要职责 | 正式跨模块入口 |
| --- | --- | --- |
| `artifacts.py` | F-alpha 标签、产物路径、NPZ/JSON/JSONL 原子发布 | `f_alpha_tag`、`Stage1ArtifactPaths`、三个发布函数、`load_stage1_npz` |
| `checkpoint.py` | 从训练 run 恢复 wrapper | `load_stage1_wrapper` |
| `full_map.py` | 80³ 滑窗与完整图概率融合 | `FullMapResult`、`window_starts_zyx`、`gaussian_window_weight`、`infer_full_map` |
| `blobs.py` | 单阈值 26 邻域连通区域 | `extract_probability_blobs` |
| `centered.py` | centered 完整前向与 ragged 字段组装 | `infer_centered_boxes`、`pack_centered_entries` |
| `scoring.py` | basic 与 Gaussian 候选分数 | `build_gaussian_distance_table`、`sum_gaussian_atom_terms`、`score_centered_candidates` |
| `evaluation.py` | 逐 PDB 事实和跨 PDB 指标 | `PdbEvaluation`、`load_occurrence_voxels`、`evaluate_centered_pdb`、`aggregate_stage1_metrics` |
| `calibration.py` | 单 alpha 语义阈值与选择参数搜索 | `calibrate_semantic_thresholds`、`tune_centered_selection` |
| `pipeline.py` | 五阶段跨 PDB 编排与发布 | 五个 `run_*_stage` |
| `cli.py` | 参数、JSON 清单、固定分片与 Dataset/wrapper 装配 | `main` |

`workflow.py` 删除。局部嵌套只保留确实需要共享当前批次或发布事务的线程回调：完整图物化与融合、centered 物化与整理，以及 pipeline 的异步发布。一次字段映射、一次路径转发和只赋值一次的逻辑直接内联。

## 文件映射

| 当前文件 | 当前职责与调用者 | 生命周期与现有测试 | 目标动作 | 理由 |
| --- | --- | --- | --- | --- |
| `src/inference/artifacts.py` | 固定 F1/F3 路径与发布；全部阶段调用 | 正式代码；产物测试 | 保留并改成动态 F-alpha | 发布逻辑可复用，固定角色必须退出 |
| `src/inference/checkpoint.py` | wrapper 恢复；CLI 调用 | 正式代码；CLI 构造测试 | 保留 | 快照导入是独立生命周期边界 |
| `src/inference/full_map.py` | 完整图 GPU/CPU 流水；pipeline 调用 | 正式代码；几何与 CUDA smoke | 保留科学逻辑 | 已验证的数值与顺序不变 |
| `src/inference/blobs.py` | 提取并发布固定角色 | 正式代码；连通区域测试 | 保留提取，发布并入 pipeline 回调 | 单次发布包装没有独立语义 |
| `src/inference/centered.py` | voxel-only/full 与多保存开关 | 正式代码；offsets、BF16 测试 | 统一完整 forward，producer 决定字段 | 减少角色分支和参数包装 |
| `src/inference/scoring.py` | 来源均值与 Find Gaussian | 正式代码；数值同源测试 | 模式名改为 basic/Gaussian | 评分定义与 alpha 解耦 |
| `src/inference/evaluation.py` | 实例事实与汇总 | 正式代码；指标测试 | 保留公式 | 命令拆分不改变科学指标 |
| `src/inference/calibration.py` | 同时冻结固定 F1/F3 | 正式代码；阈值与三阶段测试 | 单 alpha 语义与通用选择 | 独立 CPU 阶段不保留角色嵌套 |
| `src/inference/pipeline.py` | 两个单 PDB 发布事务 | 正式代码；发布测试 | 重写为五阶段入口 | 删除中间 workflow 跳转 |
| `src/inference/workflow.py` | calibrate/run 编排 | 正式代码；workflow 测试 | 删除 | 五阶段入口完全取代其职责 |
| `src/inference/cli.py` | 两个子命令与文本清单 | 正式代码；CLI 测试 | 五个薄子命令与 JSON 分片 | 提交单元已经改变 |
| `configs/inference/stage1_v3.yaml` | 固定角色与窗口 | 正式配置；解析测试 | 删除 roles，增加通用参数 | alpha 与评分不再由固定角色拥有 |
| `训练与运行/sh/infer/stage1_v3.sh` | 环境与命令透传 | 正式 shell；Bash 检查 | 保留并更新提示 | 不需要增加脚本数量 |
| `tests/inference/test_stage1_v3.py` | 固定角色与旧 workflow | 正式测试 | 更新动态路径、五阶段、分片、score-only、超量跳过 | 删除退出行为并覆盖新契约 |
| `tests/inference/test_stage1_cuda.py` | 仅覆盖完整图 CUDA 流水 | 可用 GPU 才运行的 smoke | 增加 centered CUDA 前向、D2H 和 CPU 整理 | 捕获 CPU 假 wrapper 无法暴露的设备与 shape 错误 |
| `tests/inference/README.md` | 固定角色测试说明 | 测试目录文档 | 更新覆盖范围与运行命令 | 说明动态 F-alpha 和 `_BLOB_EXCEED` 边界 |
| `ops/stage1_inference_benchmark/README.md` | 旧 `calibrate` 基准示例 | 基准工具文档 | 改为五阶段中的显式命令 | 避免复制已经删除的 CLI |
| 三份推理 README | 两个命令与固定字段 | 当前契约文档 | 按 Human MD Review 批注重写 | 使提交接口与产物字段可冷读 |
| `talk/refactor/stage1_v3_inference.md` | 上一版固定角色计划 | 本轮自包含实施规格 | 整体重写并登记自查与验证 | 计划必须与最终实现同口径 |
| AdaLigand `BOX-level数据契约.md` | 固定 F1/F3 推理契约 | 权威科学契约 | 保留训练部分并重写推理部分 | 五阶段字段与路径必须有唯一权威定义 |
| AdaLigand `Stage1_V3推理重写实施记录.md` | 上一轮执行记录 | 本轮事件与验证记录 | 重写后按关键事件补充 | 不记录逐命令流水账 |
| AdaLigand `计划执行映射.md` | 连接计划、契约与记录 | 映射索引 | 更新状态与覆盖关系 | 保证后续会话能定位权威资料 |

## 主代理逐函数自查

下表按正式文件的阅读顺序记录 2026-08-21 至 2026-08-22 的人工自查。检查内容包括职责、定义位置、生产调用者、嵌套理由、Docstring 和行内注释。数学搜索的循环、线程池回调和测试内最小假对象保留；一次字段映射、单次路径转发和 producer 参数对象没有被抽成新函数。

| 文件与函数 | 职责与调用关系 | 布局与嵌套结论 | Docstring 与注释结论 |
| --- | --- | --- | --- |
| `artifacts.py:f_alpha_tag` | 把命令中的 alpha 转成路径标签；pipeline 和测试调用 | 位于冷端路径工具区首位；无嵌套 | 简单变换的一行 Docstring 已说明示例 |
| `artifacts.py:Stage1ArtifactPaths` | 集中保存四个路径维度；五阶段和测试共同构造 | 数据类位于路径工具之后；不包裹科学对象 | 类 Docstring 逐字段说明类型与示例 |
| `Stage1ArtifactPaths.__post_init__` | 规范化四个字段的类型与大小写 | 生命周期入口紧随字段；无额外检查层 | 说明只规范化、不探测文件系统 |
| `Stage1ArtifactPaths.pdb_root` | 返回单 PDB 根目录；其余路径方法和 pipeline 使用 | 单表达式 property；保留以避免重复四段路径 | 一行 Docstring 给出完整目录模板 |
| `Stage1ArtifactPaths.artifact` | 返回 probability 或动态 blobs/centered NPZ | 只有一次 probability 分支；不建立角色注册表 | 一行 Docstring 点明三类产物 |
| `Stage1ArtifactPaths.complete` | 返回当前角色 `_COMPLETE` | 单表达式；由三个可覆盖阶段调用 | 一行 Docstring 给出状态路径 |
| `Stage1ArtifactPaths.blob_exceed` | 返回 centered `_BLOB_EXCEED` | 单表达式；只在 centered、tune、evaluate 使用 | 一行 Docstring 限定 centered 角色 |
| `artifacts.py:load_stage1_npz` | 按字段解压 NPZ；全部 CPU 阶段调用 | 冷端读取工具；单个 `with`，没有 schema 检查包装 | 输入、返回映射与 `fields=None` 语义完整 |
| `artifacts.py:publish_stage1_json` | 原子写 JSON；阶段参数、状态和指标调用 | 冷端发布工具；`try/finally` 只管理临时文件 | 说明编码、末尾换行和原子替换边界 |
| `artifacts.py:publish_stage1_jsonl` | 原子写逐 PDB JSONL；evaluate 调用 | 与 JSON 发布相邻；不复用一次性行编码包装 | 说明行顺序、编码和返回语义 |
| `artifacts.py:publish_stage1_artifact` | 原子写 NPZ 并可最后发布完成标记 | 唯一分隔线后的正式发布事务；两层条件只控制标记和 dtype 失败 | 逐参数说明 NPZ 字段、完成标记字段与事务顺序 |
| `blobs.py:extract_probability_blobs` | 从完整图提取全部 26 邻域区域；blobs 阶段调用 | 单一科学入口；区域循环内只有包围盒计算，无发布嵌套 | Docstring 列出九个字段、形状、dtype、排序和不做 `min_voxels` |
| `calibration.py:CenteredCalibrationFacts` | 保存每个 PDB 的小型选择事实；内部搜索使用 | 冷端不可变数据类；字段紧邻消费函数 | 类 Docstring 逐字段说明候选轴和命中矩阵 |
| `calibration.py:_f_beta_from_counts` | 从 TP/FP/FN 计算 F-beta；两个搜索函数调用 | 数值原语放在调用者之前；单分母分支 | 一行 Docstring 已说明零分母 |
| `calibration.py:_f_beta_from_precision_recall_counts` | 从两组命中数计算 F-beta；目标函数调用 | 与上一数值原语相邻；无包装层 | 一行 Docstring 已说明两种零分母 |
| `calibration.py:_gaussian_terms` | 按 tau 预计算 Gaussian 正负项；tune 调用 | 冷端缓存原语；一层 PDB 循环 | Docstring 说明输入事实和返回映射 |
| `calibration.py:_augment_one_to_one_match` | 为当前候选寻找一条一对一增广路；由 `_scan_actual_score_thresholds` 调用并自递归 | 冷端科学原语放在扫描函数之前；移出候选循环后不再重复创建局部函数 | Docstring 逐参数说明邻接矩阵、原位匹配状态、访问掩码和布尔返回语义 |
| `calibration.py:_scan_actual_score_thresholds` | 扫描预过滤合格候选的实际 basic 分数；tune 调用 | 复杂度来自排序扫描；一对一增广路调用前述数值原语，不含局部函数 | Docstring 说明固定预过滤、稳定阈值顺序、目标和返回字段；关键累计数组有注释 |
| `calibration.py:_selection_objective` | 在固定预过滤与当前 min_voxels 下计算三项 micro F-beta 之和；两类 tune 搜索调用 | 数值目标紧邻扫描函数；按 PDB 汇总后一次计算 | Docstring 点名固定候选资格、semantic、coverage@0.3、one-to-one@0.3 |
| `calibration.py:calibrate_semantic_thresholds` | 拟合一个 alpha 的语义阈值；blobs 阶段调用 | 正式入口位于唯一分隔线后；逐 PDB 直方图后一次向量扫描 | Docstring 列出输入、摘要字段和完整 scan 字段 |
| `calibration.py:tune_centered_selection` | 先固定统一预过滤，再执行 basic 或 Gaussian 三阶段选择；tune 阶段调用 | 正式入口紧随语义拟合；只增加一个候选布尔轴，Gaussian 网格嵌套仍是科学搜索轴，不拆成转发函数 | Docstring 逐项说明候选、真实 occurrence、固定门槛、网格、目标和输出 |
| `centered.py:pack_centered_entries` | 把逐候选映射压成无 object dtype 数组；`infer_centered_boxes` 的 packer 调用 | 冷端归档函数位于唯一分隔线前；ragged 组循环共享 offsets，不增加字段类 | Docstring 列出共同、Find 专用和空候选字段契约 |
| `centered.py:infer_centered_boxes` | 编排请求物化、完整 forward、D2H、CPU 整理和异步打包；centered 阶段调用 | 正式入口位于分隔线后；局部两个回调共享当前 PDB 大数组，避免把 Dataset/collator 变成顶层包装；Find 分支改成 U-Net 早 `continue`，移除两层大块条件嵌套 | Docstring 列出全部输入字段、并行边界、返回 Future 和性能字段；行内注释说明关键数组形状与坐标系 |
| `centered.py:materialize_batch` | 在线程池物化一批 centered 请求；只由外层预取队列调用 | 必要局部回调；一层设备条件，无额外 helper | Docstring 列出 batch 中 A 字段和来源编号 |
| `centered.py:arrange_batch` | 等待 D2H 并构造逐候选共同/Find 字段；只由整理线程调用 | 必要局部回调；U-Net 早返回当前候选，Find 逻辑顺序展开 | Docstring 逐模型输出、Dataset 输入和返回条目说明形状；CUDA smoke 已确认 `voxel_final` 高级索引直接得到 `(K_source,C_voxel)`；A 原子以显式 `distance <= 10.0` 保留端点 |
| `scoring.py:build_gaussian_distance_table` | 构造 A 原子到来源体素的 5 Å 距离表；校准和正式评分调用 | 冷端可复用科学原语；一层候选循环 | Docstring 列出 ragged 输入与三个返回字段；无上界最近邻查询后显式以 `distance <= 5.0` 保留端点 |
| `scoring.py:sum_gaussian_atom_terms` | 以同一数值顺序归约 Gaussian 正负项；校准和正式评分调用 | 与距离表相邻；一层候选循环 | Docstring 说明 float64 归约、float32 归档和 Inf 排除 |
| `scoring.py:score_centered_candidates` | 计算 basic 或 Gaussian 最终分数；centered、evaluate 和测试调用 | 唯一分隔线后的正式入口；basic 早返回，Gaussian 顺序计算 | Docstring 说明模式、参数、输出轴和公式 |
| `pipeline.py:run_probability_stage` | 跨 PDB 生成 probability；CLI 调用 | 正式阶段首位；局部发布回调是跨 PDB GPU/压缩重叠边界 | Docstring 逐参数说明配置、Dataset、模型、覆盖和事务结果 |
| `pipeline.py:publish_probability` | 在线程池发布几何、性能、概率和完成标记 | 必要局部回调；顺序写三类文件，无条件嵌套 | Docstring 说明捕获边界和无返回值 |
| `pipeline.py:run_blobs_stage` | 可选拟合阈值并并行生成 blobs；CLI 调用 | 第二阶段；拟合与发布各一个直接分支 | Docstring 逐参数说明三种阈值来源、返回摘要和完成标记 |
| `pipeline.py:publish_blobs` | 在线程池读取 probability 并发布 blobs | 必要局部回调；只捕获路径、角色和阈值 | Docstring 说明输入 PDB 和完成事务 |
| `pipeline.py:run_centered_stage` | 正常 centered 或 score-only；CLI 调用 | 第三阶段；score-only 早返回，正常路径的 `_BLOB_EXCEED` 是 PDB 循环内单一短分支 | Docstring 逐参数说明两种模式、两个独立选择门槛、1000 上限和无返回值 |
| `pipeline.py:publish_centered` | 等待打包、可选评分并发布性能与 centered | 必要局部回调；单个 selection 条件 | Docstring 说明 Future、性能和异常传播用途 |
| `pipeline.py:run_tune_stage` | 读取 blobs/centered、传入固定预过滤并冻结选择 JSON；CLI 调用 | 第四阶段；basic/Gaussian 只在字段读取处分支，不裁剪 ragged 候选轴 | Docstring 逐参数说明固定门槛、目标、输入事实和返回 selection |
| `pipeline.py:run_evaluate_stage` | 发布逐 PDB 事实和聚合指标；CLI 调用 | 第五阶段；blobs/centered 在一个循环中直接转换，不建立候选适配类 | Docstring 逐参数说明两个独立门槛、评估名、超量跳过和返回指标 |
| `cli.py:main` | 定义五个命令、读取 JSON、固定分片、按需装配模型并直接调用阶段入口 | 唯一 CLI 函数；没有局部 helper，命令分派最多两层条件，避免 parse/build/dispatch 包装链 | Docstring 说明五阶段共同参数、tune 显式预过滤、分片、完整清单和无身份摘要 |

`evaluation.py:aggregate_stage1_metrics` 是用户点名的可读性样例，本轮没有改变其计算。主代理仍重新检查了该函数：Docstring 已完整列出固定字段、按阈值生成的动态字段、top-K 字段与零分母语义；三组汇总数组均有形状注释，因此没有为改注释而制造无行为差异的补丁。

测试文件按正式数据流排列。每个测试函数只设置一个可观察契约。局部 `Dataset`、`Dataset.materialize_request`、`Dataset.__init__`、`Wrapper`、`Wrapper.__call__`、`Wrapper.to`、`collator`、`infer_full_map` 和 `infer_centered_boxes` 假对象都只属于所在测试，已经逐个补充职责 Docstring。

| 测试函数 | 单一验证职责 | 自查结论 |
| --- | --- | --- |
| `test_window_geometry_and_normalized_gaussian` | 滑窗末端覆盖与归一化 Gaussian | 直接断言两个科学原语，无 helper |
| `test_blobs_keep_all_components_and_sort_stably` | 不做最小体素过滤与稳定排序 | 最小数组覆盖两个连通区域 |
| `test_blob_sort_uses_archived_float32_mean_before_linear_tie_break` | float32 均值并列后的线性索引顺序 | 构造专门的舍入边界 |
| `test_centered_start_uses_blob_bbox_feasible_interval` | 偏斜区域仍可放入 80³ | 只检查包围盒起点语义 |
| `test_centered_packing_preserves_offsets_and_feature_dtypes` | 共同/Find ragged offsets 与 dtype | 逐字段断言，不测试发布 |
| `test_centered_cpu_arranger_accepts_bfloat16_output` | BF16 在 NumPy 前提升 float32 | 最小 Dataset/wrapper/collator 均有职责 Docstring |
| `test_find_gaussian_score_uses_five_angstrom_cutoff` | 5 Å 端点纳入、超过 5 Å 排除 | 直接调用正式评分函数 |
| `test_find_centered_keeps_atom_at_ten_angstrom_boundary` | 10 Å 端点保留在 Find A 表 | 最小真实 Find centered 输入覆盖正式整理路径；局部 Dataset、wrapper 与 collator 均有职责 Docstring |
| `test_find_gaussian_score_reuses_calibration_numeric_terms_exactly` | 校准与生产逐位数值同源 | 比较同一输入的两个正式入口 |
| `test_semantic_and_instance_metrics_follow_micro_contract` | 语义、覆盖、匹配与 top-K | 单个合成 PDB 覆盖指标字段 |
| `test_probability_science_archive_excludes_performance_fields` | 科学 NPZ 与性能 JSON 隔离 | 最小完整图假函数有职责 Docstring |
| `test_centered_selection_is_written_before_first_formal_completion` | 首次压缩同时写选择字段 | 最小 centered 假函数有职责 Docstring |
| `test_cli_builds_current_dataset_without_hydra_dataclass_conversion` | 当前 Dataset 接收真实请求对象 | 最小 Dataset/wrapper 均有职责 Docstring |
| `test_cli_rejects_duplicate_pdb_before_model_loading` | 重复 PDB 在模型恢复前失败 | 只观察 CLI 边界 |
| `test_cli_uses_fixed_random_sharding_for_production_stage` | seed 3407 分片确定性 | 捕获正式阶段接收的 PDB 顺序 |
| `test_f_alpha_tag_uses_readable_decimal_path_names` | 整数、小数 alpha 标签与相邻 Python float 不碰撞 | 三个直观标签和两组精度边界 |
| `test_centered_blob_limit_uses_strict_greater_than` | 1000 正常、1001 超量 | 同一参数化测试覆盖端点两侧 |
| `test_centered_score_only_changes_two_fields` | score-only 保留其他数组 | 发布前后逐数组比较 |
| `test_tune_prefilter_is_fixed_before_basic_parameter_search` | basic tune 先固定预过滤且不裁剪 min_voxels 搜索值 | 小型高分假阳性固定未入选，较低分真实候选仍得到满分目标 |
| `test_find_calibration_uses_coarse_refined_then_minimum_stages` | Gaussian 先固定预过滤，再执行粗搜、细搜和最小体素顺序 | 单候选事实同时覆盖 `prefiltered_min_voxel > min_voxels` 的合法空选择 |

`tests/inference/test_stage1_cuda.py` 同样只有一个分隔线。两个正式测试函数及其局部 `Dataset`、`Wrapper`、模型方法和 `collator` 均已逐项检查并补充职责 Docstring：

| CUDA 测试函数 | 单一验证职责 | 自查结论 |
| --- | --- | --- |
| `test_full_map_cuda_pipeline_keeps_scientific_shape` | probability 的真实 CUDA 前向、异步 D2H 与有序融合 | 最小两层 3D 卷积，断言完整图 shape、窗口数和有限值 |
| `test_centered_cuda_pipeline_completes_d2h_and_cpu_arrange` | centered 的真实 CUDA 完整前向、异步 D2H 与 CPU 字段整理 | 最小共享 V 特征和 ligand 头，断言 `centered_probability`、`voxel_final (1,2)` 与性能候选数 |

## 验收

### 代码与科学

- 当前 Windows 环境通过 `tests/inference` 与 `tests/datasets/test_stage1_dataset.py`。
- `src/inference` 通过 `compileall`；YAML 通过 OmegaConf 解析；Shell 通过 Bash 语法检查。
- 动态路径覆盖整数与小数 alpha。
- 固定分片对同一 JSON 和参数逐次相同。
- unet centered 不含 Find 扩展字段；Find centered 保留全部扩展字段。
- 未评分 centered 不含 `score/selected`；score-only 除两字段外逐元素相同。
- 1000 个来源 blobs 正常进入后续判断，1001 个只写 `_BLOB_EXCEED` 并跳过。
- basic tune 包含不可容纳 80³ 的 blobs；Gaussian 校准与正式评分数值同源。
- evaluation 路径名编码 alpha、artifact 和 score mode，现有指标公式不漂移。

### 可读性与审查

主代理在派发独立审查前，按文件顺序逐一检查每个新增或修改函数的职责、位置、调用关系、嵌套、Docstring 和行内注释，并把结果补入本文。

随后派发三类独立审查：

1. 技术表达、Python 函数/目录布局与双线 Git；该审查者有权批准函数布局。
2. 中文注释与 Docstring 格式、粒度和字段契约。
3. 科学逻辑、用户契约漂移和测试可能遗漏的运行时缺陷。

三类审查各进行三轮全面核查；此后只复核已经报告的问题，不扩大范围。全部批准后才建立学习线并推进 `Learn/CUMULATIVE`。

## 当前状态

代码、配置、三份 Human MD Review 源 README、BOX-level 权威契约、执行记录和映射索引已经进入实现分支。主代理逐文件逐函数自查已经完成；当前 CPU 回归为 34 passed，Windows RTX CUDA smoke 为 2 passed，编译、Black、YAML 解析、五个 CLI 帮助入口和 Bash 语法检查通过。新增 centered CUDA smoke 发现并修正了 `voxel_final` 高级索引后多余转置导致 `(C_voxel,K_source)` 的真实 shape 错误；第二轮逻辑审查发现 SciPy 最近邻上界不含端点，正式实现已改为显式纳入 5 Å Gaussian 原子和 10 Å Find A 原子，并分别增加端点测试。布局/Git、注释/文档与逻辑三类独立审查都已完成三轮全面核查，第三轮报告项也已窄口径复核并全部批准。只剩双线 Git 收口和最终 handoff。
