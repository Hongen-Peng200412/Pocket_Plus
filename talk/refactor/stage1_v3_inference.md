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

## 代码布局

正式代码只建立一个 `src/inference/` 包, 避免重新形成 `artifacts → component_lineage → inference → selector` 的多层转发。

| 文件 | 唯一主要职责 | 跨模块入口 |
| --- | --- | --- |
| `src/inference/artifacts.py` | 定义正式路径、NPZ/JSON 字段组和原子发布 | `Stage1ArtifactPaths`、`load_stage1_npz`、`publish_stage1_artifact` |
| `src/inference/checkpoint.py` | 从训练 run 快照和 resolved config 恢复完整 wrapper | `load_stage1_wrapper` |
| `src/inference/full_map.py` | 生成窗口、执行异步 CUDA 前向并按固定顺序融合完整概率图 | `window_starts_zyx`、`gaussian_window_weight`、`infer_full_map` |
| `src/inference/blobs.py` | 从一个冻结阈值提取并排序全部 26-连通区域 | `extract_probability_blobs`、`publish_probability_blobs` |
| `src/inference/centered.py` | 解析 80³/48³ 几何，批量重新前向并构造 centered 字段 | `infer_centered_boxes`、`pack_centered_entries` |
| `src/inference/scoring.py` | 计算基本分数与可选 A 原子高斯分数 | `build_gaussian_distance_table`、`score_centered_candidates` |
| `src/inference/evaluation.py` | 计算逐候选身份事实与 micro/macro 指标 | `evaluate_centered_pdb`、`aggregate_stage1_metrics` |
| `src/inference/calibration.py` | 冻结 F1/F3 语义阈值和 centered 三阶段参数 | `calibrate_semantic_thresholds`、`tune_centered_selection` |
| `src/inference/pipeline.py` | 执行单 PDB 概率与已冻结 centered 发布事务 | `produce_probability_map`、`produce_centered_role` |
| `src/inference/workflow.py` | 直接编排跨 PDB 有界流水、calibration 和冻结参数运行 | `run_calibration_workflow`、`run_frozen_workflow` |
| `src/inference/cli.py` | 解析 YAML 与 `calibrate`/`run` 子命令 | `main` |

`workflow.py` 不建立 producer callback 工厂。它直接调用单 PDB pipeline、blobs、calibration 和 evaluation 入口。一次字段映射、一次转发和只在一个位置使用的短逻辑全部内联。

## 显式配置

`configs/inference/stage1_v3.yaml` 的所有行为参数都由调用者填写。Python 不给以下参数设置默认值:

- `producer`、`checkpoint`、`resolved_config`、`model_code_source`；
- `split`、`pdb_list_path`、`density_root`、`box_pool_root`、`output_root`；
- `stride`、`sigma`、`window_batch_size`、`centered_batch_size`、`precision`；
- `window_workers`、`blob_workers`、`centered_workers` 与三个预取深度；
- `min_voxels`、`blob_limit`；
- `save_voxel_final`、`save_dense48`；
- `command` 与 `centered_roles`。

正式示例显式写 `stride=50`、`sigma=0.5`。F1 命令显式关闭两项保存开关, F3 命令显式打开。

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

`p_i` 是当前候选 A 原子的 `A_probability`, `d_{ji}` 是 A 原子到候选权威 V 体素中心的最近世界距离, 单位 Å。正负项不归一化。U-Net 没有 A 表, 两项均为 0。

Find 完整模式严格分三阶段：第一阶段扫描 500 组粗网格；第二阶段固定第一阶段 tau，扫描两个 lambda 的 5×5 乘数和分数下限的 15 个乘数，共 375 组；第三阶段冻结 Gaussian 参数，只扫描 `min_voxels=8..40`。基本模式和 U-Net 完整模式先按实际出现的来源分数冻结阈值，再单独扫描相同的最小体素数范围。

## 发布与恢复

- NPZ 和 JSON 先写同目录临时文件, 写入成功后使用 `os.replace` 发布; 大型 NPZ 不做重复解压重读。
- `_COMPLETE` 只在最终文件已经原子替换后建立。
- calibration 搜索可写带占位 `score/selected` 的临时候选 centered, 但不写 `_COMPLETE`。参数冻结后按最终 `min_voxels` 重新生成候选集合, 在第一次正式压缩前写入冻结值, 再发布完成标记。
- F3 的 eligible 候选数严格大于显式 `blob_limit` 时写 `status/F3_centered/_BLOB_EXCEED`, 并始终继续发布 centered; 不建立跳过 PDB 的特殊终态。
- 不建立隐藏 work 文件、不保留旧版本备份, 恢复依靠完整图、blobs 和 Git。

## 本轮逐文件状态表

本表以 `3cbae636f444fb6630eb505d210d7ce0586b4f76` 为共同基点，逐文件记录职责、调用者、生命周期、验证和动作理由；它取代前面按历史主题分组的旧清单，作为本轮收口权威。

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

## 验证

1. 本机 CPU/CUDA 组合套件当前 29 项通过, 覆盖窗口边界、float32 blob 稳定排序、偏斜 blob BOX、BF16 转换、centered 字段、Python 3.10 CLI、Dataset 直接构造、calibration 完成标记、Gaussian 三阶段与数值同源、micro/macro、top-K、字段选择读取和原子发布。
2. 本机 RTX 4060 CUDA smoke 已通过 125 个真实 Conv3d 窗口的异步 H2D/前向/D2H/融合执行链。
3. Dataset 测试覆盖 V3 NPY/mmap、实际 80³ 数值边界与推理线程共享 LRU 的计数一致性。
4. 真实 checkpoint smoke 仍须至少覆盖 `unet_c1` 与一个 Find。最终 checkpoint 尚未确定, 本轮不自行提交服务器任务；得到用户授权后在目标环境执行, 不通过删除科学字段规避显存问题。
5. GPU 采样工具已在本机合成 CUDA smoke 上产生 19 个样本并成功汇总。该结果只验收工具；正式 BOX/s、PDB/s、GPU 活跃比例和队列等待必须使用真实 checkpoint 另行记录。
6. 布局/Git、注释/Docstring、科学逻辑三类审查均完成两轮全面审查；第二轮报告的问题经窄口径复核后全部 `APPROVED`，没有再扩大审查范围。

## Git 双线收口

实现线已按真实顺序保存删除、实现、修复和验证。学习线已从共同基点 `3cbae636f444fb6630eb505d210d7ce0586b4f76` 按以下顺序重建：

1. 集中删除全部旧非测试文件。
2. 提交当前规格、NOTE、README 与两份学习概览。
3. 按产物契约、完整图与 blobs、centered、评分评估、编排配置的依赖顺序提交正式代码。
4. 最后集中提交全部测试文件和旧测试删除。
5. 在实现端点与学习端点运行相同测试, 比较 Python 可执行语句、配置、字段级产物和 Git tree。
6. 等价通过后将 `Learn/CUMULATIVE` 快进到学习端点。实现分支长期保留, 不推送远端。

本轮首个已批准实现端点是 `codex/stage1-inference-v3@bdecd9802fdef4dcc28401411b35f31a72e5a3cc`，首个已批准学习端点是 `Learn/stage1-inference-v3@c62de259f48da4f04962d4b68658a76a6afbafbe`。两端 tree 均为 `226b56e7ed34b44545af6236afba4653caac32b9`；后续仅追加不改变科学行为的最终验证与 handoff 记录。
