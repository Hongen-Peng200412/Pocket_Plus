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
| `src/inference/centered.py` | 解析 80³/48³ 几何, 批量重新前向并构造 centered 字段 | `run_centered_inference` |
| `src/inference/scoring.py` | 计算基本分数与可选 A 原子高斯分数 | `score_centered_candidates` |
| `src/inference/evaluation.py` | 冻结 F1/F3 阈值, 计算 f1/f2、micro/macro 和逐候选事实 | `calibrate_stage1`、`evaluate_stage1` |
| `src/inference/pipeline.py` | 直接装配 Dataset、模型和三阶段有界流水线 | `Stage1InferencePipeline` |
| `src/inference/cli.py` | 解析 YAML/命令, 选择 `calibrate`、`produce`、`evaluate`、`all` | `main` |

`pipeline.py` 不建立 producer callback 工厂。`Stage1InferencePipeline` 直接调用 `infer_full_map`、`publish_probability_blobs` 和 `run_centered_inference`。一次字段映射、一次转发和只在一个位置使用的短逻辑全部内联。

## 显式配置

`configs/inference/stage1_v3.yaml` 的所有行为参数都由调用者填写。Python 不给以下参数设置默认值:

- `stage1_model_name`、`checkpoint_path`、`resolved_config_path`、`allow_current_workspace_code`；
- `split`、`pdb_list_path`、`density_root`、`box_pool_root`、`output_root`；
- `stride`、`sigma`、`window_batch_size`、`centered_batch_size`、`precision`；
- `window_workers`、`blob_workers`、`centered_workers` 与三个预取深度；
- `min_voxels`、`blob_limit`、`enforce_blob_limit`；
- `save_voxel_final`、`save_dense48`；
- `command` 与 `centered_roles`。

正式示例显式写 `stride=50`、`sigma=0.5`。F1 命令显式关闭两项保存开关, F3 命令显式打开。

## 执行流水线

### 完整图

CPU 线程提前物化固定顺序的 80³ 请求并组 batch。主线程把页锁定 batch 非阻塞传到 GPU, 执行 `wrapper.forward_voxel_probability()` 和 sigmoid, 再把 float32 概率异步复制到专用页锁定 CPU buffer。唯一融合线程等待 CUDA event 后按窗口提交顺序执行 float32 Gaussian 累加。

完整图概率不乘 hardmask。窗口沿每轴从 0 开始, 最后一个窗口强制以 `length-80` 结束, 因而无 padding 且覆盖边界。

### 连通区域

完整概率图原子发布后, CPU 进程读取该 PDB 的概率文件并并行提取 F1/F3 26-连通区域。GPU 主线程同时开始下一个 PDB 的完整图。每个阈值下的全部区域均落盘; 排序键为平均来源概率降序, 再按区域最小完整图 C-order 线性索引升序。

`fits_centered_box` 只说明区域能否完整放入 80³ BOX。它不删除区域。`min_voxels` 也不在本阶段生效。

### centered

F1 使用 voxel-only centered 前向, 保存来源概率、重算概率和共同几何, 不保存 A/P、`voxel_final` 或 48³ 数组。

F3 使用完整 wrapper 前向。所有模型保存 `voxel_final` 和三张 V-centered 48³ 数组。Find 额外保存 A/P 表; U-Net 不出现 A/P 字段。

CPU 线程提前物化 centered batch。主线程执行 H2D 与 GPU 前向。完成的 batch 非阻塞复制到 CPU 后交给写出线程。每个 PDB 的所有候选齐备后一次性发布最终 NPZ 和 `_COMPLETE`。

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

## 发布与恢复

- NPZ 和 JSON 先写同目录临时文件, 完整重读后使用 `os.replace` 发布。
- `_COMPLETE` 只在最终文件已重读后原子建立。
- calibration 可以先写没有 `score/selected` 的 centered 文件, 但不写 `_COMPLETE`。参数冻结后原子替换同一路径, 增加 `score/selected`, 再发布完成标记。
- F3 的 eligible 候选数严格大于显式 `blob_limit` 时写 `status/F3_centered/_BLOB_EXCEED`。`enforce_blob_limit=true` 时不发布 centered; 为 false 时继续。
- 不建立隐藏 work 文件、不保留旧版本备份, 恢复依靠完整图、blobs 和 Git。

## 文件清单

以下表覆盖本轮开始时仍位于活动树中的全部旧推理、评估、Selector 配置、脚本与专项测试。

### 正式与运维文件

| 当前路径 | 当前职责或调用者 | 生命周期 | 测试 | 目标动作与理由 |
| --- | --- | --- | --- | --- |
| `ops/Gauss_Scorer/tune.py` | 旧 forest/Fα/Li 高斯调参 | 退出 | `tests/inference/test_gauss_scorer_tuning.py` | 删除; 新评分直接读取 F3 centered |
| `ops/Gauss_Scorer/grid_find0_calibration.json` | Find_0 旧两阶段参数网格 | 退出 | 同上 | 删除; 新网格由 calibration 配置显式给出 |
| `ops/Gauss_Scorer/build_refinement_grid_find0.sh` | 生成旧第二阶段网格 | 退出 | shell 语法 | 删除 |
| `ops/Gauss_Scorer/evaluate_find0_calibration.sh` | 提交旧单项调参 | 退出 | shell 语法 | 删除 |
| `ops/Gauss_Scorer/merge_find0_calibration.sh` | 合并旧数组任务 | 退出 | shell 语法 | 删除 |
| `configs/selector/Find_0.yaml` | 构造已删除 Selector | 退出 | `tests/selector/test_selector_configs.py` | 删除 |
| `configs/selector/Find_1.yaml` | 构造已删除 Selector | 退出 | 同上 | 删除 |
| `configs/selector/Find_2.yaml` | 构造已删除 Selector | 退出 | 同上 | 删除 |
| `configs/selector/unet_c1.yaml` | 构造已删除 Selector | 退出 | 同上 | 删除 |
| `训练与运行/sh/infer/prepare_inference_pdb_lists.sh` | 旧 split 清单入口 | 退出 | shell 语法 | 删除; V3 split 清单由配置显式传入 |
| `训练与运行/sh/infer/Find_0_calibration_probability.sh` | 旧概率角色 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_freeze_thresholds.sh` | 旧 Fα 阈值冻结 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_calibration_F1.sh` | 旧 F1 生产 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_validation_F1.sh` | 旧 validation F1 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_train_F1.sh` | 旧 train F1 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_train_F1_shard_01.sh` | 固定分片一次脚本 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_Falpha.sh` | 旧七种 Fα 角色 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_Li.sh` | 旧 Li 角色 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_Gauss.sh` | 旧 forest 回填 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_calibration_CLG.sh` | 旧 CLG 生产 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_validation_CLG.sh` | 旧 CLG 生产 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/Find_0_train_CLG.sh` | 旧 CLG 生产 | 退出 | shell 语法 | 删除 |
| `训练与运行/sh/infer/README.md` | 说明已经删除的入口 | 当前文档 | 冷读 | 从新代码与字段重写 |

### 旧专项测试

| 当前路径集合 | 当前状态 | 目标动作 |
| --- | --- | --- |
| `tests/artifacts/test_stage1_io.py`、`test_stage1_states.py` | import 已删除的 `src.artifacts` | 删除并由 `tests/inference/test_artifacts.py` 替代 |
| `tests/component_lineage/test_stage1_clg.py`、`test_stage1_structures.py` | import 已删除的 CLG/forest | 删除, 不建立替代 |
| `tests/evaluation/test_stage1_calibration.py`、`test_stage1_instance_metrics.py` | import 已删除的 `src.evaluation` | 删除并由新 calibration/evaluation 测试替代 |
| `tests/inference/test_stage1_full_map.py`、`test_stage1_checkpoint.py`、`test_stage1_centered.py`、`test_stage1_assembly_cli.py` | 旧接口 | 删除后按新入口重写同职责测试 |
| `tests/inference/test_li_centered.py` | Li 已退出 | 删除, 不建立替代 |
| `tests/inference/test_gauss_scorer.py`、`test_gauss_scorer_tuning.py` | forest/Fα 回填接口 | 删除并由新 scoring/calibration 测试替代 |
| `tests/selector/test_antichain_dp.py`、`test_ccln.py`、`test_dataset_and_freeze.py`、`test_density_munet_lite.py`、`test_inference_schema.py`、`test_input_fusion.py`、`test_selector_calibration.py`、`test_selector_configs.py`、`test_train_sampler.py`、`test_wrapper_losses.py` | Selector 已退出 | 整组删除, 不建立替代 |

### 保留并扩展的训练接缝

| 路径 | 当前职责 | 本轮动作 |
| --- | --- | --- |
| `src/datasets/stage1_dataset.py` | 从 V3 NPY/mmap 物化 80³ producer 输入 | 保留; 增加显式原始 exp/sim 80³ 读取入口供 dense48 |
| `src/datasets/stage1_collate.py` | 组装固定体素与 Find ragged 原子字段 | 保留不改科学语义 |
| `src/model/stage1_model.py` | 提供 voxel-only 和完整 forward | 保留; 不增加推理包装 |
| `src/wrappers/voxel_point_stage1.py` | wrapper checkpoint 生命周期与两个 forward | 保留; 修正文档中已退出的 hardmask 后处理描述 |
| `src/stage1_producers.py` | 四个 producer 名单 | 保留并由新 CLI 直接读取 |
| `CLAUDE.md` | 仓库导航 | 收口时删除 Selector/forest/CLG 旧导航并指向新 README |

## 验证

1. 纯 CPU 单元测试覆盖窗口边界、Gaussian 融合、连通区域排序、80³/48³ 几何、offsets、空产物、评分、F1/F2/F3、micro/macro、top-K 与原子发布。
2. CUDA 测试比较串行和异步 D2H 完整图逐元素等价, 并检查 centered batch 顺序。
3. 代表性真实 Dataset 测试使用 V3 NPY/mmap 和一个可读 PDB, 核对 exp/sim、50 维 A 基础特征和 48³ 数组。
4. 真实 checkpoint smoke 至少覆盖 `unet_c1` 与一个 Find。若本机 8 GB RTX 4060 不能容纳 Find, 服务器正式环境补测, 不通过降低科学字段回避。
5. GPU 基准保存串行/流水线 BOX/s、PDB/s、GPU 活跃比例、利用率分布和队列等待。性能事实进入执行记录。
6. 第一轮三类全面审查在核心三阶段与评分链完成后进行。第二轮在配置、shell、文档、真实 smoke 和基准完成后进行。此后只窄口径复核已报告问题。

## Git 双线收口

实现线按真实顺序保存删除、实现、修复和验证。稳定后从共同基点重建 `Learn/stage1-inference-v3`:

1. 集中删除全部旧非测试文件。
2. 提交当前规格、NOTE、README 与两份学习概览。
3. 按产物契约、完整图与 blobs、centered、评分评估、编排配置的依赖顺序提交正式代码。
4. 最后集中提交全部测试文件和旧测试删除。
5. 在实现端点与学习端点运行相同测试, 比较 Python 可执行语句、配置、字段级产物和 Git tree。
6. 等价通过后将 `Learn/CUMULATIVE` 快进到学习端点。实现分支长期保留, 不推送远端。
