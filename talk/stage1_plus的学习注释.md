# Pocket_Plus AdaLigand Stage1 学习注释

## 1. 文档角色

本文记录 Pocket_Plus 中 AdaLigand Stage1 增量的代码结构、数据流和个人学习注释。通用注释与 Git 规则见 `talk/注释规则.md`，本文不重复完整规则。

本轮只解释代码、补注释、整理本文和组织学习分支。测试文件原样归位但不阅读、不注释、不运行；目标差异之外的既有依赖只读，不修改注释；最终删除的旧文件不纳入学习说明。

## 2. 本轮范围

| 项目 | 值 | 含义 |
| --- | --- | --- |
| 共同基点 | `4d798c9c6e49d031e8f5fefa38f99fee4f463209` | 本轮实现线与学习线共同起点 |
| 实现线 | `codex/adaligand-stage1` | 保留真实实现与调试历史，只读 |
| 实现终点 | `8e7eed0df4008cb8ddb8ec74027be49bc725a8d4` | 本轮最终正确代码基准 |
| 学习线 | `Learn/00-stage1-cleanup` 至 `Learn/07-stage1-selector` | 按逻辑依赖重建的线性历史 |
| 累计学习分支 | `Learn/model-cumulative` | 完成后与最新学习里程碑指向同一提交 |

本轮允许相对实现终点新增普通注释、Docstring、YAML 注释以及 `talk/注释规则.md` 和本文。其它代码与配置差异均视为学习线错误。

## 3. 里程碑与内部提交

```text
4d798c9
  └─ 00 删除旧链与悬空引用                         [Learn/00-stage1-cleanup]
      ├─ 01 请求、split 与 BOX pool
      └─ 01 Dataset、materializer 与 collator      [Learn/01-stage1-data]
          ├─ 02 producer 模型与 voxel-only
          └─ 02 wrapper、loss、CPC 与配置           [Learn/02-stage1-producers]
              └─ 03 artifact 路径、状态与 NPZ       [Learn/03-stage1-artifacts]
                  ├─ 04 tree/working-tree 结构
                  └─ 04 forest、CLG 与 overlap      [Learn/04-stage1-lineage]
                      ├─ 05 probability/full-map/指标
                      └─ 05 阈值校准与报告           [Learn/05-stage1-full-map]
                          ├─ 06 三类 centered artifact
                          └─ 06 runner/assembly/CLI  [Learn/06-stage1-centered-runtime]
                              ├─ 07 反链 DP/oracle
                              ├─ 07 Selector 输入
                              ├─ 07 V5+D/CCLN
                              └─ 07 训练/校准/推理    [Learn/07-stage1-selector]
                                                     [Learn/model-cumulative]
```

实现线中的数值等价、BF16 桥接、边界校验和命名修复不会作为独立修复提交重演，而是直接进入各自所属层。

## 4. 系统总览

三个 Stage1 producer 是 `Find_0`、`Find_1` 和 `unet_c1`。它们共享 80³ 请求与物化路径，输出完整图 ligand probability；随后由阈值校准、component forest/CLG、三类 centered artifact 和 Selector 逐层消费。

```text
Stage G keep-list
  → PDB split
  → train/validation BOX pool
  → ResolvedStage1Crop
  → Stage1Dataset / Stage1BatchCollator
  → Find_0 / Find_1 / unet_c1
  → full-map probability
  → threshold calibration
  → component forest / CLG
  → F1_centered / CLG_centered
  → Selector
  → selection
  → Selected_Refined_Centered
```

模块职责预览：

| 层 | 主要目录 | 职责 |
| --- | --- | --- |
| 数据 | `src/datasets/stage1_*` | 冻结身份和整数起点，把整图资产统一物化为 80³ 样本 |
| producer | `src/model/`、`src/wrappers/`、`configs/` | 三个模型、训练监督、CPC 接缝和 voxel-only 推理入口 |
| artifact | `src/artifacts/` | 路径、原子发布、完成状态和无 pickle NPZ schema |
| 谱系 | `src/component_lineage/` | 多阈值连通组件森林、CLG 和 occurrence overlap |
| 完整图 | `src/inference/`、`src/evaluation/` | 滑窗融合、阈值冻结、centered 生产和续跑编排 |
| Selector | `src/selector/` | 冻结 CLG 输入、V5+D/CCLN、精确反链选择和校准 |

## 5. 数据请求与冻结池

### 5.1 身份划分

`stage1_split.py` 读取 Stage G `keep_list.jsonl`，先按 `pdb_id` 聚合，再用稳定哈希排名冻结 validation 300、calibration 100、train 75% 和 held-out pool。同一 PDB 的所有 pair/occurrence 不能跨 split。

### 5.2 请求对象

`ResolvedStage1Crop` 只描述一次裁剪请求，不读取数据。核心字段为：

| 字段 | 类型 / shape | 语义 |
| --- | --- | --- |
| `pdb_id` | `str` | 整图身份，规范化为小写 |
| `box_start_zyx` | `tuple[int,int,int]`，`(3,)` | 已解析的 80³ 整数起点 |
| `require_targets` | `bool` | 是否需要训练/validation 监督 |
| `role` | `str` | `center/bias/context/sliding/centered` |
| `occurrence_id` | `int|null` | 上游 ligand occurrence 身份 |
| `candidate_index` | `int|null` | bias/context 冻结候选下标 |

`resolve_stage1_start()` 对每个轴执行：

$$
resolved\_start_a=\operatorname{clamp}(requested\_start_a,0,L_a-80).
$$

因此请求只会向图内平移，不做图外补零。数组索引与 shape 使用 ZYX；世界坐标使用 XYZ。

### 5.3 BOX pool

`stage1_box_pool.py` 只保存整数起点和身份索引，不保存 80³ 数组：

- 每个 occurrence 保存一个 center 起点。
- 每个 occurrence 按体积均匀球偏移冻结 30 个 bias 起点。
- 每个 PDB 独立生成至少包含 1000 个 core receptor heavy atoms 的 context 起点池。
- pool 根 `manifest.json` 是正式文件发现入口，目录中未列出的遗留 NPZ 不会被消费。
- validation 把一次固定的 `1:5:3` 选择保存到 `validation_selection.npz`。

训练时每个 PDB 每 epoch 最多选择 50 个 occurrence；每个入选 occurrence 使用 1 个 center、5 个 bias，以及在 context 池非空时抽取的 3 个 context。context 只有 1–2 项时允许有放回抽样，空池则省略 context。

### 5.4 统一物化链

`Stage1Dataset` 不决定样本位置，只消费 `ResolvedStage1Crop`。一次物化按以下顺序进行：

```text
request
  → 读取或复用 receptor/label 与 exp/sim/union 整图
  → 从整图裁 80³ density 和 target
  → 选择 core+8 Å receptor atoms
  → 构造 hardmask、voxel_label 与原子坐标
  → 按 producer 构造 density_input
  → train-only 同步 90° 旋转
  → 转为固定 dtype 的 tensor sample
```

worker-local LRU 按数组真实字节数缓存整图资产，不缓存裁好的 BOX，所以缓存只改变 I/O 次数，不改变请求身份或样本数值来源。

### 5.5 Find 与 unet_c1 的样本差异

共同 dense 字段包括：

| 字段 | 单样本 shape / dtype | 语义 |
| --- | --- | --- |
| `density_input` | `(C,80,80,80) float32` | C=56（Find）或 1（unet_c1） |
| `hardmask` | `(80,80,80) bool` | core receptor home-voxel 集合 |
| `ligand_area_target` | `(80,80,80) bool` | train/validation 的 ligand union target |
| `voxel_label` | `(80,80,80) bool` | binding receptor home-voxel target |
| `box_origin_world` | `(3,) float32` | BOX 的 XYZ 世界原点 |
| `voxel_size_world` | `(3,) float32` | XYZ 轴 voxel size |

Find 额外返回 core+8 Å 原子表：`atom_feat (N_A,49)`、三套 XYZ 坐标、`atom_global_indices`、`atom_is_in_core_box` 和可选 `atom_label`。`unet_c1` 仍会在 Dataset 内读取 receptor 以构造 auxiliary target，但不会把原子表作为模型输入返回。

### 5.6 Collator 的 ragged 原子组织

80³ dense 字段直接堆叠为 batch。原子表沿第 0 轴拼接，不 pad：

- `atom_counts`: `(B,) int64`，每个 BOX 的原子数。
- `atom_offsets`: `(B+1,) int64`，首项为 0，`[offsets[i], offsets[i+1])` 切出第 i 个 BOX 的所有原子。
- `atom_batch_index`: `(N_A_total,) int64`，逐原子记录所属 BOX。

因此模型既能连续处理整张原子表，又能用 offsets 或 batch index 恢复样本边界。

## 6. Producer 模型主链

### 6.1 三个 producer

| producer | density 输入 | receptor voxel 输入 | point 路径 |
| --- | --- | --- | --- |
| `unet_c1` | 1D `exp_clipnorm_nopost` | 无 | 无 |
| `Find_0` | 56D `ALL` | core atoms 的 raw49 hard floor/sum scatter | 共同 `[8,4,0]` point blocks |
| `Find_1` | 56D `ALL` | 49D learned value + 2D occupancy soft scatter | 共同 `[8,4,0]` point blocks |

`Find_0` 和 `Find_1` 的科学差异只在送入 voxel backbone 前的 receptor grid。两者的 point-side `Stage1EmbedHead`、point backbone 和 A/P 输出路径保持一致。

### 6.2 完整 forward

`VolumePointStage1Model.forward()` 的主顺序为：

```text
规范化 Stage1 batch
  → real-only Stage1EmbedHead
  → density + receptor grid
  → 1–3 次 voxel recycle
  → 最后一轮生成候选 C 与 P anchors（若配置启用）
  → point backbone
  → A/P heads
  → sparse refine（若配置启用）
  → 发布 V/P/A centered feature hooks
```

AdaLigand 当前五套 producer 配置不实例化 sparse refine，但通用实现继续保留。学习时要区分“代码具有该能力”和“当前配置实际启用该能力”。

### 6.3 voxel-only 入口

`forward_voxel_probability(batch)` 固定执行三次 recycle，并跳过 point blocks、point backbone、候选 C、P、A/P heads 和 sparse refine。返回值是 sigmoid 前的 `voxel_logits_ligand (B,1,80,80,80)`。

三条最短路径为：

```text
unet_c1: density → voxel backbone
Find_0:  density → raw49 core hard scatter → voxel backbone
Find_1:  density → non-block embed/scatter → voxel backbone
```

Find_1 必须先对 core+8 Å完整原子表执行 feature projection，再筛出 core 原子进行 scatter。虽然“先筛 core 再投影”在数学表达上等价，但会改变矩阵形状和底层 kernel，无法保证与完整 forward 逐元素一致。

### 6.4 centered 特征出口

完整 forward 通过 `_publish_stage1_feature_hooks()` 把内部多尺度输出发布为稳定直键：voxel 层形成 V 特征，Find 另外形成 P/A 特征。这里只建立模型到 artifact 层的接口；具体 NPZ 对齐与 ragged schema 在后续 artifact/centered 里阅读。

### 6.5 Wrapper 与四类监督

`VoxelPointStage1Wrapper` 是 Lightning 生命周期的 thin coordinator。它把 backbone 输出交给四类可选监督：

| loss | target 来源 | CPC1 权重 | CPC2 权重 |
| --- | --- | ---: | ---: |
| `L_atom` | `binding_atom` 对齐真实 receptor atom | 1.0 | 1.0 |
| `L_voxel_aux` | `voxel_label`，仅在 `hardmask` home voxels 计算 | 0.1 | 0.0 |
| `L_voxel_ligand` | schema-v3 union `ligand_area_target` | 1.0 | 0.0 |
| `L_P` | P anchor home voxel 采样同一 ligand target | 0.1 | 0.1 |

`hardmask` 只限定 auxiliary receptor loss；ligand target 本身不乘 hardmask。通用 Pocket_Plus 的旧 `ligand_dist_map` 路径仍保留，但 AdaLigand Dataset 直接提供 union mask crop。

### 6.6 五套训练配置

```text
unet_c1                  从头训练
Find_0/CPC1 → Find_0/CPC2
Find_1/CPC1 → Find_1/CPC2
```

CPC2 只能读取同名 CPC1 BEST。`src/train.py::_load_model_only_checkpoint()` 加载完整 `state_dict` 并调用 `on_load_checkpoint()`，但不恢复 optimizer、scheduler、epoch 或 global step。随后 `adaligand_stage2.yaml` 冻结 CPC1 的 voxel/point/embed 主干，只训练 interaction 与 A/P 尾部。

### 6.7 正式推理恢复

`src/inference/checkpoint.py::load_stage1_wrapper()` 与 CPC2 初始化不同：它从相邻 resolved config 实例化完整 wrapper，strict 加载完整 `state_dict`，执行 `on_load_checkpoint()` 并切到 eval。推理入口因此不依赖旧的裸 backbone loader。

## 7. Artifact 协议

### 7.1 路径身份

`Stage1ArtifactPaths` 用 `(stage1_model_name, split, pdb_id)` 唯一定位一个 PDB 的全部正式产物：

```text
stage1_outputs/<producer>/<split>/<pdb_id>/
├─ probability/
├─ components/
├─ centered/
└─ status/<output_role>/_COMPLETE
```

producer 只能是 `Find_0`、`Find_1` 或 `unet_c1`。五个 output role 是 `probability`、`components`、`F1_centered`、`CLG_centered` 和 `Selected_Refined_Centered`。

### 7.2 发布状态

```text
PDB _RUNNING              临时互斥租约，任一 role 生产期间阻止其它 worker
status/<role>/_COMPLETE   该 role payload 已关闭、校验并发布
PDB _BLOB_EXCEED          t_F1 eligible component 超过上限的可识别终态
```

`pdb_is_consumable()` 的条件是：不存在 `_RUNNING`，不存在 `_BLOB_EXCEED`，并且消费者要求的全部 role 都有 `_COMPLETE`。完成标记与 payload 分离，因此 centered 目录可以只包含三个正式聚合 NPZ。

### 7.3 原子 IO

JSON 和 NPZ 都先写同目录临时文件。NPZ 发布前后均以 `allow_pickle=False` 读取，并拒绝 object dtype；正式路径只在临时归档通过结构校验后由 `os.replace()` 替换。

这套 IO 保证“文件存在”不会被误解为“半写入文件可消费”。role `_COMPLETE` 仍必须在 payload 发布之后单独形成。

### 7.4 Ragged offsets

一个 PDB 的同类 centered entries 聚合成一个 NPZ。长度变化的数据不使用 object array，而是拼接 value 表并保存 `int64 (N_entry+1,)` offsets：

```text
第 i 个 entry 的值 = values[offsets[i] : offsets[i+1]]
```

主要分组包括 voxel values、voxel auxiliary values、P values、A values，以及 CLG candidate membership。每个 offsets 字段必须从 0 开始、单调不减，并以对应 value 表长度结束。

`Selected_Refined_Centered` 的失败 entry 不伪造全零固定 V grid。只有 `refine_status=success` 的 entry 保存 V grid，并由 `feature_entry_index` 映射回全部 entry 表。

## 8. Component 谱系对象

### 8.1 阈值方向

完整图概率在多个阈值上做 26-连通组件。降低阈值时组件只会扩大或合并，因此树的方向定义为：

- parent：更低阈值、mask 更大的直接包含组件。
- child：更高阈值、mask 更小的直接被包含组件。
- root：该棵树最低阈值方向的最老组件。

`threshold_grid_index=j` 对应物理阈值 `j/32768`。节点的 voxel mask 以完整图 ZYX C-order linear index 保存。

### 8.2 对象层次

```text
ComponentForest
  └─ ComponentTree (唯一 root)
       └─ ComponentNode
            ├─ parent
            ├─ children
            └─ voxel/bbox/centroid/probability/eligibility
```

`ComponentTree` 校验 parent/children 双向一致、无环和 root 全可达；`ComponentForest` 用 `(tree_id,node_id)` 唯一寻址节点。对象可以编码为 `forest.npz` 的 ragged arrays，也可以从 arrays 恢复回连接好的对象树。

### 8.3 WorkingTree

CLG 枚举不能修改正式 forest，因此 `WorkingTree` 只复制拓扑 identity 和 `active` 状态，不复制大体积 voxel mask。一次扫描选中 seed `g` 后，`delete_D(g)` 删除：

$$
D_W(g)=Ancestors_W(g)\cup Subtree_W(g).
$$

这里的删除只把工作副本节点标记为 inactive；原 `ComponentNode` 及其 payload 始终只读。后续算法由此可以反复查询 active ancestors、children、sisters 和 subtree。

### 8.4 Forest 构造

`build_component_forest()` 对去重后的实际阈值按从高到低顺序处理：

1. 在 `probability >= j/32768` 上计算 26-连通组件。
2. 保存每个组件的完整图 linear voxel index、bbox、centroid 和概率统计。
3. 根据最小/最大 voxel 数和解析后的 80³ BOX 能否包含 bbox，标记 candidate eligibility。
4. 高阈值组件在相邻低阈值层中必须落入唯一组件；该包含组件成为 direct parent。
5. 按稳定深度优先顺序分配 `tree_id/node_id`。

eligibility 只决定节点能否参与候选，不会把不合格节点从 forest 删除。因此 forest 仍完整描述全部实际阈值层的连通谱系。

### 8.5 CLG 枚举

CLG seed 先取 `t_F1` 层 eligible 节点，再按阈值从高到低继续扫描更低阈值节点。一次尝试：

```text
seed
  → 向高阈值沿 children 展开；多 child 计一次 split event
  → 向低阈值沿 parent 展开；出现 sisters 计一次 merge event
  → merge 时原子加入 parent + 全部 sisters
  → 找到覆盖全部 candidates 的唯一 oldest node
```

depth1/depth2 分别由 split/merge 预算和 32/64 node cap 控制。超过 node cap 只拒绝当前尝试，不产生 `CLG_id`；成功与失败最终都对当前 seed 应用相同的 `D_W(g)`。

同一 forest node 出现在多个 CLG 中是允许的：`D_W(g)` 删除的是 seed 的 ancestors 与 subtree，不等于删除该尝试临时收集到的所有 sisters。下游需要按 CLG 保留局部竞争关系，并在最终唯一预测集合处再去重。

### 8.6 GT overlap

`overlap.npz` 只保存 candidate mask 与 GT occurrence mask 的正交集体素数。`overlap_occurrence_index` 是指向同文件 `occurrence_id` 表的局部行号，不是 occurrence identity 本身。

IoU、online oracle 和 Selector loss 在下游根据交集、candidate voxel count 与 occurrence voxel count计算；谱系层不提前做选择。

## 9. 完整图概率与指标

### 9.1 窗口覆盖

每个轴使用 80 长度、40 stride，并强制加入 `length-80` 作为末端起点。三轴起点做笛卡尔积，因此所有窗口都是完整真实 crop，且边缘不会因 stride 余数漏掉。

每个 80³ 窗口使用归一化坐标 `[-1,1]³` 上、`sigma=0.5` 的严格正 Gaussian 权重。融合维护两张 float32 完整图：

$$
P_{sum}(v)=\sum_w G_w(v)P_w(v),\qquad
W_{sum}(v)=\sum_w G_w(v).
$$

最终概率为 `P_sum/W_sum`，并要求每个 voxel 的 `W_sum>0`。

### 9.2 Producer 后处理

`forward_voxel_probability()` 返回 logits；`logits_to_probability()` 统一执行稳定 sigmoid 并转为 CPU float32。

- `Find_0/Find_1`：完整融合后把 receptor `hardmask` home voxels 概率清零。
- `unet_c1`：不使用 receptor hardmask，保留融合概率。

同一 `postprocess_ligand_probability()` 也供 centered BOX 使用，避免完整图与居中推理采用不同遮蔽语义。

### 9.3 完整图发布

`probability_map.npz` 保存 `(D,H,W) float32` 概率以及 `(3,) float32` 的 `origin_xyz`、`voxel_size_xyz`，因此单个 NPZ 已包含把离散体素索引换算到世界坐标所需的几何。`geometry.json` 保存相同的 XYZ 原点与体素尺寸，并额外保存完整图 shape、窗口 shape/stride 和 Gaussian sigma；两个文件的同名几何字段必须逐值一致。两份 payload 完成后才发布 `probability` role。

### 9.4 指标分工

- voxel AP：每个 PDB 在完整网格上计算，再对含 GT 正 voxel 的 PDB 做 macro 平均。
- semantic Dice：在冻结阈值上统计完整图 TP/FP/FN。
- coverage：允许多预测对同一 GT 或多 GT 对同一预测，只问双向覆盖是否达标。
- one-to-one：先对连续分数 `sqrt(c_pred*c_GT)` 做一次固定 Hungarian，再在同一配对上应用多个 coverage 阈值。
- top-K：按连续候选质量分取前 K，只问是否存在任一双向 coverage 达标 pair，不做 Hungarian。

其中 `c_pred=intersection/pred_size`，`c_GT=intersection/GT_size`。指标可以直接消费 `overlap.npz` 的交集计数，无需重新物化所有完整 mask。

### 9.5 Calibration 阈值扫描

每个 voxel 概率映射为整数 bin：

$$
j=\lfloor p\times32768\rfloor,\qquad j\in[0,32768].
$$

`ThresholdHistogram` 分别累计 GT 正、负 voxel 的 32769-bin 直方图。对直方图从高 bin 向低 bin 反向累积，即可一次得到所有阈值 `p>=j/32768` 的 micro TP/FP/FN。

七个固定 `alpha` 为：`1/2, 2/3, 4/5, 1, 5/4, 3/2, 2`。每条 `F_alpha` 曲线都按 `j=0..32768` 升序取第一个最大值，因此并列时冻结较低的整数阈值。`alpha=1` 对应 `t_F1`。

### 9.6 两阶段 calibration

1. calibration 100 先全部发布 probability。
2. 第一遍只累计直方图并冻结七个阈值。
3. 第二遍在 `t_F1` 临时构造单层组件，计算 fitted Dice、coverage、固定 Hungarian 和 top-K。
4. 原子发布 `thresholds.json`、`threshold_scan.npz`、`metrics.json`，最后发布 producer 级 calibration `_COMPLETE`。

阈值选择和 fitted 报告都只使用 calibration，不反向选择训练 epoch 或 checkpoint。

## 10. 当前阅读进度

- 已完成：旧链清理、数据层、producer、artifact、谱系、完整图概率、指标与阈值 calibration。
- 下一层：F1/CLG/Selected 三类 centered BOX 如何恢复模型特征并写入聚合 NPZ。
- 暂不修改：目标差异之外的既有 geometry、density builder 和 backbone 注释。

## 11. 留给实现线

当前没有需要移交的事项。
