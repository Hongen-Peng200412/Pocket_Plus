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

## 6. 当前阅读进度

- 已完成：旧链清理边界、请求对象、PDB split、BOX pool、统一 Dataset/materializer/collator。
- 下一层：三个 producer 如何消费 batch，以及 Find_0/Find_1 的 voxel scatter 差异。
- 暂不修改：目标差异之外的既有 geometry、density builder 和 backbone 注释。

## 7. 留给实现线

当前没有需要移交的事项。
