# Stage1 推理重写 NOTE 清单

本文整理 `talk/global/global_8.19.md` 与 `C:/Users/15919/Desktop/Matcher/matcher/data/` 中现存 `# NOTE`，用于约束 Pocket Plus 的 Stage1 推理重写。本文只区分已经确认的决定、当前实现事实、对 Stage1 的影响和仍未决定的接口；它不是实现日志，也不把 2026-08-19 的设想冒充为已经存在的代码行为。

## 1. 已经确认的 Stage1 主线

1. 当前项目不再训练或运行 Selector。`src/selector/`、Selector 输入与输出、`Selected_Refined_Centered` 以及只为 Selector 服务的代码应退出活动代码树，历史只由 Git 保存。
2. 新主线不再构造 CLG（Candidate Lineage Group，组件谱系组）、`clg.npz` 或 `forest.npz`。连通组件仍然是阈值化概率图上的 26-连通区域，但不再为多阈值谱系关系建立森林。
3. Li 逐图阈值与 `Li_centered.npz` 退出活动主线。统一概率阈值由 calibration 数据确定。
4. 完整图概率使用无 padding 的 80³ 滑窗、`sigma=0.5` Gaussian 权重和 `stride=30` 拼接。当前生产代码仍固定 `stride=40`，因此这是待实现变化。
5. `unet_c1` 只在语义 micro-F1 最优阈值上生成基本模式的 `F1_centered`。
6. Find 系列先在语义 micro-F1 最优阈值上生成基本模式的 `F1_centered`，再在语义 micro-F3 最优阈值上生成 `F3_centered`。
7. 新版 Matcher 的正式 Stage1 输入是 Find 的 `F3_centered`，不是 `F1_centered`。`F1_centered` 只承担与 `unet_c1` 相同的基本模式评估，不进入 Matcher。
8. Find 的 `F3_centered` 需要执行居中模型前向并补齐 A、P、V 等 Matcher 所需字段；供 Matcher 读取的最终 `F3_centered` 应继承当前 Find `*_centered.npz` 的有效字段语义，并增加已经明确需要的 48³ 切块与几何字段。
9. 高斯打分保留并扩展。完整模式只用于 Find 的 `F3_centered`；基本模式用于 `unet_c1` 与 Find 的 `F1_centered`。
10. calibration 参数选择只使用两个目标函数：
    - $f_1=\mathrm{semantic\ micro\ F1}+\mathrm{coverage\ micro\ F1}+\mathrm{one\text{-}to\text{-}one\ micro\ F1$；
    - $f_2=\mathrm{semantic\ micro\ F2}+\mathrm{coverage\ micro\ F2}+\mathrm{one\text{-}to\text{-}one\ micro\ F2$。
11. Find `F3_centered` 的完整模式以 $f_2$ 选参，并同时报告 $f_1$。现有两阶段高斯参数搜索之后增加第三阶段，在前两阶段冻结的高斯参数下扫描连通组件最小体素数 `8..40`。
12. 所有语义与实例指标还要报告 macro 版本：先在单个 PDB 内计算，再对 PDB 等权平均。逐 PDB、逐预测候选的评估事实也要落盘，不能只保存总分。
13. calibration 与 validation 应采用不因预测组件数量较多而跳过 PDB 的强制模式。train 是否继续保留 `_BLOB_EXCEED` 的省时终态尚未决定。

## 2. Matcher `# NOTE` 对 Stage1 的直接要求

### 2.1 48³ 切块、几何与概率

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/geometry.py`、`voxel_input.py`、`pp_input.py`、`schema.py`。

- Stage1 应按当前 Matcher 的 V-centered 规则确定每个 Find `F3_centered` 候选的 48³ 范围。Matcher 正式代码只保留 V-centered，不再保留固定取 80³ 中心的 `fixed_center` 分支。
- Stage1 应直接保存 V 体素质心 `v_centroid_zyx`、48³ 实际裁剪位置或等价的中心偏移信息。当前 Matcher 中的 `crop_center_offset_zyx` 是候选实际裁剪中心相对 V 质心的 ZYX 偏移；最终字段名与“保存起点还是保存偏移”仍需冻结。
- Stage1 应预先保存同一 48³ 范围内的原始实验密度、受体模拟密度和 Stage1 融合概率。Matcher 不应再从完整 `exp`、`sim` 与 `probability_map` 读取后自行裁块。
- PP 是从 48³ 概率中选取的高概率体素 token。48³ 概率进入 Find `F3_centered` 后，Matcher 可以现场构造 PP，使 `top_k` 成为低成本可调参数，不必继续维护固定 `top_k` 的 PDB 级 PP 缓存。
- 用户倾向把候选所需字段保存在同一个 NPZ；是否把三张 48³ 稠密数组直接放进 `F3_centered.npz`，以及多候选时采用固定候选轴还是拼接数组，尚未冻结。

### 2.2 受体原子特征

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/voxel_input.py`。

- Matcher 最终需要 50 维受体基础特征，其中第 50 维是主链标志。
- 当前 Stage1 资产契约仍是 `receptor_tokens.npz:feat` 保存 49 维特征，`is_backbone` 保存独立 bool 数组；`Stage1Dataset` 继续返回 49 维 `atom_feat` 与独立 `atom_is_backbone`，模型输入边界按模型需要拼成 50 维。
- 当前 Find centered 归档仍保存 `A_feat_L0: float32 (L_A,49)`，没有保存独立的 A 主链标志。因此不能在实现时悄悄把既有 `A_feat_L0` 改成 50 维；应先决定是新增 `A_is_backbone: bool (L_A,)`，还是为 Matcher 定义新的 50 维字段。

### 2.3 Stage1 与 Matcher 的职责边界

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/geometry.py`、`precompute.py`、`manifest.py`。

- Stage1 负责候选切块、几何整理和可直接复用的基础数组；Matcher 负责读取、少量适配和模型输入的现场组合。
- Matcher 当前 voxel/PP 的预计算、缓存和加载链需要整体重写。Stage1 不应为了兼容将被删除的 Matcher 缓存目录而复制旧缓存结构。
- Matcher 正式 manifest 主线应收敛为两个主要能力：读取现有 manifest；从明确的 Stage1 产物根和数据根建立 manifest。一次性合法性审计与不通用诊断应移到 `ops/` 或 `tmp/`，不进入科学主线。
- 正式代码不应为了“过度防御”堆积大量 `_validate_*` 包装；外部文件读取与发布边界仍要保留足以阻止错误产物继续传播的契约检查。

## 3. Matcher `# NOTE` 中不由 Stage1 单独解决的事项

这些事项必须保存在接口对照中，但主要实施位置属于 Matcher。

### 3.1 标签命名与旁路标签产物

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/labels.py`。

- `occurrence` 是真实配体实例，`candidate` 是 Stage1 预测连通组件。两者不能继续共用 `candidate_id` 这个名称；真实配体编号应明确命名为 `occurrence_id`，预测候选编号应明确命名为 `candidate_id`。
- 标签生成应成为独立脚本：输入一组 Stage1 centered 产物，在同一 PDB 目录生成对应的 `*_label.npz`；支持“文件存在则跳过”和“强制覆盖”两种显式模式。
- 当前 `assign_candidate_target` 的双向覆盖匹配算法可以复用，但输出字段与 Docstring 必须完成 occurrence/candidate 术语纠正。

### 3.2 评估覆盖阈值

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/labels.py`。

- coverage 与 one-to-one 指标的双向覆盖阈值从当前 `(0.3, 0.5)` 扩展为 `(0.3, 0.5, 0.6)`。
- top-K 成功率至少报告覆盖阈值 `0.5` 和 `0.6`；是否继续同时报告 `0.3` 尚未明确，但保留并不改变候选定义。

### 3.3 Matcher schema 与路径简化

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/schema.py`。

- 最终正式目录用明确目录身份区分契约，不在正式代码中继续维护 `*_VERSION` 常量。
- 身份监督只保留 `all_non_background`，删除 `small_molecule` 单独分支。
- 局部裁剪只保留 `v_centroid`。
- `OCCURRENCE_TYPE_TO_CLASS` 过度抽象；五类前景保持原类别，只有 `other -> background` 需要单独处理。
- `LANGUAGE_FAMILIES`、`LANGUAGE_MODEL_NAMES` 的双层映射过度抽象；最终只保留实际需要的语言家族身份来源。
- 模态名称在面向人和产物的字段中统一使用大写 `A`、`P`、`PP`、`V`。
- 现有 `stage1_preparation_box_pool_2/ligand_language_models` 是过时路径。第三版数据根的实际配体语言模型位置必须从当前 AdaLigand 产物契约重新确认，不能复制旧常量。

### 3.4 LigandObject 图特征

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/ligand_graph.py`。

- 当前 LigandObject 图的点与边特征需要重新审计：可能缺少残基类型、主链标志等字段，并可能把中心化坐标错误地当作节点特征。
- 本地 PocketXMol 是否把中心化坐标作为节点特征只能作为对照线索，不能直接替代 Matcher 的科学决定。
- 这一项不改变 Stage1 `F3_centered` 的 blob、A/P/V 或 48³ 密度契约，但会影响 Matcher 后续的配体模态输入宽度。

### 3.5 注释的具体性

来源：`C:/Users/15919/Desktop/Matcher/matcher/data/precompute.py`。

- “清单划分”“清单候选编号”等可能有多种解释的词必须点名实体并给出例子，例如明确是 `train/validation/calibration`，还是 `9qd7/7pa9`，以及候选编号具体索引哪个 NPZ 的哪一维。
- 该写作要求适用于后续 Stage1 推理代码、字段 Docstring、README 和 Matcher 适配，不只用于 `precompute.py`。

## 4. 已核实的当前实现事实

### 4.1 Pocket Plus 当前推理实现

- `src/inference/runner.py::make_probability_role_producer` 当前固定 80³、`stride=(40,40,40)`、`sigma=0.5`。
- `src/inference/assembly.py::_TaskDatasetMaterializer.window_batch` 在推理主线程串行物化一批窗口，然后把整个 batch 搬到 GPU；正式仓库尚未把 CPU 窗口准备、H2D、GPU forward、D2H 和 NumPy 融合组成重叠流水线。
- 当前生产角色包括 `probability`、`components`、七个固定 Fα centered、`Li_centered`、`CLG_centered` 与 `Selected_Refined_Centered`。组件阶段同时发布 `forest.npz`、`clg.npz`、`overlap.npz` 和 `summary.json`。
- 当前 `F1_centered` 和其它 Fα centered 都执行一次完整 centered 模型前向，保存共同的 `voxel_final`；Find 额外保存 A/P 字段。新设计中，`unet_c1` 与 Find 的 `F1_centered` 不应继续为基本模式执行不需要的 Find A/P 前向。
- `src/inference/Gauss_Scorer/scorer.py` 当前分数为：候选平均概率 + 正 A 原子高斯项系数 × 正项 − 负 A 原子高斯项系数 × 负项；再用最终分数阈值筛选。该实现依赖 centered A 概率和 forest 节点身份。
- `ops/Gauss_Scorer/tune.py` 当前采用两阶段搜索，目标是 `semantic_dice_micro + coverage_f1_0p3 + one_to_one_f1_0p3`。它已经报告语义 macro Dice，但实例指标按跨 PDB累计计数计算，没有逐 PDB等权的实例 macro Fα。
- 当前正式 coverage/one-to-one 阈值只有 `0.3`、`0.5`，top-K 为 `K=3,4,5` 与同样两个覆盖阈值。
- 当前 CLI 的 `freeze-thresholds --min-voxels` 默认值是 `10`，2026-08-04 的既有 Find_0 正式推理也冻结为 10。新完整模式第三阶段从 8 扫描到 40，是新的契约，不是当前实现的延续。

### 4.2 Matcher 当前读取行为

- 当前 Matcher README、manifest、voxel 与 PP 读取器都硬编码读取 `centered/F1_centered.npz`。新版必须改为读取 Find `F3_centered`。
- 当前 Matcher 从 `probability/probability_map.npz` 读取完整概率图，从完整 `exp`、`sim` 读取 80³ 后再裁成 48³；这与“Stage1 预先保存候选 48³”目标不一致。
- 当前 Matcher 的 voxel 缓存是 `float16 (105,48,48,48)`：56 个密度派生通道 + 49 个受体特征硬散射通道。NOTE 中的“剩余 106 维现场构建”与现有 105 通道差 1，来源正是未来增加的主链标志；最终宽度应以 56 + 50 = 106 明确定义。
- 当前 Matcher 的 `F1_centered.npz:voxel_final` 文档固定写成 48 维，而 Pocket Plus 的通用 artifact 契约把 `C_voxel` 定义为 checkpoint 决定的宽度。新版适配必须用实际训练 checkpoint 和最终 `F3_centered` 产物核对，不能只沿用文档常数。

### 4.3 AUTO 速度参照与阈值事实

- AUTO 提交 `c69e02945727c070d30e664db48623b686fd704a` 已实现完整图流水线：16 个 CPU 窗口线程、4 个预取 batch、页锁定 H2D、GPU forward、异步 D2H，以及唯一 CPU 线程按原窗口顺序融合；最多保留两个尚未融合的概率 batch。
- AUTO 的该实现保持窗口顺序、batch 边界和 float32 Gaussian 累加顺序，但它来自较早数据代码，不能整文件复制到当前第三版 NPY/mmap Dataset。
- AUTO 的结果文档记录该流水线在 H100 上真实完成了固定 30 张 calibration map，重启后使用 window batch 16；目前没有隔离的“改造前/改造后完整图秒数”基准，因此只能确认实现可运行，不能声称已经量化了固定倍数的推理加速。
- 同一 Find_0 checkpoint、同一 30 张 calibration map 上，`sigma=0.5,stride=30` 的经典统一阈值联合分数为 `1.5027065957`，对应 Li 为 `0.9941331076`；`stride=40` 的经典统一阈值为 `1.4384185233`，Li 为 `1.0347824001`。这些事实支持当前项目删除 Li，但只代表该 checkpoint 与该 30 图集合，不应写成适用于所有未来模型的数学定理。

## 5. 实现前仍需冻结的边界

1. **Find F3 文件名**：建议采用与 `F1_centered.npz` 对称的 `F3_centered.npz`；当前路径表没有 alpha=3，只支持到 alpha=2，因此必须显式新增，而不能套用旧七层映射。
2. **基本模式归档字段**：`unet_c1/F1_centered` 与 `Find_*/F1_centered` 是否共用一份最小字段集合；基本模式是否保留 `voxel_final`，还是只保存连通组件位置、概率和必要几何。
3. **F3 两步发布**：第一步基本连通组件产物与第二步 Find 居中补充是否写同一最终文件；如果中途失败，使用临时文件后一次性发布，还是定义明确的两个完成标记。
4. **48³ 稠密数组字段**：实验密度、模拟密度、概率的字段名、dtype、候选轴布局，以及是否直接位于 `F3_centered.npz`。
5. **48³ 几何字段**：保存 `crop_start_zyx`、`crop_center_offset_zyx`，还是两者都保存；必须明确它们相对于候选 80³ BOX 的坐标系。
6. **主链标志**：建议保留 `A_feat_L0: float32 (L_A,49)` 的既有语义并新增 `A_is_backbone: bool (L_A,)`，由 Matcher 现场拼成 50 维；用户可另行决定是否直接保存新的 50 维字段。
7. **基本模式分数参数化**：平均概率系数与最终得分阈值存在尺度冗余。需要冻结一个系数为 1，实际只搜索另一个等价自由度，并明确最小体素数何时参与。
8. **完整模式第三阶段**：扫描 `8..40` 时，前两阶段是否统一使用最小体素数 8；第三阶段选定新值后是否只重算筛选与指标，不重新执行 centered 模型前向。
9. **macro 实例指标**：单个 PDB 内没有预测、没有 occurrence 或两者都为空时，各项 precision、recall、Fα 的确定值与 macro 分母规则。
10. **逐候选评估产物**：需要冻结文件名、候选身份、与 `F3_centered` 的索引关系、真实 occurrence 匹配字段，以及 train 无标签时的行为。
11. **`_BLOB_EXCEED`**：calibration/validation 已确定强制完成；train 是继续强制完成，还是允许发布可恢复的省时终态。
12. **完整概率图是否长期保存**：新管线仍需要 probability map 进行阈值校准和重跑恢复；是否在全部 centered 与评估完成后保留正式 `probability_map.npz`，尚未明确删除策略。

## 6. 后续对照规则

- 开始实现前逐项解决第 5 节会改变文件字段、指标定义或失败恢复语义的问题。
- 实现过程中每次删除 Selector、CLG、forest、Li 或旧 Fα 角色时，检查命令行、shell、测试、README、artifact 路径和 Matcher 读取器是否仍引用旧实体。
- 完成 `F3_centered` 后，用同一代表性 PDB 同时核对：80³ 来源组件、48³ V-centered 切块、A/P/V offsets、50 维主链拼接来源、概率与密度几何，以及 Matcher 实际读取结果。
- 评估实现必须同时输出 micro 总计、PDB 等权 macro 和逐 PDB/逐候选事实；不能用测试通过代替字段契约核验。
