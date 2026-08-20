# Stage1 V3 推理重写 NOTE 清单

本文把 `talk/global/global_8.19.md`、Matcher `matcher/data/` 中的 `# NOTE` 与 2026-08-20 已确认决定整理为当前实现约束。旧实现事实只用于解释删除范围, 不定义新版行为。

## 当前主线

1. Stage1 只保留完整图概率、单阈值 26-连通区域、`F1_basic`、`F3_centered`、两类评分和评估。Selector、CLG、组件森林、Li 阈值、旧七种 Fα centered 与 Selected 产物退出活动代码树, 历史只由 Git 保存。
2. `unet_c1`、`Find_0`、`Find_1`、`Find_2` 都支持基本版和完整版:
   - 基本版使用语义 micro-F1 冻结的体素阈值, 写 `F1_blobs.npz` 与 `F1_basic.npz`, 评分参数用 $f_1$ 选择。
   - 完整版使用语义 micro-F3 冻结的体素阈值, 写 `F3_blobs.npz` 与 `F3_centered.npz`, 评分参数用 $f_2$ 选择并同时报告 $f_1$。
3. Matcher 正式读取各模型的 `F3_centered.npz`。`F1_basic.npz` 是便捷推理与消融产物, 正式命令显式关闭 `voxel_final` 和 48³ 稠密数组。
4. Find 完整版使用 A 原子概率高斯项。U-Net 没有 A/P 表, 完整版分数退化为 `source_probability_mean`, 仍以 $f_2$ 联合选择分数阈值和最小体素数。
5. 所有模型的配体概率都不乘受体 hardmask。`hardmask` 只保留训练输入、辅助监督和 `voxel_aux` 稀疏索引语义。
6. 滑窗 `stride`、`sigma`、`save_voxel_final`、`save_dense48`、`enforce_blob_limit` 等会改变行为的参数必须由命令或 YAML 显式提供, Python 不设置隐式默认值。正式示例显式采用建议值 `stride=50` 与 `sigma=0.5`。
7. 连通区域第一份产物不按体素数删除候选。它保存阈值下的全部 26-连通区域; `min_voxels` 只在评分和 centered 重跑之前生效。
8. 完整模式第三阶段在前两阶段冻结的参数下扫描 `min_voxels=8..40`。小于最终最小体素数的连通区域保留在 `F3_blobs.npz`, 但不进入 `F3_centered.npz`。
9. `_BLOB_EXCEED` 只属于 F3 完整模式。候选数上限和是否强制终止都由命令显式传入; 未启用强制终止时可以写标记并继续生产。

## 评分与评估

两个参数选择目标固定为:

$$
f_1=\mathrm{semantic\ micro\ F1}+\mathrm{coverage\ micro\ F1}_{0.3}+\mathrm{one\text{-}to\text{-}one\ micro\ F1}_{0.3}
$$

$$
f_2=\mathrm{semantic\ micro\ F2}+\mathrm{coverage\ micro\ F2}_{0.3}+\mathrm{one\text{-}to\text{-}one\ micro\ F2}_{0.3}
$$

- 基本分数固定为 `source_probability_mean`, 不再引入与最终阈值尺度冗余的平均概率系数。
- Find 完整分数为来源概率均值加 A 原子正高斯项减 A 原子负高斯项。U-Net 的 A 原子项缺席且按 0 处理。
- 双向 coverage 与固定 Hungarian 一对一指标报告阈值 `0.3`、`0.5`、`0.6`。top-K 报告 `K=3,4,5` 与相同三个 coverage 阈值。
- micro 指标先跨 PDB 汇总分子和分母。macro 指标先在每个 PDB 内计算, 再对 PDB 等权平均。空分母值固定为 `0.0`。
- 每个 PDB 保存完整预测候选与真实 occurrence 的交集计数、候选体素数、真实 occurrence 体素数、匹配与 top-K 事实; 汇总 JSON 不能取代这些基础数组。

## 产物字段

每个 producer、数据划分和 PDB 的正式目录只包含以下角色:

~~~text
<output_root>/<stage1_model_name>/<split>/<pdb_id>/
├── probability/probability_map.npz
├── blobs/F1_blobs.npz
├── blobs/F3_blobs.npz
├── centered/F1_basic.npz
├── centered/F3_centered.npz
└── status/<role>/_COMPLETE
~~~

- `probability_map.npz` 保存有限 `float32 (D,H,W)` 概率、完整图世界 XYZ 原点和 XYZ 体素尺寸。概率融合只使用完整 80³ 窗口与 float32 Gaussian 权重。
- `F1_blobs.npz` 与 `F3_blobs.npz` 保存全部连通区域的全图 ZYX 体素坐标、来源概率、平均概率、稳定 `blob_index` 和是否能完整放入 80³ centered BOX。
- centered 文件只保存能够完整放入 80³ BOX 且达到显式 `min_voxels` 的候选。`source_blob_index` 精确指向对应 blobs 文件的 `blob_index`。
- `source_probability` 来自完整图融合; `centered_probability` 来自当前 80³ BOX 的重新前向, 两者不能互相替代。
- 所有 F3 正式命令显式保存 `voxel_final` 和 V-centered 48³ 数组。所有 F1 正式命令显式关闭两组字段。
- 48³ 数组固定为 `experimental_density_48`、`simulated_density_48`、`source_probability_48`, 使用 `float32 (N_candidate,48,48,48)`。持久化密度不降为 float16; GPU 计算可以使用显式混合精度。
- V-centered 起点采用 `rint(mean(V_zyx)+0.5-24)` 后逐轴限制到 `[0,32]`。产物同时保存 `v_centroid_local_zyx`、`crop_start_local_zyx`、`crop_center_offset_zyx` 和 `crop_clipped_axis_mask`。
- `voxel_final` 与 Find A/P 学习特征使用 float16; 概率、密度和几何使用 float32。`A_feat_L0` 是运行时 49 维基础特征与主链标志拼成的 `float32 (L_A,50)`。
- `unet_c1` 不得出现 A/P 字段。Find 的 F3 必须保存同序 A/P 字段; Find 的 F1 基本版不执行点分支, 因而不保存 A/P。

## CPU 与 GPU 并行

1. 完整图阶段并行执行 CPU 窗口物化、页锁定 H2D、GPU voxel-only 前向、异步 D2H 和唯一有序 CPU 融合。
2. GPU 开始下一个 PDB 的完整图时, CPU 进程并行读取已经原子发布的概率图并生成两个 blobs 文件。
3. centered 阶段并行执行 CPU 请求物化、H2D、GPU centered 前向、D2H、CPU 字段打包和原子发布。单 GPU 只有一个明确调度者, 不由多个进程争抢显存。
4. 队列容量、CPU worker 数、GPU batch size 和精度均由配置显式给出。并行只改变执行重叠, 不改变 PDB 顺序、候选顺序、融合累加顺序和产物数值定义。
5. GPU 利用率基准同时保存串行与流水线吞吐、GPU 活跃时间比例、`nvidia-smi` 利用率分布和各队列等待时间。首次验收记录证据, 不凭空设定固定百分比门槛。

## Matcher NOTE 的职责分界

- Stage1 负责候选 80³ 几何、V-centered 48³ 切块、A/P/V 基础数组和完整概率。Matcher 负责读取、少量现场组合与自身模型输入, 不要求 Stage1 复制 Matcher 将被删除的缓存目录。
- Stage1 不修改 Matcher 的 manifest、标签生成、LigandObject 图特征或语言模型路径。这些 NOTE 继续由 Matcher 仓库处理。
- occurrence 是真实配体实例, blob/candidate 是 Stage1 预测连通区域。代码、注释和文档必须点名二者, 不能继续共用含义不清的 `candidate_id`。
- 正式主线只做阻止错误产物继续传播所必需的边界检查。一次性合法性审计与不通用诊断放到 `ops/` 或 `tmp/`, 不建立大量 `_validate_*` 包装。

## 实施对照

- 重写时不恢复删除提交 `3cbae636f444fb6630eb505d210d7ce0586b4f76` 中的旧文件。参考实现只能通过 Git 读取。
- 每次删除旧 shell、Selector 配置或测试时, 同步检查 import、README、Hydra 配置和 AdaLigand BOX 契约中的旧引用。
- 完成 F3 后至少用一个代表性 PDB 逐字段核对: blobs 到 centered 的索引关系、80³/48³ 坐标、A/P/V offsets、50 维 A 基础特征、三张 48³ 数组和 Matcher 读取需要的字段。
- 本清单保存当前决定; 作业编号、逐次故障和性能采样进入执行记录, 不写回当前契约。
