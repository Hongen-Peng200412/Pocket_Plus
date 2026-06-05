# 推理与评估阅读说明

若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；本文只是推理/评估链路中的子说明。

本文只做 `src/inference` 的阅读路径和检查点，不复述完整实现。当前推理、评估、阈值搜索、缓存和可视化代码仍可用；如果本文与代码或配置冲突，以实际代码、配置、预测缓存、评估输出和日志为准。

## 1. 当前阅读入口

建议按下面顺序读当前实现：

1. `configs/infer_or_eval/*`
   - 先确认当前运行使用哪个配置、哪些 override、哪个 checkpoint。
2. `src/inference/main/run.py` 和 `src/inference/main/two_stage_basic.py`
   - 看当前命令入口如何组织推理、缓存、后处理、评估和可视化。
3. `src/inference/get_pred.py`
   - 看 checkpoint 加载、训练配置恢复、模型调用和预测输出。
4. `src/inference/parse_input.py`
   - 看 raw CIF/map 输入如何变成模型可读的体素和点云输入。
5. `src/inference/voxel_postprocess.py`
   - 看 voxel probability 如何变成候选区域或实例。
6. `src/inference/voxel_tuning.py`
   - 看阈值搜索、参数搜索和缓存评估逻辑。
7. `src/inference/voxel_gt.py`
   - 看评估 GT 如何从结构、labels 或其它来源构造。
8. `src/inference/voxel_evaluator.py`
   - 看 voxel/instance metric 的实际定义。
9. `src/inference/utils/`
   - 看 raw pair JSON、可视化 bundle、Excel/报告等辅助产物。

## 2. 推理检查点

排查推理问题时优先核对：

- 输入 pair JSON 是否对应正确的 CIF/map。
- 推理使用的 receptor 结构来源是真实结构、cryoatom 结构还是其它结构。
- 模拟 receptor map 的来源是否与当前场景匹配。
- checkpoint 对应的训练配置是否和当前输入字段、类别数、shape 兼容。
- raw 输入解析后的 voxel grid、origin、spacing、axis order 是否与模型期望一致。
- 滑窗、padding、box size、stride 和回填逻辑是否保持同一坐标系。
- 预测缓存是否来自当前 checkpoint、当前配置和当前后处理前版本。

涉及 shape、坐标、类别和缓存时，不要只读文档；请抽样打开实际缓存或中间 `.npz`。

## 3. 评估检查点

排查评估问题时优先核对：

- `eval_gt` 是否启用。
- `structure_input_source`、`sim_map_source`、`gt_receptor_source` 是否明确指向本轮应使用的 JSON 字段。
- GT 来自结构、labels 还是其它缓存；structure GT 的 ligand 固定来自 `cif_gt_path`。
- metric 输入是原始 probability、threshold 后 mask、后处理实例，还是按类别拆开的结果。
- 阈值搜索的 objective 是否只用于评估/校准，不要误认为普通推理必需。
- 可视化和报告字段是否来自同一次预测缓存和同一套后处理参数。

## 4. 未来重构边界

后续 `src/inference` 预计会大幅重构。推荐代码层面把 inference 和 evaluation 解耦：

- inference 负责输入解析、模型加载、体素 ligand 粗预测、伪原子初始化、点云 refinement、预测产物落盘。
- evaluation 负责读取预测产物和 GT，计算 voxel、instance、atom、ligand、receptor 等指标。
- 可以保留组合入口，用于一次性执行推理再评估。

尤其是未来如果使用体素分支 ligand 粗预测初始化点云分支伪原子，推理本身会成为两阶段或多阶段链路；评估、GT、阈值搜索和 metric objective 不应嵌入核心推理流程。

本文只记录阅读路线和边界提醒。具体重构方案应另写 implement plan。
