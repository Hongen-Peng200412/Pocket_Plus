# Pocket_Plus agent 指针

本文是新手和 AI agent 的总入口，只做导航，不替代代码阅读。

## 1. 权威层级

遇到字段、shape、路径、类别语义、mask 语义或运行流程不一致时，按下面顺序判断：

1. 实际代码
2. Hydra 配置和 checkpoint 中保存的配置
3. 真实 `.npz` / `.json` / `.csv` 产物与运行日志
4. 本仓库 notes 文档
5. 旧探索记录或历史计划

如果文档与实现冲突，把冲突当作潜在 bug 或待更新文档处理。

## 2. AI agent 工作规约

1. 先用本文定位相关说明文档。
2. 再打开对应代码、配置和生成脚本核对。
3. 涉及字段、shape、类别、路径时，尽量抽样查看真实产物或日志。
4. 不要因为说明文档写得很详细就跳过代码检查。

## 3. 训练链路

训练相关问题优先读：

* `Make_Data/notes_of_dataset.md`
  * PDB/mmCIF 解析、候选 ligand、点云级 atom/residue/graph/label 产物。
* `Bundle_of_Maps/simulated_map/notes_of_dataset.md`
  * 训练和推理使用的 receptor 模拟密度图来源与边界。
* `processedPDB_EMDB_binder/notes_of_dataset.md`
  * EMDB map、模拟 map、体素标签、ligand 距离图和 BOX 训练样本。
* `src/model/notes_of_network.md`
  * Stage1 模型、wrapper、dataset、collate、loss 的阅读路径和检查点。
* `src/datasets/README_STAGE1.md`
  * AdaLigand Stage1 的 BOX pool、Dataset、请求比例与训练数据边界。

关键代码入口通常包括：

* `src/train.py`
* `src/datasets/`
* `src/wrappers/voxel_point_stage1.py`
* `src/model/`
* `configs/model/`
* `configs/dataset/`
* `configs/loss/`
* `configs/train/`

## 4. AdaLigand Stage1 推理、评估与 Selector

推理和评估相关问题优先读：

* `src/inference/README.md`
  * 完整图概率、阈值冻结、组件森林、CLG 与三类 centered 产物的入口和发布顺序。
* `src/evaluation/`
  * 体素、coverage、固定连续分数 Hungarian、top-K 与报告实现。
* `src/selector/README.md`
  * Selector 冻结输入、训练、校正、selection 与 Selected-refined 的边界。
* `configs/selector/`
  * Find_0、Find_1、unet_c1 三个 producer 的 Selector 配置。

旧的混合式推理、评估、保存与可视化管线不属于本分支接口；不要根据 Git 历史中的旧入口补回兼容层。

## 5. 下游分子对接链路（暂时不需要考虑）

下游 docking 相关问题优先读：

* `Ligand/notes_of_dataset.md`
  * `/storage/penghongen/CIF_Ligand` 的最终产物、字段、mapping、RCSB native mol2、SMILES 和工具输入状态。
* `Ligand/doc/explore.md`
  * 为什么选择 RCSB 单源路线。
* `Ligand/doc/implement_plan.md`
  * 正式实现计划。
* `Ligand/doc/exec_plan.md`
  * 实际执行、调试和验收记录。

## 6. 维护规则

* 子说明文件开头应保留指向本文的短提示，避免 agent 从局部文档进入后跳过总入口。
* 数据说明文件只做必要补充，不做删减和大幅重写：
  * `Make_Data/notes_of_dataset.md`
  * `Bundle_of_Maps/simulated_map/notes_of_dataset.md`
  * `processedPDB_EMDB_binder/notes_of_dataset.md`
  * `Ligand/notes_of_dataset.md`
* 代码链路说明允许随实现演进重写，但应保持克制，只写阅读路径和检查点：
  * `src/model/notes_of_network.md`
  * `src/datasets/README_STAGE1.md`
  * `src/inference/README.md`
  * `src/selector/README.md`
