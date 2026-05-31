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

关键代码入口通常包括：

* `src/train.py`
* `src/datasets/`
* `src/wrappers/voxel_point_stage1.py`
* `src/model/`
* `configs/model/`
* `configs/dataset/`
* `configs/loss/`
* `configs/train/`

## 4. 推理链路（有过期的风险）

推理和评估相关问题优先读：

* `src/inference/notes_of_infereval.md`
  * 当前推理/评估代码的阅读路径、检查点和未来拆分边界。
* `src/inference/`
  * 当前可用的推理、缓存、后处理、评估、可视化实现。
* `configs/infer_or_eval/`
  * 当前推理/评估配置入口。

注意：当前 `src/inference` 代码仍可用，但推理、评估、阈值搜索和可视化可能混在同一运行链路中。未来重构时建议代码层面拆分 inference 与 evaluation，组合入口可以保留。

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
  * `src/inference/notes_of_infereval.md`