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

* `ops/stage1_data_preparation/README.md`
  * 完整体数组 NPY 迁移、V3 split 与 0:5:5 BOX pool 的正式路径和执行证据。
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

## 4. AdaLigand Stage1 V3 推理与评估

推理和评估相关问题优先读：

* `src/inference/README.md`
  * 完整图概率、F1/F3 blobs、F1 basic、F3 centered、校准、评估和跨 PDB 有界流水。
* `configs/inference/README.md`
  * 四个 producer 共用的显式科学与并行配置。
* `训练与运行/sh/infer/README.md`
  * `calibrate`/`run` 命令、正式产物路径和提交方式。
* `talk/stage1_v3_inference_inputs_outputs.md`
  * Stage1 V3 输入与数组字段的学习型概览。

旧的混合式推理、评估、保存与可视化管线不属于本分支接口；不要根据 Git 历史中的旧入口补回兼容层。

## 5. 下游分子对接链路（暂时不需要考虑）

下游 docking 相关问题优先读：

* `Ligand/`
  * 下游配体资产仍在活动目录中；需要进入该时间节点时再按实际文件读取。

## 6. 维护规则

* 子说明文件开头应保留指向本文的短提示，避免 agent 从局部文档进入后跳过总入口。
* 已退出主线的 `Make_Data/`、`Bundle_of_Maps/`、`processedPDB_EMDB_binder/` 与 `Docking/` 只通过 Git 历史阅读，不恢复为活动入口。
* 代码链路说明允许随实现演进重写，但应保持克制，只写阅读路径和检查点：
  * `src/model/notes_of_network.md`
  * `src/datasets/README_STAGE1.md`
  * `src/inference/README.md`
  * `src/selector/README.md`
