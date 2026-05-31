# Stage1 网络阅读说明

若从本文直接进入项目，请先回到根目录 `CLAUDE.md` 阅读总指针、权威层级和 agent 工作规约；本文只是训练/模型链路中的子说明。

本文只做阅读路径和检查点，不试图复述完整实现。修改模型、训练 wrapper、loss、dataset 或配置时，请以实际代码、Hydra 配置、checkpoint 中保存的配置和真实 batch 为准。

如果本文与实现冲突，把冲突当作潜在 bug 或待更新文档处理，不要用本文覆盖代码判断。

## 1. 先读哪些入口

建议按下面顺序建立上下文：

1. `src/train.py`
   * 看 Hydra 如何实例化 DataModule、Wrapper、Trainer、logger 和 checkpoint。
2. `configs/model/*`
   * 看 wrapper、model、head、backbone、loss 相关配置实际指向哪些类。
3. `configs/dataset/*`
   * 看训练/验证数据根目录、split、类别配置、字段开关和增强配置。
4. `src/datasets/box_point_dataset.py`
   * 看单样本从哪些 `.npz` 文件读取字段。
5. `src/datasets/box_point_collate.py`
   * 看单样本字段如何变成 batch 字段。
6. `src/wrappers/voxel_point_stage1.py`
   * 看训练/验证 step 如何调用模型、取输出、算 loss、记 metric。
7. `src/model/stage1_model.py`
   * 看 batch 中哪些字段真正进入模型。
8. `src/model/stage1_embed_head.py`
   * 看输入特征如何嵌入，尤其是 atom/voxel/pseudo atom 相关字段。
9. `src/model/stage1_voxel_backbone.py`
   * 看体素分支输入、输出 shape 和语义。
10. `src/model/stage1_point_backbone.py`
    * 看点云分支输入、输出 shape 和语义。
11. `src/model/pseudo_atoms.py`
    * 看 pseudo atom 的生成、更新和回收机制。

## 2. 训练数据链路检查点

训练链路通常跨越以下说明文件和代码：

* `Make_Data/notes_of_dataset.md`
  * 点云级 PDB 解析、候选 ligand、atom/residue/graph/label 落盘说明。
* `Bundle_of_Maps/simulated_map/notes_of_dataset.md`
  * 模拟 receptor map 的来源、用途和训练/推理差异。
* `processedPDB_EMDB_binder/notes_of_dataset.md`
  * EMDB map、模拟 map、体素标签、ligand 距离图如何绑定并切成 BOX。
* `src/datasets/box_point_dataset.py`
  * 训练时真正读取哪些 BOX、点云和标签字段。
* `src/datasets/box_point_collate.py`
  * padding、mask、索引、shape 在 batch 维度上的实际规则。

字段名、shape、类别 ID、路径和 mask 语义都可能随代码变化。涉及这些内容时，优先抽样查看真实 `.npz` / `.json`，再核对 dataset 和 collate。

## 3. 模型与 wrapper 检查点

读模型时建议从 wrapper 反推，不要只从 `stage1_model.py` 向下猜：

* wrapper 实际传给模型的 batch 字段是什么。
* wrapper 实际读取哪些模型输出。
* loss 配置实际启用了哪些监督。
* metric 统计使用的是 logits、probability、mask 还是后处理结果。
* checkpoint 恢复时是否覆盖或携带了训练配置。

高风险字段包括但不限于：

* `emdb_exp`
* `emdb_sim`
* `voxel_label`
* `ligand_dist_map`
* `atom_pos`
* `atom_feat`
* `atom_label`
* `atom_valid_mask`
* `voxel_valid_mask`
* `hardmask`
* pseudo atom 相关字段

不要假设字段一定存在；先看 dataset/collate，再看 model forward。

## 4. Shape、坐标与类别检查点

改动前必须核对：

* 体素张量的维度顺序是 `(D, H, W)`、`(C, D, H, W)` 还是 batch 后的 `(B, C, D, H, W)`。
* 点坐标使用的是物理坐标、voxel index，还是某种归一化/局部坐标。
* batch 内 atom/pseudo atom 的 padding 和 mask 是否一致。
* 二分类和多分类配置下 logits 维度、target 维度、loss 输入是否匹配。
* 类别顺序来自配置、数据生成脚本还是代码常量。
* `voxel_valid_mask`、`hardmask`、`atom_valid_mask` 是否在 loss 和 metric 中被同样解释。

如果 shape 或类别配置有任何不确定，优先打印一个真实 batch，而不是补充猜测性文档。

## 5. Loss 与配置检查点

读 loss 时至少核对：

* `configs/loss/*`
* `configs/model/*`
* `src/wrappers/voxel_point_stage1.py`
* 模型输出 dict 的 key
* batch target 字段

重点确认：

* 哪些 loss 当前启用，哪些只是代码支持。
* 每个 loss 的 target 来源。
* 每个 loss 使用哪个 mask。
* 二分类、多分类、类别加权、ignore index 等配置是否一致。
* validation/test metric 是否与训练 loss 使用同一语义。

## 6. 修改前清单

改网络、wrapper、dataset、collate、loss 或相关配置前，至少完成：

1. 定位当前实验实际使用的 Hydra 配置。
2. 读 dataset/collate，确认 batch 字段和 shape。
3. 读 wrapper，确认模型输出如何进入 loss/metric。
4. 读 model/head/backbone，确认字段消费路径。
5. 抽样检查真实 `.npz` / `.json` 或一个真实 batch。
6. 如果改了字段、shape、类别或 mask，回头更新对应数据说明和本文。

本文保持克制：只记录阅读路线和检查点。实现细节请直接看代码。