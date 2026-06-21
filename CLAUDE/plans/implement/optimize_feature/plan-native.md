# 特征/梯度通路优化 —— native 版（已降级为指针）

> 本文原为 claude 原生撰写的 file-by-file 清单，现**降级为指针**：完整且权威的规格以同目录 [plan-implement-plan-writer.md](plan-implement-plan-writer.md) 为准。本文不再整份复制正文，只保留"native 视角的差异/补充"，以免双写漂移（背景见权威版顶部 NOTE）。

## 与权威版的关系

- 设计决策、已有可复用代码、file-by-file 改动、配置改动、验证关卡：**全部见** [plan-implement-plan-writer.md](plan-implement-plan-writer.md)。
- 五点范围一致：
  1. 拆 `detach_voxel_feat_into_point` → 真实/伪两开关。
  2a. SparseRefineHead `mode` 与 `use_voxel_logits` 解耦(删耦合约束)。
  2b. atom 前/后置头新增 `atom_head_concat_receptor_base_logit`(与末尾加残差正交)。
  3. 真实原子 density cube + 零初始化 combine。
  5. voxel→point 采样新增 `weighted_cube`/`cube_mean` + per-BOX 循环向量化。

## native 视角差异 / 补充（仅记不同处）

- 暂无与权威版的实质差异。
- 若后续实现中 native 落地顺序、代码组织或命名与权威版分歧，只在此处**追加差异条目**，不重写正文，也不回填权威版细节。
