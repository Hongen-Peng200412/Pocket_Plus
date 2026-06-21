# Phase 1 handoff：删 front/back + 死代码 + mask 裁边族

目标见 `最后重构代码_v3.md` §3。本文件记录调用点清单、关键等价性、逐文件计划与剩余工作。

## 三块删除目标

A. **front/back 双分类头**：`enable_atom_head_front` / `enable_atom_head_back`、`atom_logits_front` / `pseudo_logits_front`、`pseudo_logit_head_front` / `atom_logit_head_front`、`*_loss_front_weight` / `*_loss_back_weight`。改成只剩一套 real/pseudo 头（结构由 Phase 2 建，Phase 1 先把 front/back 拆掉）。

B. **stage1_atom_head.py 死代码**：`Stage1SerializedAttentionStack`、`PseudoGeometricAggregation`(V1)、`append_coord_mask`/`atom_tokens`、`density_residual`/`pseudo_density_residual`。注：`PseudoGeometricAggregationV2` + `pseudo_geo_head` 的实现思想留给 Phase 2 改造成 `GeometricCrossAttention`，Phase 1 不删 V2。stage1_atom_head.py 实际会在 Phase 2 整体重写，Phase 1 只先摘掉它在 stage1_model 里的 front/back 构造与调用。

C. **mask 裁边族**：`valid_crop_margin`、`build_atom_valid_mask`、`build_voxel_valid_mask`、`voxel_valid_mask`、`atom_valid_mask`。

## 关键等价性（让 C 在主线 numerically 无变化）

当前所有现役 dataset 配置 `valid_crop_margin: 0`。在 margin=0 下：

- `build_atom_valid_mask(..., margin<=0)` 直接 `return atom_is_in_core_box`。⇒ **`atom_valid_mask ≡ atom_is_in_core_box`**。所以"原子监督用 atom_is_in_core_box"是行为保持的改写。
- `build_voxel_valid_mask(..., margin<=0)` 返回全 True。⇒ **`voxel_valid_mask ≡ 全 True`**。所以体素 loss 去掉 valid_mask（用整张 BOX）行为保持；`ligand_refine_valid_mask_C`（由 voxel_valid_mask 派生）变成全 True。

⇒ Phase 1-C 在主线是机械改写、不改数值；风险是"漏改某个 consumer 导致 crash/字段缺失"，不是训练数值漂移。**服务器必须跑测试确认没漏。**

## Mask consumers 清单（已 grep 核对）

- 生产：`datasets/box_geometry.py`(build 两函数)、`datasets/box_sample_builder.py`、`datasets/box_point_dataset.py`(导入+旋转+输出)。
- collate：`datasets/box_point_collate.py`(字段列表含 voxel_valid_mask/atom_valid_mask)。
- voxel_valid_mask 消费：`wrappers/voxel_point_stage1.py`(`_normalize_voxel_valid_mask`、voxel ligand/receptor loss、`_sample_ligand_refine_supervision` 派生 `ligand_refine_valid_mask_C`、receptor_mask 853/859)、`wrappers/voxel_point_stage1_losses.py`(compute_voxel_* 的 valid_mask 174/180/218/224)。
- atom_valid_mask 消费：`wrappers/voxel_point_stage1_losses.py`(atom loss hardmask=atom_valid_mask 81/96)、`wrappers/voxel_point_stage1.py`(845/849 是 front/back 分支)、`model/stage1_model.py`(透传 1516/1773/1781/1788/2044)、`model/stage1_atom_head.py`(append_coord_mask 用，随 B 删)。
- 配置：~90 个 .yaml 里有 `valid_crop_margin`（dataset/* 与大量 infer_or_eval/*；多数是 `: 0` 或 `: null`）。inference 侧字段先保留兼容直到 inference 适配（见下）。
- inference：`inference/parse_input.py`、`get_pred.py`、`main/voxel_pipeline.py` 也引用 mask/valid_crop_margin（推理链路，单独处理，注意 §推理）。

## front/back consumers（已 grep 核对，18 文件）

src: `model/stage1_atom_head.py`、`model/stage1_model.py`、`wrappers/voxel_point_stage1.py`、`wrappers/voxel_point_stage1_losses.py`、`wrappers/voxel_point_stage1_metrics.py`。
configs: `experiment/CPC/CPC_main.yaml`、`CPC/CPC_A2_no_pseudo.yaml`、`experiment/other/MINI_*`、`loss/sparse_refine.yaml`、`loss/other/tri_*`、`model/default.yaml`、`model/detach_residual/max_detach.yaml`。
tests: `tests/model/test_stage1_atom_head.py`、`test_stage1_model.py`（重写）。

## 逐文件计划与状态

- [x] `datasets/box_geometry.py`：已删 `build_atom_valid_mask`、`build_voxel_valid_mask`（grep 0 残留）。
- [x] `datasets/box_sample_builder.py`：已删 `valid_crop_margin` 形参、两处 build 调用、两个输出字段、`_TENSOR_DTYPE_MAP` 两项、import（grep 0 残留）。`atom_is_in_core_box` 文档已注明"也是原子级损失监督的唯一判据"。
- [x] `datasets/box_point_dataset.py`：已删 import、`__init__` 形参/赋值/校验/docstring、`_rotate_voxel_arrays`(4→3 数组)及调用解包、旋转后 atom_valid_mask 重算、输出 docstring 条目、唯一一处 `build_box_point_numpy_sample` 的 `valid_crop_margin=` 实参（grep 0 残留）。注：实际只有 1 处 build_box_point_numpy_sample 调用（另一处旧 869 行是 build_atom_valid_mask，随 mask 删）。
- [x] `datasets/box_point_collate.py`：已从 `_VOXEL_STACK_FIELDS` 删 voxel_valid_mask、`_ATOM_CONCAT_FIELDS` 删 atom_valid_mask、删相关 docstring/内联注释（grep 0 残留）。

> 至此 **dataset 层 mask 已删干净**：batch dict 不再含 `voxel_valid_mask` / `atom_valid_mask`；原子监督唯一判据 = `atom_is_in_core_box`；体素全监督无 mask。

### 剩余高风险核心（未做，需谨慎 + 服务器跑测试）

下面三件在 `stage1_model.py`(2150 行) / `stage1_atom_head.py` / 三个 wrapper 文件里**互相缠绕**（front/back 与 mask consumer 与死代码在同一批行），且改的是 **loss / refine 监督路径**，本机无法验证：
- [ ] `datasets/box_point_collate.py`：从字段列表与文档删两个 mask。
- [ ] `wrappers/voxel_point_stage1_losses.py`：voxel loss 去掉 valid_mask（整张 BOX）；atom loss `hardmask=atom_valid_mask` → 用 `atom_is_in_core_box`（real-only 顺序）。
- [ ] `wrappers/voxel_point_stage1.py`：删 `_normalize_voxel_valid_mask`，voxel loss / refine 监督 / receptor_mask 不再用 voxel_valid_mask（全 True）；`ligand_refine_valid_mask_C` 改为按 candidate 数全 True。front/back 分支随 A 删。
- [ ] `model/stage1_model.py`：删 front/back 构造与 front 头调用；atom_valid_mask 透传改名/随 B 调整（与 Phase 2 协同）。
- [ ] configs：删 dataset/* 与 experiment/* 的 `valid_crop_margin` 行；front/back 开关行。infer_or_eval/* 的 `valid_crop_margin` 留到推理适配阶段。
- [ ] tests：`test_stage1_atom_head.py`、`test_stage1_model.py` 等按新结构重写（Phase 2 后一起定稿）。

## 推理链路注意

`src/inference/` 仍在用 mask 与 valid_crop_margin；推理链路有过期风险（见 CLAUDE.md §4）。本轮以训练链路为主：训练侧删干净后，inference 侧同步去 mask；若 inference 适配工作量大，可在 handoff 标注"inference 待适配"，不要为了 inference 在训练主线保留 mask。

## 服务器必跑测试（信任前）

`pytest tests/datasets/ tests/model/test_stage1_model.py tests/model/test_stage1_atom_head.py tests/test_ligand_sparse_refine_loss.py -x`，以及一次 MINI 配置的 1-2 step 烟雾训练。

## 剩余工作

（执行中实时更新此节）当前：清单与等价性已定，尚未落任何代码 edit。
