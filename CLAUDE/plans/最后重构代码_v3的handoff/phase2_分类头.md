# Phase 2 handoff：统一 cross-attn + 轻量分类头

## 已完成（静态、未在服务器跑）

`src/model/stage1_atom_head.py` 已**整体重写**为新结构，旧文件全部清掉（无 Stage1SerializedAttentionStack / PseudoGeometricAggregation V1 / V2 / append_coord_mask / atom_tokens / density_residual / typed token proj / PTV3 Block 依赖）。新文件只有两个类，已逐行核对：零初始化 `output_proj`、纯残差、执行顺序无环、real-only 安全、prior 初始化保留。

### 新 API（`stage1_model.py` 接线必须照这个）

```python
class GeometricCrossAttention(nn.Module):
    def __init__(self, query_channels, source_channels, num_heads, radius,
                 max_neighbors, detach_source_feat, act_layer, use_source_bind_prob=False)
    def forward(self, *, query_feat, source_feat, query_coord, source_coord,
                query_batch, source_batch, source_bind_prob=None) -> Tensor(N_query, query_channels)  # 零初始化增量

class Stage1AtomHead(nn.Module):
    def __init__(self, point_channels, hidden_dim, atom_logit_dim, pseudo_ligand_logit_dim,
                 act_layer, interaction_radius, interaction_max_neighbors, interaction_num_heads,
                 interaction_detach_source_feat, enable_real_to_pseudo, enable_pseudo_to_real,
                 prior_prob=None, prior_probs=None, prior_prob_point_ligand=None)
    def forward(self, point_feat, point_state, atom_coord_centered_world, pseudo_mask=None)
        -> dict  # 6 键: real_feat_before_interaction / real_feat_after_interaction /
                 #       pseudo_feat_before_interaction / pseudo_feat_after_interaction /
                 #       atom_logits / pseudo_logits（real-only 时 pseudo_* 为 None）
```

- forward 内部按 pseudo_mask 切 real/P，双向 cross-attn 纯残差加回；零初始化/冻结 ⇒ `*_after == *_before`。
- `atom_logits` 恒产出（不再有 `enable_atom_head_back` 开关）；如旧逻辑需要"关 real 头"，本头不再提供，需在 stage1_model 接线层决定（建议：不再需要，直接恒产）。

## 未完成：`stage1_model.py` 接线（高风险，需服务器测试）

`stage1_model.py`(2150 行) 仍在构造旧 `Stage1AtomHead`（旧签名）、跑前置头、产旧 outputs。要改：

1. **构造**：把旧 `Stage1AtomHead(...)` 调用换成新签名。删所有旧形参（num_layers/patch_size/serialization_orders/shuffle_orders/qkv*/append_coord_mask/atom_head_ffn_type/cpe*/enable_atom_head_back/pseudo_feature_dim/pseudo_density_*/enable_pseudo_geo_head/pseudo_geo_*/concat_receptor_base_logit/typed_point_cfg 等）。新形参里 `interaction_*` 来自配置（见 §2.2 推荐值：radius 4.0 / max_neighbors 48 / num_heads 4 / detach true），`enable_real_to_pseudo` / `enable_pseudo_to_real` 来自配置；`prior_*` 沿用现有 prior schema。
2. **`_run_atom_head`**：调用 `self.atom_head(point_feat=<最后一轮 mixed point feat>, point_state=..., atom_coord_centered_world=..., pseudo_mask=...)`。把返回 6 键写进 outputs；**删前置头** `atom_logit_head_front` / `pseudo_logit_head_front` 的构造与调用、删 `atom_logits_front` / `pseudo_logits_front` 输出；删 `atom_tokens` / `atom_hidden` 输出；旧 `pseudo_feature` 输出改名 `pseudo_feat_before_interaction`（并新增 after 与 real before/after）。删 `atom_valid_mask` 透传（监督判据移到 wrapper 用 `atom_is_in_core_box`，见 phase1）。
3. **`_run_sparse_refine_head`**：P 特征源由旧 `pseudo_feature` 改为 `pseudo_feat_before_interaction`（默认）+ `pseudo_feat_after_interaction`（Phase 3 的 after 开关）。构造注入维度 `P_atom_head_dim` → `P_final_point_dim`（=point_channels）等（与 Phase 3 协同）。
4. **删旧约束**："启用 sparse_refine 必须启用 atom head" 改为 "必须存在 final P before 特征"。

> 顺序建议：先做 stage1_model 接线（本节）→ 再做 Phase 3 sparse_refine 改名（两者在 _run_sparse_refine_head 交汇）→ 再做 wrapper/loss 的 front/back+mask 清理（Phase 1 剩余）。三者都改完后，全跑 `pytest tests/model/test_stage1_model.py tests/model/test_stage1_atom_head.py -x` + MINI 烟雾训练。

## 配置侧（Phase 5 时落）
atom_head 配置组要从旧的一堆 attention/PTV3 参数，改成 `interaction_radius/max_neighbors/num_heads/detach` + `enable_real_to_pseudo/enable_pseudo_to_real`。
