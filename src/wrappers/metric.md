> 本文是局部说明，进入前请先读仓库根目录 `CLAUDE.md` 总入口与权威层级。
> 阅读路线补充：`src/model/notes_of_network.md` 第 8、9 节给出 wrapper 各文件职责与“CPC 指标怎么读”的高层导览，本文是它的细化版（逐个 metric 的语义 + 计算公式）。

# Stage1 wandb 验证指标详解

本文专门说明 **wandb / Lightning 日志里各项验证指标的精确语义**，并把每个指标和
[voxel_point_stage1_diagnostics.py](voxel_point_stage1_diagnostics.py) 里 `def _register_stat_buffers` 定义的
`buffer_specs` 缓存值对应起来，写清“这个指标是用哪些 buffer、按什么公式算出来的”。

指标分两大类：

- **类别一（CPC diagnostics 面板）**：全部由 `buffer_specs` 里的 histogram / count buffer 派生。
  面板有 `val_uncapped_best`、`val_uncapped_sampling`、`val_capped`、`val_unrefined`、`val_refined`、`val_score`（端到端部分），
  外加两个 histogram artifact。计算入口在
  [CpcValidationDiagnostics.compute_payload](voxel_point_stage1_diagnostics.py)。
- **类别二（不经 buffer_specs）**：AP/PRAUC 与 train/val loss。它们**与 `buffer_specs` 没有任何关系**，
  只能写语义、写不出 buffer 公式。AP/PRAUC 来自
  [voxel_point_stage1_metrics.py](voxel_point_stage1_metrics.py) 的 `ValidationMetricManager`（TorchMetrics）；
  loss 来自 [voxel_point_stage1.py](voxel_point_stage1.py) 的 `_log_loss_terms`。

> 注意：`val_score/` 这个面板名被两类指标**共用**——CPC 端到端的 `*_F1/*_p/...`（类别一）和
> `*_PRAUC`（类别二）都挂在 `val_score/` 下，但来源完全不同。详见第 5 节。

权威层级与全仓库一致：实际代码 > Hydra/checkpoint 配置 > 真实产物/日志 > 本文。若本文与代码冲突，以代码为准并视为待更新文档。

---

## 1. metric key 命名规则

所有 key 由 [build_metric_key](voxel_point_stage1_logging.py) 统一构造，格式为：

```
panel / scope / metric_leaf
```

- `panel`：顶层面板，如 `val_uncapped_best` / `val_uncapped_sampling` / `val_capped` / `val_refined` / `val_score`。
- `scope`：作用域，**当前只允许 `global`**（`_ALLOWED_SCOPES` 限制）。

> wandb 的 workspace 自动 section **按第一个 `/` 之前的段分组**，所以 `val_uncapped_best` 与 `val_uncapped_sampling` 会落到两个不同的 section；这正是把原 `val_uncapped/best`、`val_uncapped/sampling` 拆成两个顶层 panel 的原因。`subpanel` 概念已移除。
- `metric_leaf`：指标名。**二分类**直接用 metric 名（如 `F1`）；**多分类**（`num_classes > 2`）会在尾部追加 task class 名后缀，
  即 `f"{metric}_{task_class_name}"`（如 `F1_small_molecule`）。

loss 指标不走 `build_metric_key`，直接拼成 `{prefix}/global/{name}`（见第 3 节）。

> 下文除非特别说明，metric 名均按二分类写（不带 `_{task_class_name}` 后缀）。多分类时每个 candidate / task class 各产生一份。

---

## 2. 类别一：CPC diagnostics 面板（来自 `buffer_specs`）

### 2.0 通用计算约定

CPC 面板的核心是 **per candidate class 的概率 histogram**。约定：

- `T = num_bins`（`threshold_grid` 长度）；bin `i` 覆盖概率区间 `[i/T, (i+1)/T)`，`threshold_grid[i] = i/T` 是该 bin 左边界。
- score 一律是**概率**，不是 logit：二分类用 `sigmoid(logit)`，多分类用对应 candidate class 的 `softmax` 概率。
- 给定正例 histogram `pos_hist`、负例 histogram `neg_hist`（形状 `(T,)`），以“阈值取 `threshold_grid[i]`、即把概率 `>= i/T` 判正”为口径，做**后缀和**得到每个阈值下的混淆量：
  - `tp[i] = Σ_{j>=i} pos_hist[j]`
  - `fp[i] = Σ_{j>=i} neg_hist[j]`（代码用 `flip → cumsum → flip` 实现）
  - `precision[i] = tp[i] / (tp[i] + fp[i])`（分母 `clamp_min(1)`）
  - `recall[i] = tp[i] / num_gt`（分母按面板不同，见下）
  - `F1[i] = 2·P·R / (P + R)`
- **best 阈值** = `argmax_i F1[i]`；并列最大时 `torch.argmax` 取**第一个**（即更低 `p`、更高 recall 的那个）。对应截断概率 `p = threshold_grid[best_idx]`。

> recall 分母是区分 **local** 与 **端到端** 的关键：
> `val_unrefined/val_refined` 用 **C 内正例数** 做分母（local），`val_score` 用 **dense 全空间 GT 正例数** 做分母（端到端）。

---

### 2.1 `val_uncapped_best/global/*`

含义：**dense ligand 分支在全空间上的理论最优**。把 dense 全空间 valid voxel 的概率打进 histogram，遍历阈值找 best-F1，回答“如果 dense 头自己挑最佳截断概率，F1 上限是多少、该切多少候选”。

- 计算函数：[_best_f1_scalars_from_hist](voxel_point_stage1_diagnostics.py)。
- 源 buffer：`uncapped_best_pos_hist`、`uncapped_best_neg_hist`（形状 `(K,T)`，dense 全空间各概率 bin 的 GT 正/负例数）。
- 累积位置：[update_uncapped_best](voxel_point_stage1_diagnostics.py) → `_accumulate_hist_by_class`。

| metric leaf | 含义 | 公式（在 `best_idx = argmax F1`） |
|---|---|---|
| `p_best` | dense best-F1 的截断概率 | `threshold_grid[best_idx]` |
| `best_F1` | dense 全空间理论 best-F1 | `F1[best_idx]` |
| `best_precision` / `best_recall` | best 阈值下 P / R（recall 分母=`num_gt`） | `precision[best_idx]` / `recall[best_idx]` |
| `best_tp` / `best_fp` / `best_fn` | best 阈值下混淆量 | `tp[best_idx]` / `fp[best_idx]` / `num_gt - tp[best_idx]` |
| `num_gt` | dense 全空间该类 GT 正例总数 | `pos_hist.sum()` |
| `numC_p_best_cutoff` | 按 `p_best` 截断会选入的候选 voxel 数 | `tp[best_idx] + fp[best_idx]` |
| `p_sampling` | sampling 策略的目标截断概率 | `threshold_grid[sampling_idx]`（见下） |
| `numC_p_sampling_target` | sampling 策略 cap 前的目标候选数 | `ceil((tp[best_idx]+fp[best_idx]) · adaptive_expand_factor[class])` |

- `sampling_idx`：从高概率往低累计候选数 `cumulative_all`，首次 `>= numC_p_sampling_target` 的那个阈值；目标为 0 时退化为 `best_idx`，目标超过总候选数时取 `0`（最低阈值）。
- **无正例**（`num_gt <= 0`）：只产出 `best_F1 = NaN` 并附 `no_positive_gt` warning。

---

### 2.2 `val_uncapped_sampling/global/*`

含义：**真实 sampling 策略**对 dense GT 的覆盖。和 `best` 不同，这里不是理论最优阈值，而是**每个 BOX / 每个 class 实际用的 per-sample sampling boundary** 切出来的覆盖统计。回答“当前 sampling 策略打算怎样切 C、它在 dense 空间命中/漏掉多少 GT”。

- 计算函数：[_sampling_scalars_for_class](voxel_point_stage1_diagnostics.py)。
- 源 buffer：`uncapped_sampling_tp` / `uncapped_sampling_fp` / `uncapped_sampling_fn`、`uncapped_sampling_num_gt`、`uncapped_sampling_target_count`。
- 累积位置：[update_uncapped_sampling](voxel_point_stage1_diagnostics.py)。

| metric leaf | 含义 | 公式 |
|---|---|---|
| `sampling_tp` / `sampling_fp` / `sampling_fn` | sampling 边界在 dense 空间命中 / 误选 / 漏掉的 GT 量 | 直接来自 `uncapped_sampling_tp/fp/fn` |
| `sampling_precision` | `tp / (tp + fp)` | 分母 `clamp_min(1)` |
| `sampling_recall` | `tp / (tp + fn)` | 分母 `clamp_min(1)` |
| `sampling_F1` | `2·P·R / (P + R)` | — |
| `num_gt` | dense 全空间该类 GT 正例数 | `uncapped_sampling_num_gt` |
| `numC_sampling_target` | sampling cap 前目标候选数 | `uncapped_sampling_target_count` |

> `uncapped_sampling_boundary_hist` 不在这里出标量，作为 histogram artifact 输出（见 2.7）。

---

### 2.3 `val_capped/global/*`

含义：**实际进入候选集 C 之后**的覆盖统计（已经过 cap 截断）。回答“最终落进 C 的候选里有多少是 GT 正例（precision）、dense GT 正例被 C 覆盖了多少（recall）、C / P 规模多大”。

- 计算函数：[_capped_scalars_for_class](voxel_point_stage1_diagnostics.py)。
- 源 buffer：`capped_tp` / `capped_fp` / `capped_fn`、`capped_num_C`、`capped_num_P`、`capped_box_count`。
- 累积位置：[update_capped](voxel_point_stage1_diagnostics.py)。

| metric leaf | 含义 | 公式 |
|---|---|---|
| `tp` / `fp` / `fn` | 进入 C 的 GT 正例 / GT 负例 / 未进入 C 的 dense GT 正例 | `capped_tp` / `capped_fp` / `capped_fn` |
| `precision` | C 中是 GT 正例的比例 | `tp / (tp + fp)` |
| `recall` | dense GT 正例被 C 覆盖的比例 | `tp / (tp + fn)` |
| `F1` | `2·P·R / (P + R)` | — |
| `num_C` | 每个 BOX 平均候选数 | `capped_num_C / max(capped_box_count, 1)` |
| `num_P` | 每个 BOX 平均 P anchor 数 | `capped_num_P / max(capped_box_count, 1)` |

---

### 2.4 `val_unrefined/global/*`（C 内 local）

含义：**只在 C 内、用候选位置的原始 dense logit** 做判别的 local best-F1。回答“假设已经进了 C，C 内的 unrefined 排序/分类好不好”。分母是 **C 内正例数**，不含 C 外漏检。

- 计算函数：[_C_panel_scalars_from_hist](voxel_point_stage1_diagnostics.py)（local 部分）。
- 源 buffer：`unrefined_pos_hist`、`unrefined_neg_hist`。
- 累积位置：[update_unrefined](voxel_point_stage1_diagnostics.py) → `_accumulate_C_score_hist`（取 `candidate_outputs["candidate_logits"]`）。

| metric leaf | 含义 | 公式（在 `local_best_idx = argmax local_F1`） |
|---|---|---|
| `p` | C 内 local best-F1 截断概率 | `threshold_grid[local_best_idx]` |
| `F1` | C 内 local best-F1 | `local_F1[local_best_idx]` |
| `precision` | `tp / (tp + fp)` | `precision[local_best_idx]` |
| `recall` | C 内 local recall（分母=C 内正例数） | `tp / num_gt_in_C` 在 best |
| `tp` / `fp` / `fn` | best 阈值下 C 内混淆量 | `tp[idx]` / `fp[idx]` / `num_gt_in_C - tp[idx]` |
| `num_gt_in_C` | C 内该类 GT 正例总数 | `pos_hist.sum()` |

- **C 内无正例**（`num_gt_in_C <= 0`）：local 只产出 `F1 = NaN`。

---

### 2.5 `val_refined/global/*`（C 内 local）

含义：与 2.4 完全同口径的 **local** 指标，但用的是 **sparse refine 之后** 的 `ligand_refine_logits_C`。回答“refine 后 C 内判别好不好”。

- 计算函数：同 `_C_panel_scalars_from_hist`（local 部分）。
- 源 buffer：`refined_pos_hist`、`refined_neg_hist`。
- 累积位置：[update_refined](voxel_point_stage1_diagnostics.py) → `_accumulate_C_score_hist`（取 `refined_logits_C`）。
- metric leaf 与 2.4 完全一致：`p` / `F1` / `precision` / `recall` / `tp` / `fp` / `fn` / `num_gt_in_C`，只是数据源换成 refined histogram。

> `val_unrefined` 与 `val_refined` 都是 **C 内 local 最优 p**，各自服务自身，**不改语义**。它们之间的差值反映“在已进 C 的前提下，refine 对 C 内判别的改善”。

---

### 2.6 `val_score/global/{unrefined,refined}_*`（CPC 端到端）

含义：**端到端**分数。只允许 **C 内体素被判正**，C 外所有 GT 正例自动计入 FN（C 外 GT 负例不可能成为 FP）。等价于“把 C 外 score 视作低于所有候选阈值”。这正是上一轮 grill 的结论：**recall 分母换成 dense 全空间 GT**，并且 **`p` 独立地选成让端到端 F1 最大**（服务自身），不再沿用 C 内 local 的 `p`。

- 计算函数：[_C_panel_scalars_from_hist](voxel_point_stage1_diagnostics.py)（score 部分）。
- 源 buffer：
  - unrefined：`unrefined_pos_hist`、`unrefined_neg_hist`、`unrefined_dense_gt`（recall 分母）。
  - refined：`refined_pos_hist`、`refined_neg_hist`、`refined_dense_gt`（recall 分母）。

公式（`e2e_recall[i] = tp[i] / dense_gt`，`e2e_F1[i] = 2·precision[i]·e2e_recall[i] / (...)`，
`score_best_idx = argmax e2e_F1`，与 local 的 `local_best_idx` **相互独立**）：

| metric leaf | 含义 | 公式（在各自 `score_best_idx`） |
|---|---|---|
| `unrefined_p` / `refined_p` | 端到端 best-F1 的截断概率（概率本身，非 logit） | `threshold_grid[score_best_idx]` |
| `unrefined_F1` / `refined_F1` | 端到端 best-F1（dense GT 作分母） | `e2e_F1[score_best_idx]` |
| `unrefined_precision` / `refined_precision` | 端到端 best 阈值下 precision（仅由 C 内决定） | `precision[score_best_idx]` |
| `unrefined_recall` / `refined_recall` | 端到端 recall = `tp / dense_gt` | `e2e_recall[score_best_idx]` |

- `unrefined_*` 与 `refined_*` 各自基于自己的 histogram **独立选 p**，不共享阈值——这样才是同口径比较 refine 是否有帮助。
- **边界约定**（与上一轮确认一致）：
  - dense 有正例但 C 内无正例：`*_F1 = 0.0`、`*_recall = 0.0`、`*_precision = NaN`、`*_p = NaN`。
  - dense 无正例：`*_F1 / *_precision / *_recall / *_p` 全为 `NaN`。
- 默认 `monitor_metric = val_score/global/refined_F1`，所以这套语义同时也是训练监控目标。

---

### 2.7 histogram artifact（CSV/表格，非 scalar）

下面两个不是 wandb scalar，而是经 `_histogram_payload` 转成 `class_name/class_pos/class_id/bin_index/bin_left/bin_right/count` 表格、由 rank0 落 artifact / W&B curve。

| 名称 | 源 buffer | 含义 |
|---|---|---|
| `uncapped_sampling_boundary_hist` | `uncapped_sampling_boundary_hist` | 各 BOX 实际 sampling 截断概率落入各概率 bin 的次数 |
| `capped_routed_prob_hist` | `capped_routed_prob_hist` | C 内按 **routed**（路由结果，而非 GT）类别统计的 `candidate_prob` 分布；含全部 C 且不分正负例，故二分类时也不等于 `unrefined_pos_hist` |

---

## 3. 类别二：不经 `buffer_specs` 的指标

> 以下指标**与 `_register_stat_buffers` 无关**，没有对应 buffer，本节只写语义和来源模块。

### 3.1 `val_score/global/{branch}_PRAUC*`（AP / PRAUC）

含义：常规 validation 的 **平均精度 AP（≈ PRAUC）**，由 TorchMetrics `BinaryAveragePrecision` 在整个 validation 上累积概率与标签后 `compute()`。

- 来源：[ValidationMetricManager](voxel_point_stage1_metrics.py)，分支配置在
  [_build_metric_branch_specs](voxel_point_stage1.py)，每个 batch 在
  [validation_step](voxel_point_stage1.py) 里 `update_branch(...)`。
- key：二分类 `val_score/global/{branch}_PRAUC`；多分类对每个前景类 `..._{class_name}`，并额外输出
  `val_score/global/{branch}_PRAUC_macro`（各前景类均值）。
- 分支只有对应 loss 模块启用时才注册：

| `branch` | logits 源 | target / mask | 含义 | AP 阈值 |
|---|---|---|---|---|
| `atom` | `atom_logits`（后置 atom 头） | `atom_target`/`atom_label`，`atom_mask` | 结合位点 atom 级 AP | `None`（精确，list state 在 CPU） |
| `atom_front` | `atom_logits_front`（前置 atom 头） | 同上，`front_mask` | 前置 atom 头 AP | `None` |
| `receptor` | `voxel_logits_aux` | `voxel_label`，`receptor_mask` | receptor 体素 AP | `None` |
| `voxel_ligand` | `voxel_logits_ligand` | 由 `ligand_dist_map` 生成的 target，`ligand_valid` | dense ligand 体素 AP | `voxel_ligand_pr_auc_thresholds`（默认 1024，binned） |

> 这是 `val_score/` 面板里**唯一不来自 `buffer_specs`** 的部分；它与 2.6 的 `*_F1/*_p` 共享面板名但来源、口径都不同。

### 3.2 `train_loss/global/*` 与 `val_loss/global/*`

含义：训练 / 验证 loss。由 [_log_loss_terms](voxel_point_stage1.py) 记录，key 为 `{prefix}/global/{name}`，`prefix ∈ {train_loss, val_loss}`。
`train_loss` 同时记 step 级与 epoch 级；`val_loss` 只记 epoch 级。

| metric leaf | 含义 |
|---|---|
| `total` | 加权总损失 `Σ weight · term.value`（refine 项额外 `nan_to_num`） |
| `atom` | 后置 atom 头分支损失（原始未加权 `logged_value`） |
| `atom_front` | 前置 atom 头分支损失 |
| `receptor` | receptor 体素辅助分支损失 |
| `voxel_ligand` | dense ligand 体素分支损失 |
| `ligand_sparse_refine` | sparse refine 分支损失（schedule 硬 0 阶段原始值可能为 NaN，仅在 `total` 里被 `nan_to_num`） |
| `ligand_sparse_refine_weight_effective` | 当前 step refine 项的 schedule 有效权重 |
| `sparse_refine_temperature` | refine 距离 softmax 的可学正温度 `exp(log_temperature)` |

> 各分支只有启用时才出现；`total` 始终存在。每个分支 leaf 记的是**原始未加权**损失，权重只体现在 `total`。

### 3.3 `train/runtime/recycle_passes`

含义：当前 step 实际用的 recycle 轮数（`outputs["recycle_passes_used"]`）。仅训练 step 级记录，不进 epoch 聚合。

---

## 4. buffer → 指标 反查表

直接服务“`buffer_specs` 关系”：每个 buffer 喂给哪些指标。形状里 `K = num_candidate_classes`、`T = num_bins`。

| buffer（`buffer_specs`） | 形状 | 喂给的指标 |
|---|---|---|
| `threshold_grid` | `(T,)` | 所有 `p_*` / `p` 取值（bin 左边界 `i/T`），是常量，不单独记 |
| `uncapped_best_pos_hist` / `uncapped_best_neg_hist` | `(K,T)` | `val_uncapped_best/*` 全部 |
| `uncapped_sampling_tp` / `fp` / `fn` | `(K,)` | `val_uncapped_sampling/{tp,fp,fn,precision,recall,F1}` |
| `uncapped_sampling_num_gt` | `(K,)` | `val_uncapped_sampling/num_gt` 及 recall 分母 |
| `uncapped_sampling_target_count` | `(K,)` | `val_uncapped_sampling/numC_sampling_target` |
| `uncapped_sampling_boundary_hist` | `(K,T)` | artifact `uncapped_sampling_boundary_hist` |
| `capped_tp` / `fp` / `fn` | `(K,)` | `val_capped/{tp,fp,fn,precision,recall,F1}` |
| `capped_num_C` | `()` | `val_capped/num_C`（除以 box 数） |
| `capped_num_P` | `()` | `val_capped/num_P`（除以 box 数） |
| `capped_box_count` | `()` | `val_capped/num_C`、`num_P` 的除数 |
| `capped_routed_prob_hist` | `(K,T)` | artifact `capped_routed_prob_hist` |
| `unrefined_pos_hist` / `unrefined_neg_hist` | `(K,T)` | `val_unrefined/*`（local）+ `val_score/unrefined_*`（端到端） |
| `unrefined_dense_gt` | `(K,)` | `val_score/unrefined_recall` 的 dense 分母 |
| `refined_pos_hist` / `refined_neg_hist` | `(K,T)` | `val_refined/*`（local）+ `val_score/refined_*`（端到端） |
| `refined_dense_gt` | `(K,)` | `val_score/refined_recall` 的 dense 分母 |

> 同步：所有 `(K,*)` / `()` buffer 在 `compute_payload` 里先经 `sync_fn`（DDP all-reduce sum）再算标量；
> DDP 下所有 rank 必须对称调用，不能因本地无正例提前跳过。

---

## 5. 易混点提醒

1. **`val_score/` 面板混两类来源**：`*_F1/*_p/*_precision/*_recall`（CPC 端到端，类别一，来自 buffer）与
   `*_PRAUC*`（AP，类别二，来自 TorchMetrics）共享面板名，口径无关，不要互相比较。
2. **三套 F1 的 recall 分母不同**：
   - `val_unrefined/F1`、`val_refined/F1`：分母 = **C 内正例数**（local，各自选 local 最优 p）。
   - `val_score/{unrefined,refined}_F1`：分母 = **dense 全空间 GT**（端到端，各自独立选端到端最优 p）。
   - 想看 refine 是否有帮助：在 **同一口径** 下比较（local 对 local，端到端对端到端），不要跨口径减。
3. **`p` 是概率不是 logit**：二分类是 `sigmoid(logit)` 前景概率，多分类是该 candidate class 的 `softmax` 概率。
4. **并列 best 取第一个**：`torch.argmax` 在 F1 并列时取第一个最大值（更低 p、更高 recall），local 与端到端均沿用。
5. **loss 分支 leaf 是原始未加权值**，权重只体现在 `*/total`。
