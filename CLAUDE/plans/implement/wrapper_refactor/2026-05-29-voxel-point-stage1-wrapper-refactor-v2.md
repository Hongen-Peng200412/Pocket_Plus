# VoxelPointStage1Wrapper 分层重构与 CPC 诊断计划 v2

## 背景与目标

`src/wrappers/voxel_point_stage1.py` 当前把训练封装、loss、验证指标、C/P 候选统计、阈值缓存、W\&B 记录、checkpoint metadata 和 scheduler 管理都放在一个大文件里。它能运行，但阅读路径太长，初学者很难按“先理解训练流程，再理解诊断细节”的顺序进入代码。

本次重构的目标不是改变模型主干，也不是重写 sparse refine 结构，而是把 wrapper 拆成清晰层次，并把验证期诊断变成可解释、可追踪、可在 W\&B 中阅读的体系。尤其要解决二分类下重复记录 `*_macro`、`aux` 命名不直观、P-C 统计散乱、p\_sampling 异常难排查等问题。

用户已确认的关键方向：

* 保留 `VoxelPointStage1Wrapper` 作为唯一 Hydra / Lightning 入口。
* 拆出同目录 helper 模块承接 loss、metrics、diagnostics、logging、scheduler 等职责。
* 日志彻底改成 `train_loss/*`、`val_loss/*`、`val_score/*`、`val_pc/*`、`val_curve/*`，不保留旧 `val/*` 别名。
* 业务语义上把 `voxel_aux` 对外改称 `receptor`，但不改 backbone 内部权重 key 和输出 key，旧 checkpoint 推理仍应可用。
* 二分类不再记录 macro；只有前景类数量大于 1 时才记录 macro。
* PR 曲线、阈值表和 C/P 诊断服务于 debug，不做 box 级或 voxel 级明细。
* 计划书本身重点解释实际意义和实现边界，不陷入代码细节。

## AI 建议的采纳结论

### 1. TorchMetrics 不能藏在普通 Python 类里

采纳。`BinaryAveragePrecision` 等 torchmetrics 对象必须被 PyTorch 模块树看见。新版计划要求 metrics manager 继承 `nn.Module`，或把 metrics 放在 wrapper 的直接属性 / `ModuleDict` 中。普通 Python helper 只能管理命名映射、配置和纯数值，不持有 torchmetrics 状态。

实际意义：避免 Lightning 设备迁移、DDP、checkpoint 和状态管理出现隐性不一致。

:::comment{#comment-1780030767119 text="请注意，原始数据集的所谓__DQUOTE__类别标签__DQUOTE__和真实的标签类别是不同的，中间还差着  class_mapping 以及特例random_BOX这一类。我建议，如果某类__DQUOTE__类别标签__DQUOTE__不存在就fail-fast"}
2\. 诊断统计不应因为无正例中断训练
采纳并调整。验证诊断模块在 DDP 全局汇总后，如果某个类别没有 GT 正例，应跳过该类别的有效指标、写 warning/skip 标记、重置本轮 buffer，并继续训练。只有配置错误、shape 错误、checkpoint cache 不一致等结构性问题才 fail-fast。

实际意义：诊断模块应该帮助理解训练，而不是在小验证集或类别稀疏时把训练炸掉。
:::

### 3. `class_folder_names` 分组必须静态 Tensor 化

采纳。按 folder 分组时从配置显式传入 `group_names`，构造固定的 `group_to_idx`，所有统计 buffer 固定形状，例如全局 `(K, bins)`，分组 `(G, K, bins)`。DDP 汇总只汇总形状一致的 tensor。

实际意义：保证不同 rank 即使看到的 folder 不同，也能安全汇总。

### 4. W\&B PR 曲线要稳定 key，不制造散乱表格

采纳并结合用户需求调整。标量指标用 W\&B 折线图；PR 曲线用稳定 key 按 step/epoch 追加，供 W\&B UI 用 `>` 翻阅历史。每个 `class_name` 单独一栏，全局不分组曲线也保留独立栏。

实际意义：W\&B 页面既能看趋势，也能翻阅每次验证的 PR 曲线，不会出现几十个不同 key 的零散表格。

## 当前约束

* 推理路径不依赖 W\&B 日志名。`src/inference/get_pred.py` 从 checkpoint 中取 `backbone.*` 权重并 `strict=True` 加载到 backbone。
* `receptor` 推理头当前映射到 `voxel_logits_aux`。因此对外日志可改成 receptor，内部输出 key 暂不改。
* `BoxPointDataset` 已把 `class_name` 写入样本；`box_point_collate.py` 会把它保留成 batch 中的 `class_name: list[str]`。
* `configs/dataset/emb_unet.yaml` 中已有 `class_folder_names`，未来可能新增 `random_BOX2` 等 folder。
* `src/train.py` 使用 `cfg.model.monitor_metric` 做 checkpoint 监控。日志名彻底改掉后，当前训练配置必须同步更新 monitor。

## 总体设计

### 1. Wrapper 变成编排层

`voxel_point_stage1.py` 保留入口类，但只负责按时间顺序组织训练：

1. 初始化 backbone、loss、metrics、diagnostics、scheduler。
2. `training_step` 前向、算 loss、写训练 loss。
3. `validation_step` 前向、算 loss、更新 AP、更新 dense threshold 诊断、更新 C/P 诊断。
4. `on_validation_epoch_end` 汇总本轮验证指标、写 W\&B、本地落盘、更新 p\_best/p\_sampling cache、推动 scheduler。
5. checkpoint 保存和恢复时处理 p\_best/p\_sampling 等 wrapper 级 metadata。

实际意义：初学者读主文件时能先理解 Lightning 生命周期，而不是被细节函数淹没。

### 2. 新日志命名

| 目标       | 新前缀                                                                                                                                                                               | 实际意义                              |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| 训练 loss  | `train_loss/*`                                                                                                                                                                    | 看训练是否收敛、各监督分支是否失衡。                |
| 验证 loss  | `val_loss/*`                                                                                                                                                                      | 看验证损失与训练损失是否背离。                   |
| 模型评分     | `val_score/*`                                                                                                                                                                     | 看模型质量，例如 PR-AUC、best-F1。          |
| P-C 详细统计 | :comment[val\_pc/\*]{#comment-1780032267897 text="建议把这部分再分成三类： val_uncapedCP 、 val_capedCP、 val_refinedCP。他们分别掌控： 粗截断C（大致对应下文的第三步）、按照 cap 上限实际的候选C（大致对应下文的第四步）、refined之后的候选C的状况"} | 看 C/P 候选是否撞上 cap、阈值是否异常、正例覆盖是否足够。 |
| PR 曲线    | `val_curve/*`                                                                                                                                                                     | 翻阅每次验证的 PR 曲线和阈值表。                |

`aux` 对外统一改为 `receptor`：

* `val_score/receptor_pr_auc`
* `train_loss/receptor_loss`
* `val_loss/receptor_loss`

内部仍可使用 `voxel_logits_aux` 读取 backbone 输出。

### 3. 二分类 macro 裁剪

所有 macro 指标都必须满足“前景类数量大于 1”才记录。二分类只记录 foreground 单项。

受影响的 macro 包括：

* AP macro
* dense voxel ligand best-F1 macro
* sparse refine best-F1 macro
* candidate recall macro

实际意义：二分类下 macro 与 foreground 单项等价，继续记录会增加 W\&B 噪声。

## :comment[CPC 诊断设计]{#comment-1780031319135 text="接下来我将会修改名字"}

这里的 CPC 指的是 dense voxel 候选 C、anchor P、再回到 sparse refine C logits 的训练/验证链条。新版诊断按时间顺序记录，不保存 voxel 级明细。

### 第一步：dense voxel ligand 评分

模型先对每个有效 voxel 输出 ligand 概率。wrapper 在验证时按阈值直方图累计正负例。

记录：

* `val_score/`:comment[voxel\_ligand\_pr\_auc]{#comment-1780031528896 text="改为 voxel_ligand_PRAUC。__NEWLINE__之后涉及到的 __DQUOTE__pr_auc__DQUOTE__要改成__DQUOTE__PRAUC__DQUOTE__"}
* `val_pc/`:comment[total\_gt\_pos\_<class>]{#comment-1780031338649 text="num_gt_<class>"}
* `val_curve/`:comment[voxel\_ligand\_pr\_global\_<class>]{#comment-1780031488232 text="改为：voxel_ligand_PRcurve___BSLASH__<class>__NEWLINE____NEWLINE__但注意，明确标出横轴与纵轴(precision、recall)。__NEWLINE__以及，定义以下通用规则：__NEWLINE__1. 如果某次训练是关于二分类的，那么之后的所有指标都省略__DQUOTE_____BSLASH__<class>__DQUOTE__(设置不需要写成foreground)__NEWLINE__2. 描述曲线时，pr 改成PRcurve，描述PRAUC标量时，pr_auc等改为PRAUC"}

实际意义：回答“dense ligand head 本身有没有分辨能力”。

### 第二步：寻找 best-F1 阈值

验证结束时，在 histogram 上扫描阈值，找到 F1 最好的 `p_best`。

记录：

* `val_pc/p_best_<class>`
* `val_score/`:comment[voxel\_ligand\_best\_f1\_before\_refine\_<class>]{#comment-1780031721064 text="改成 voxel_ligand_best_f1_unrefined.__NEWLINE____NEWLINE__这是另一条通用规则： before_refine 改成 unrefined"}
* `val_pc/best_threshold_tp_<class>`
* `val_pc/best_threshold_fp_<class>`
* `val_pc/best_threshold_fn_<class>`
* :comment[val\_pc/n\_best\_total\_<class>]{#comment-1780031976720 text="改成 num_p_best_cutoff_<class>"}

实际意义：回答“如果直接按一个全局阈值切 dense 图，最好的 F1 是多少，TP/FP/FN 结构是什么”。

### 第三步：计算 sampling 阈值

根据 `n_best_total` 和 `adaptive_expand_factor` 计算扩张目标，再反推出 `p_sampling`。这是 recorded-threshold 模式后续生成 C 候选的依据。

记录：

* :comment[val\_pc/sampling\_target\_total\_<class>]{#comment-1780032075552 text="这个指标看起来冗余了？如果是就删掉"}
* `val_pc/p_sampling_<class>`
* `val_pc/sampling_threshold_tp_<class>`
* `val_pc/sampling_threshold_fp_<class>`
* `val_pc/sampling_threshold_fn_<class>`
* :comment[val\_pc/sampling\_threshold\_total\_<class>]{#comment-1780032038008 text="改成 num_p_sampling_cutoff_<class>"}

实际意义：回答“扩张后的采样阈值会放进多少正例和负例；p\_sampling 是否因为最低 bin 变成接近全选”。

### 第四步：生成候选 C

candidate builder 在每个 box / class 内按 warmup topK、adaptive threshold 或 recorded threshold 生成 C。由于有 per-box cap，理论阈值命中量和实际 C 数量可能不同。

记录：

* :comment[val\_pc/candidate\_tp\_<class>
  val\_pc/candidate\_recall\_<class>]{#comment-1780032408281 text="这两条记录修改为与第三步类似格式的四条：__NEWLINE__unrefinedC_F1_<class>__NEWLINE__unrefinedC_tp_<class>__NEWLINE__unrefinedC_fp_<class>__NEWLINE__unrefinedC_fn_<class>__NEWLINE____NEWLINE__另外:__NEWLINE__1. 再次提醒，前缀不再是 val_pc/ 而是 val_capedCP__NEWLINE__2. 所有f1改为大写 F1"}
* :comment[val\_pc/candidate\_cutoff\_prob\_<class>]{#comment-1780032445936 text="改名为 C_cutoff_prob_<class>"}
* `val_pc/num_C`

`candidate_recall` 与用户提出的 `C_recall` 同义，保留 `candidate_recall` 以贴合现有代码语义。

实际意义：回答“进入 C 的候选是否覆盖了足够多真实正例；是否因为 cap 导致大量正例被截掉”。

### 第五步：从 C 采样 P anchor

anchor sampler 从 C 中采样 P，用于后续稀疏消息或 refine。

记录：

* :comment[val\_pc/num\_P]{#comment-1780032644458 text="这是 caped 之后，当然放到 val_capedCP这里"}

实际意义：回答“P 是否撞上上限，C 到 P 的压缩是否过强”。

### 第六步：C 内 sparse refine 评分

sparse refine head 对 C 中候选做 refined logits，wrapper 统计 C 内 best-F1 和 recall。

记录：

* :comment[val\_score/ligand\_sparse\_refine\_best\_f1\_<class>]{#comment-1780032759256 text="sparse_refine 这字样也该为紧凑名字 refined"}
* :comment[val\_pc/candidate\_recall\_<class>]{#comment-1780032877616 text="这个指标是否意为：__NEWLINE__在被选中的 C 内部算出的 recall 而不是在全体素算出的 recall ?__NEWLINE__如果你确实是这个意思，那么我建议你在 val_refinedCP 这一栏增加下面的五个变量（同样对称）：__NEWLINE__p_refined_<class>  (对应达到下面四个指标的局部threshold)__NEWLINE__refinedC_F1_<class>__NEWLINE__refinedC_tp_<class>__NEWLINE__refinedC_fp_<class>__NEWLINE__refinedC_fn_<class>"}

实际意义：回答“C 覆盖足够时，refine head 是否进一步提升候选质量”。

## W\&B 展示要求

### 标量折线图

:comment[以下指标用普通 self.log 记录，W\&B 自然形成折线图：]{#comment-1780033725865 text="上面所讲的所有通用规则仍然有效（比如标量 pr_auc 写成 PRAUC，曲线 pr 写成 PRcurve, _<class>在二分类时省略，等等）"}

* `val_score/voxel_ligand_pr_auc`
* `val_score/voxel_ligand_best_f1_before_refine_<class>`
* `val_score/ligand_sparse_refine_best_f1_<class>`
* `val_pc/p_best_<class>`
* `val_pc/p_sampling_<class>`
* `val_pc/candidate_recall_<class>`
* `val_pc/num_C`
* `val_pc/num_P`

### PR 曲线翻阅栏

PR 曲线使用稳定 key，不按 epoch 生成新 key。

全局不分组曲线：

* `val_curve/global/voxel_ligand_pr_<class>`
* `val_curve/global/sparse_refine_pr_<class>`

按 `class_name` 分组曲线：

* `val_curve/by_folder/metal_ion/voxel_ligand_pr_<class>`
* `val_curve/by_folder/peptide/voxel_ligand_pr_<class>`
* `val_curve/by_folder/nucleic/voxel_ligand_pr_<class>`
* `val_curve/by_folder/small_molecule/voxel_ligand_pr_<class>`
* `val_curve/by_folder/random_BOX/voxel_ligand_pr_<class>`
* 未来新增 `random_BOX2` 时生成 `val_curve/by_folder/random_BOX2/...`

> \[!IMPORTANT]
> 每个 `class_name` 单独一栏，不把多个 folder 混在同一张 W\&B table/plot 里。全局曲线也保留自己的独立栏。

### W\&B 表格策略

每次 validation 都可上传 PR table/plot，但必须使用稳定 key，让 W\&B 按 step/epoch 历史翻阅，而不是每轮创建一批新的散乱 key。

实际意义：用户可以用 W\&B 的历史浏览查看某一轮验证的 PR 曲线，同时标量图仍能看长期趋势。

## 本地运行产物

每次 validation 本地保存完整诊断表，供 AI agent 或用户离线 debug。

建议目录：

```text
<run_dir>/validation_diagnostics/
```

建议文件：

| 文件                                                   | 意义                                                        |
| ---------------------------------------------------- | --------------------------------------------------------- |
| `epoch_XXXX_step_YYYY_global_voxel_ligand_pr.csv`    | 全局 dense voxel ligand PR 曲线、TP/FP/FN、precision/recall/F1。 |
| `epoch_XXXX_step_YYYY_by_folder_voxel_ligand_pr.csv` | 按 `class_name` 分组的 dense PR 曲线。                           |
| `epoch_XXXX_step_YYYY_sparse_refine_pr.csv`          | C 内 sparse refine PR 曲线。                                  |
| `epoch_XXXX_step_YYYY_summary.json`                  | 本轮验证的核心标量、skip warning、p\_best/p\_sampling cache。         |

这些文件不保存 box 级和 voxel 级明细。

实际意义：W\&B 用来看图；本地 CSV/JSON 用于复查、对比和 AI debug。

## 文档交付

### 面向 AI 的文档

新增：

```text
CLAUDE/docs/wrapper_metrics_reference_ai.md
```

内容：

* 每个指标的定义、分母、分子、更新阶段。
* W\&B key 与本地 CSV/JSON 字段对应关系。
* 训练运行产物位置，例如：
  * `<run_dir>/wandb/`
  * `<run_dir>/checkpoints/`
  * `<run_dir>/validation_diagnostics/`
  * checkpoint 中 p\_best/p\_sampling cache 的字段名。
* AI agent 登录服务器后如何快速检查一次 run：
  1. 找 run dir。
  2. 看 config。
  3. 看 checkpoint cache。
  4. 看 validation\_diagnostics summary。
  5. 对照 W\&B 标量和 PR 曲线。
* 常见异常解释：
  * `p_sampling=0`
  * `num_C` 撞 cap
  * `num_P` 撞 cap
  * `candidate_recall` 低但 best-F1 高
  * 某 folder 无正例导致指标 skip

### 面向用户的文档

新增：

```text
docs/model/stage1_wrapper_cpc_guide.md
```

内容：

* 按时间顺序解释训练/验证中的 CPC：
  1. dense voxel ligand 评分。
  2. dense 阈值扫描。
  3. sampling 阈值计算。
  4. C 候选生成。
  5. P anchor 采样。
  6. sparse refine 回到 C 上打分。
* 每一步的实际意义，而不是代码细节。
* W\&B 上各指标如何阅读：
  * loss 类
  * score 类
  * P-C 统计类
  * PR curve 类
* 如何判断问题来自 dense head、阈值离散化、C cap、P cap，还是 refine head。

## 模块拆分计划

### `voxel_point_stage1.py`

保留入口，只组织生命周期和 helper 调用。主文件应该能让读者快速看懂“训练一步”和“验证一步”发生了什么。

### `voxel_point_stage1_losses.py`

承接 atom、receptor、voxel ligand、sparse refine 四路 loss。对外仍让 wrapper 拿到总 loss 和分项 loss。

### `voxel_point_stage1_metrics.py`

承接 AP/PR-AUC 指标。metrics manager 必须是 `nn.Module` 或使用模块树可见的注册方式。

### `voxel_point_stage1_diagnostics.py`

承接 dense threshold、sampling threshold、C/P、folder 分组、PR 曲线表、本地 CSV/JSON。

要求：

* 全局 buffer 固定形状。
* 分组 buffer 固定形状。
* DDP 先汇总，再计算。
* 无正例跳过，不中断训练。

### `voxel_point_stage1_logging.py`

集中管理日志 key 映射，确保 `aux` 对外统一显示为 `receptor`。

### `voxel_point_stage1_scheduler.py`

可选拆分。若实现时风险过高，可把 scheduler 暂留主 wrapper，后续单独重构。

## 配置变更

### `configs/model/default.yaml`

新增 diagnostics 配置：

```yaml
monitor_metric: val_score/atom_pr_auc

validation_diagnostics:
  enabled: true
  log_wandb_curves: true
  write_local_tables: true
  group_by_meta_field: class_name
  group_names: ${dataset.class_folder_names}
  output_subdir: validation_diagnostics
```

说明：

* `group_names` 必须显式来自 dataset 配置，保证 DDP 下维度固定。
* `log_wandb_curves` 控制 PR 曲线上传。
* `write_local_tables` 控制 CSV/JSON 落盘。

### 当前实验配置

需要把仍在使用的 `monitor_metric` 改成新 key：

* `val/voxel_ligand_pr_auc` -> `val_score/voxel_ligand_pr_auc`
* `val/loss` -> `val_loss/loss`
* 二分类 sparse refine：`val/ligand_sparse_refine_best_f1_macro` -> `val_score/ligand_sparse_refine_best_f1_foreground`
* 多分类 sparse refine 才使用 `val_score/ligand_sparse_refine_best_f1_macro`

旧实验配置目录可以不批量修改，除非后续要直接重跑。

## 验证计划

### 自动测试

1. threshold 诊断测试
   * p\_best/p\_sampling 更新仍正确。
   * sampling\_threshold TP/FP/FN 正确。
   * 无正例时不中断，指标 skip。
2. sparse refine C/P 测试
   * `candidate_recall` 仍等于 C 覆盖正例数 / dense GT 正例数。
   * `num_C`、`num_P` 是每 box 平均值。
   * `candidate_cutoff_prob` 可从输出统计得到。
3. folder 分组测试
   * batch 中构造不同 `class_name`。
   * 全局统计保留。
   * 每个 folder 单独统计。
   * 未出现在当前 batch 的 folder 维度仍存在但本轮无有效指标。
4. 二分类 macro 测试
   * 二分类不出现任何 `macro` key。
   * 多分类仍出现 macro。
5. W\&B key 生成测试
   * 全局曲线 key 与 folder 曲线 key 分开。
   * 每个 folder 独立 key。
   * 不生成旧 `val/*` key。
6. 推理兼容测试
   * `output_heads: ["receptor"]` 仍读取 `voxel_logits_aux`。
   * 不改 backbone 权重 key。

### 手动验证

1. 用小配置跑一次 validation。
2. 检查 W\&B：
   * 标量按折线图出现。
   * PR 曲线在 `val_curve/global/...` 和各 `val_curve/by_folder/...` 中分别出现。
   * 每个 folder 独立栏。
   * 不出现旧 `val/*` 和二分类 macro。
3. 检查本地：
   * `<run_dir>/validation_diagnostics/` 存在。
   * summary JSON 能解释本轮 p\_best、p\_sampling、num\_C、num\_P、skip warning。
   * CSV 能重建 PR 曲线。
4. 用旧 checkpoint 跑一次推理，确认 strict load 不受日志重构影响。

## 不做的事

* 不做 box 级明细表。
* 不做 voxel 级明细表。
* 不改 backbone 内部 `voxel_aux_head` 和 `voxel_logits_aux`。
* 不保留旧 `val/*` 日志别名。
* 不把用户文档写成代码讲解；用户文档解释实际意义和 W\&B 指标阅读方式。

## 实施顺序

1. 先实现日志 key 映射，确保 `receptor` 对外命名和 `val_loss/val_score/val_pc` 三类成立。
2. 再拆 metrics，并保证 torchmetrics 注册在模块树中。
3. 再拆 diagnostics，先完成全局统计，再完成 folder 分组。
4. 加入 sampling\_threshold TP/FP/FN 和 C/P 新命名。
5. 加入 W\&B PR curve 稳定 key 与本地 CSV/JSON。
6. 更新 monitor 配置。
7. 更新 AI 文档和用户文档。
8. 跑自动测试和一次小验证。