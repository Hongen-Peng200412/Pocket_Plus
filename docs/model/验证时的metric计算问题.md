# 验证阶段 PR-AUC 指标的设备策略与根修方案

## 结论摘要

这次卡住不是因为 `voxel_ligand_pr_auc_thresholds=4096` 天然算不动，而是因为后续重构把原本应在 GPU 上做的 binned PR-AUC 阈值扫描重新放回了 CPU。

更准确地说，当前实现把两个不同问题混在了一起：

1. **显存节省问题**：非 binned AP 会保存所有 `preds/targets`，体素任务上很容易占用大量显存，所以把这类指标状态放在 CPU 是合理的。
2. **计算速度问题**：binned AP 不需要保存所有预测，而是按固定 thresholds 累积统计量。它的状态很小，但 update 时要做大量阈值比较，尤其 `4096` thresholds 乘以大体素网格，非常适合 GPU，不适合 CPU。

因此根本修正不应该是直接回退旧提交，而应该引入清晰的 metric device policy：

- `thresholds is None` 的非 binned AP：默认 CPU，节省显存。
- `thresholds is not None` 的 binned AP：默认 GPU，利用 GPU 做阈值扫描，同时保持较小状态量。
- batch size tuner 阶段也完整计算验证指标，让真实验证链路的问题尽早暴露；因此设备策略本身必须足够稳，不能依赖跳过 AP 来规避问题。

## 现场现象

任务 `243840`，实验 `tri000_justemb`，节点 `hnode02` 的诊断结果显示：

- 主进程 `284986` 占用 GPU 约 `57GB` 显存。
- `nvidia-smi pmon/dmon` 中 GPU 0 的 SM 利用率为 `0%`。
- `py-spy dump --pid 284986` 显示主线程卡在：

```text
_binary_precision_recall_curve_update_loop
torchmetrics/functional/classification/precision_recall_curve.py
_update_metric_on_tensor_device
_update_binary_or_multiclass_ap
_update_val_voxel_ligand_metric
validation_step
batch_size_scaling.py / Tuner.scale_batch_size
```

这说明程序不是卡在 dataloader、模型 forward 或 backward，而是在 Lightning 的 batch size tuner 过程中进入了 validation，然后在 CPU 上更新 TorchMetrics 的 PR-AUC。

同时日志停止在：

```text
[Train] 开始自动探测显存，寻找最佳 Batch Size (目标 Global Batch = 40)...
Loading `train_dataloader` to estimate number of stepping batches.
```

这是因为 `configs/train/B40_L3.yaml` 默认启用了：

```yaml
enable_batch_size_tuning: true
tuning_init_batch_size: 12
tuning_steps_per_trial: 10
val_per_epoch: 3
```

所以 tuner 初始就以 per-device batch size 12 试跑。对于大体素网格，这会让一次 validation metric update 处理非常多 voxel，再乘以 `4096` thresholds，CPU 上极慢。

## 当前代码证据

当前 `src/wrappers/voxel_point_stage1.py` 中，指标构造位置：

```python
self.val_voxel_ligand_pr_auc = BinaryAveragePrecision(
    compute_on_cpu=True,
    thresholds=voxel_ligand_pr_auc_thresholds,
)
```

多分类逐类 AP 的构造也统一使用：

```python
metric_kwargs = {"compute_on_cpu": True}
if thresholds is not None:
    metric_kwargs["thresholds"] = thresholds
```

指标 update 的统一 helper 当前是：

```python
metric_obj.to(preds.device)
metric_obj.update(preds, targets)
```

这个 helper 本身是为了解决 TorchMetrics 内部 `thresholds` 与输入张量设备不一致的问题。之前报错为：

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

这个修正方向是对的：metric 内部状态必须和 `preds/targets` 在同一设备上。

问题在于 `_update_binary_or_multiclass_ap()` 里已经提前把输入搬到了 CPU：

```python
preds = torch.sigmoid(...).detach().float().reshape(-1)[mask.reshape(-1)].cpu()
targets = target.reshape(-1).long()[mask.reshape(-1)].cpu()
```

多分类逐类 AP 也是：

```python
preds = prob[:, class_id].reshape(-1)[mask_flat].cpu()
targets = (target_flat[mask_flat] == class_id).long().cpu()
```

因此 `_update_metric_on_tensor_device()` 会把 metric 也搬到 CPU，最终形成：

```text
4096 thresholds × 大体素网格 × CPU 阈值扫描
```

这就是当前长时间无 GPU 计算但占用大量显存的直接原因。

## Git 历史证据

### 1. 早期 CPU 化是为了解决显存问题

在 `d4cc160` 附近，引入 voxel ligand/voxel aux 指标时，注释明确写着：

```python
# BinaryAveragePrecision, 验证阶段的 PR-AUC 指标（在 CPU 上计算以节省显存）
```

当时的逻辑把 `atom`、`voxel_aux`、`voxel_ligand` 的预测和标签都 `.cpu()` 后再 update。这个改动不是随意的，它是在试图解决验证指标占显存的问题。

这一步对 **非 binned AP** 是合理的，因为 `thresholds=None` 时，TorchMetrics 需要保存大量 `preds/targets`，体素数据上留在 GPU 会很危险。

### 2. `3b5adfe` 曾经专门让 binned ligand PR-AUC 回到 GPU

提交 `3b5adfe` 加入了：

```yaml
voxel_ligand_pr_auc_thresholds: 4096
```

并且在 ligand metric update 里有专门逻辑：

```python
metric_device = self.device if self._voxel_ligand_pr_auc_is_binned else torch.device("cpu")
preds = torch.sigmoid(logits_flat).detach().float()[mask_flat].to(metric_device)
targets = target_flat[mask_flat].to(metric_device)
self.val_voxel_ligand_pr_auc.update(preds, targets)
```

这说明历史上确实已经意识到：

- 非 binned AP 放 CPU 是为了省显存。
- binned ligand AP 有 thresholds，应该放 GPU 扫阈值。

所以用户记得“之前 4096 都可以算得动”是对的。它之所以能动，是因为当时 binned ligand AP 走 GPU。

### 3. `8c4f55f` 多分类 AP 重构时丢失了这个特殊策略

提交 `8c4f55f` 加入了 `_update_binary_or_multiclass_ap()`，把 atom、voxel_aux、voxel_ligand 的二分类/多分类 AP 更新统一起来。

统一之后，旧的 ligand 专用逻辑：

```python
metric_device = self.device if self._voxel_ligand_pr_auc_is_binned else torch.device("cpu")
```

不再存在，取而代之的是统一 `.cpu()`：

```python
preds = ...cpu()
targets = ...cpu()
self._update_metric_on_tensor_device(...)
```

这次重构本身不是坏事，因为它解决了逐类 AP、macro AP、主监控指标等新需求。但它无意中把 `4096` binned ligand AP 从 GPU 又带回 CPU。

### 4. 不能直接回退

直接回退到旧提交大概率不可取，原因有四个：

1. 旧逻辑只照顾了 binary ligand AP，没有覆盖现在的多分类逐类 AP 和 macro AP。
2. 当前 `tri000_justemb` 监控的是 `val/voxel_ligand_macro_ap`，回退会丢失这个指标链路。
3. 最近的 `_update_metric_on_tensor_device()` 是为了解决 TorchMetrics thresholds 与输入设备不一致的真实报错，简单回退可能重新触发 CPU/CUDA mismatch。
4. `compute_on_cpu=True` 和 `.cpu()` 当初是为了解决显存压力，粗暴改成全 GPU 会把非 binned AP 的所有预测状态留在显存中，容易制造新的 OOM。

所以根修应当保留这些历史改动背后的意图，而不是机械撤销某一次提交。

## TorchMetrics 行为对本项目的影响

`BinaryAveragePrecision` 大致有两种使用方式。

### 非 binned AP：`thresholds=None`

特点：

- 更接近精确 AP。
- 需要累计预测分数和标签，等 compute 时排序或构造曲线。
- 对大体素网格非常吃状态存储。
- 放在 GPU 上容易占显存；放在 CPU 上更稳。

适合：

- atom 级指标。
- 小规模 aux 指标。
- 不频繁的评估。

不适合：

- 大体素 dense voxel 的高频训练验证，尤其每个 epoch 多次 validation。

### Binned AP：`thresholds=int`

特点：

- 不需要保存所有预测。
- 状态量主要随 thresholds 数量增长，远小于保存所有 voxel 预测。
- update 时需要对每个 threshold 做比较和累计，计算量大。
- thresholds 越多，越应该用 GPU 做 update。

适合：

- dense voxel ligand AP。
- 训练中需要稳定趋势但又不能保存全部预测的场景。

注意：

- `4096` thresholds 在 GPU 上可以合理工作，但在 CPU 上会非常慢。
- 多分类 macro AP 会为每个前景类别各算一个 AP，计算量近似乘以 `(num_classes - 1)`。
- 如果 `tri000_justemb` 的前景类别较多，`4096 × 类别数 × 有效体素数` 会明显重，即使在 GPU 上也应关注验证频率。

## 根本修正目标

根修要同时满足：

1. **快**：binned AP 的阈值扫描走 GPU。
2. **省显存**：非 binned AP 的大量预测状态仍然走 CPU。
3. **兼容多分类**：binary AP、逐类 AP、macro AP 使用同一套设备策略。
4. **兼容 Lightning**：metric 内部 `thresholds`、`preds`、`targets` 永远在同一设备。
5. **覆盖 tuner**：batch size tuner 期间也应完整计算验证 AP，以便尽早暴露 CPU/GPU 设备不一致或性能问题。
6. **保留当前监控链路**：`val/voxel_ligand_macro_ap`、warmup plateau、checkpoint monitor 继续可用。

## 推荐设计

### 1. 为每个 metric 建立明确的设备策略

建议在 wrapper 中维护一个轻量 registry，例如：

```python
self._val_metric_specs[metric_name] = {
    "thresholds": thresholds,
    "binned": thresholds is not None,
    "branch": prefix,
}
```

然后提供统一设备选择函数：

```python
def _metric_update_device(self, metric_name: str, source_device: torch.device) -> torch.device:
    policy = getattr(self.hparams, "val_metric_device_policy", "auto")
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "gpu":
        return source_device

    spec = self._val_metric_specs[metric_name]
    if spec["binned"]:
        return source_device
    return torch.device("cpu")
```

默认策略 `auto` 即：

- binned metric：GPU。
- non-binned metric：CPU。

这样不需要在业务逻辑里猜测某个 metric 应该 `.cpu()` 还是 `.cuda()`。

### 2. 构造 metric 时不要所有指标都 `compute_on_cpu=True`

当前 `_init_multiclass_ap_metrics()` 统一使用：

```python
metric_kwargs = {"compute_on_cpu": True}
if thresholds is not None:
    metric_kwargs["thresholds"] = thresholds
```

建议改为：

```python
metric_kwargs = {"compute_on_cpu": thresholds is None}
if thresholds is not None:
    metric_kwargs["thresholds"] = thresholds
```

对于主 ligand binned AP 也一样：

```python
self.val_voxel_ligand_pr_auc = BinaryAveragePrecision(
    compute_on_cpu=voxel_ligand_pr_auc_thresholds is None,
    thresholds=voxel_ligand_pr_auc_thresholds,
)
```

这不是为了完全依赖 `compute_on_cpu` 控制设备，而是为了避免 binned metric 被 TorchMetrics 的 CPU state 语义误导。真正的 update 设备仍由我们的 policy helper 控制。

### 3. update 时不要提前硬编码 `.cpu()`

当前逻辑的问题是先 `.cpu()`，再让 metric 跟随 `preds.device`。

建议改成：

```python
preds = torch.sigmoid(...).detach().float().reshape(-1)[mask.reshape(-1)]
targets = target.reshape(-1).long()[mask.reshape(-1)]
self._update_metric(metric_name, metric_obj, preds, targets)
```

统一 helper：

```python
def _update_metric(self, metric_name, metric_obj, preds, targets):
    device = self._metric_update_device(metric_name, preds.device)
    preds = preds.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    metric_obj.to(device)
    metric_obj.update(preds, targets)
```

这样：

- binned voxel ligand AP 保留在 GPU。
- 非 binned atom/aux AP 仍然会被搬到 CPU。
- metric 内部 thresholds 和输入张量保持同设备，不会再出现 CPU/CUDA mismatch。

### 4. 多分类逐类 AP 使用同一策略

多分类分支中，每个 `val_{prefix}_ap_{class_name}` 都应注册自己的 metric spec。

例如：

```python
metric_name = f"val/{prefix}_ap_{class_name}"
self._val_metric_specs[metric_name] = {
    "thresholds": thresholds,
    "binned": thresholds is not None,
    "branch": prefix,
    "class_id": class_id,
}
```

然后 class AP update 也不要提前 `.cpu()`：

```python
preds = prob[:, class_id].reshape(-1)[mask_flat]
targets = (target_flat[mask_flat] == class_id).long()
self._update_metric(metric_name, metric_obj, preds, targets)
```

macro AP 本身不是 TorchMetrics 对象，而是多个逐类 AP compute 后求均值，所以它不需要单独设备策略。

### 5. batch size tuner 阶段也完整计算验证 AP

这是用户明确选择的行为：tuner 期间不跳过 AP，有问题尽早显现。

Lightning 的 `Tuner.scale_batch_size()` 会真的跑 loop。当前配置又有 `val_per_epoch=3`，因此 tuner 期间可能进入 validation，并触发 PR-AUC update。tuner 的目的只是寻找可用 batch size，不需要验证 PR-AUC。

虽然从纯粹搜索 batch size 的角度看，跳过 AP 会更快，但这会隐藏验证链路中的设备错误和性能问题。本项目此前已经遇到过 TorchMetrics `thresholds` 与输入张量设备不一致的问题，因此更稳妥的策略是：tuner 也完整跑 AP，但用正确的设备策略保证它能跑得动。

这意味着：

1. binned AP 必须走 GPU，否则 tuner 会再次卡在 CPU 阈值扫描。
2. 非 binned AP 仍然走 CPU，否则 tuner 可能更早触发显存压力。
3. 如果 tuner 期间仍然慢，优先降低 `voxel_ligand_pr_auc_thresholds` 或验证频率，而不是隐藏 validation metric。

### 6. 配置层建议

建议新增或约定以下配置：

```yaml
model:
  val_metric_device_policy: auto  # auto|cpu|gpu
```

对于 dense voxel ligand：

```yaml
model:
  voxel_ligand_pr_auc_thresholds: 4096
```

可以继续保留，但要满足两个条件：

1. binned AP update 走 GPU。
2. batch size tuner 也完整计算验证 AP，用来提前暴露真实验证链路问题。

对于多分类 ligand macro AP，如果类别数较多，训练期可以考虑：

```yaml
model:
  voxel_ligand_pr_auc_thresholds: 1024
```

最终评估或少量验证再使用：

```yaml
model:
  voxel_ligand_pr_auc_thresholds: 4096
```

如果坚持训练期也用 `4096`，建议至少降低验证频率，并重点观察 tuner 期间的 validation metric 耗时。`4096` 在 GPU 上可以跑，但多分类逐类 AP 会把计算量乘以前景类别数。

## 推荐落地顺序

### 第一步：修复设备策略

修改 `src/wrappers/voxel_point_stage1.py`：

1. 增加 `_val_metric_specs`。
2. 构造 metric 时按 `thresholds is None` 决定 `compute_on_cpu`。
3. 把 `_update_metric_on_tensor_device()` 改成 `_update_metric(metric_name, metric_obj, preds, targets)`。
4. 删除 `_update_binary_or_multiclass_ap()` 中硬编码 `.cpu()`。
5. binned metric 使用 logits/preds 的原始设备，通常是当前 GPU。
6. non-binned metric 使用 CPU。

这是最核心的修正。

### 第二步：保持 tuner 完整验证

不增加 `_skip_heavy_validation_metrics` 之类的绕过开关。`src/train.py` 中 batch size tuner 仍然按 Lightning 原流程运行 validation，wrapper 也照常 update/compute AP。

这样做牺牲一些 tuner 速度，但能尽早发现验证指标在真实 batch size、真实 valid mask 和真实类别数下是否存在设备或性能问题。

### 第三步：保守调整实验配置

对重型体素实验，建议显式写清：

```yaml
train:
  enable_batch_size_tuning: false
  batch_size: 4
  strict_global_batch_size: false
```

如果后续确认 tuner + 完整 AP 的成本过高，优先调低训练期 thresholds 或验证频率，而不是直接跳过 AP。

`tri000_justemb` 这类多分类 macro AP 实验，训练期可以先用 `1024` 或 `2048` thresholds 观察趋势；最终 eval 再用 `4096`。如果确实需要训练期 `4096`，应配合 GPU binned update 和较低验证频率。

## 验证方案

### 1. 单元级验证

构造小张量，分别测试：

- `thresholds=None` 时 metric update 设备为 CPU。
- `thresholds=4096` 时 metric update 设备为 CUDA。
- binary AP 和 multiclass class AP 都走同一 helper。
- `metric.compute()` 后返回值能 `.to(self.device)` 并正常 `self.log(sync_dist=True)`。

### 2. 服务器小跑验证

先用小 batch 跑：

```bash
python -u /home/penghongen/My_Project/Pocket_Plus/src/train.py \
  "+experiment=tri000_justemb" \
  train.enable_batch_size_tuning=false \
  train.batch_size=2 \
  train.strict_global_batch_size=false \
  model.voxel_ligand_pr_auc_thresholds=4096
```

观察：

- 不再出现 CPU/CUDA mismatch。
- `py-spy dump` 不应长时间停在 CPU 的 `_binary_precision_recall_curve_update_loop`。
- `nvidia-smi pmon` 在 validation metric update 期间应看到 GPU SM 有短时活动。
- 显存相比原训练不应大幅增长，因为 binned metric 不保存所有 preds。

### 3. tuner 验证

再测试：

```bash
python -u /home/penghongen/My_Project/Pocket_Plus/src/train.py \
  "+experiment=tri000_justemb" \
  train.enable_batch_size_tuning=true \
  train.tuning_init_batch_size=12 \
  model.voxel_ligand_pr_auc_thresholds=1024
```

预期：

- tuner 期间会完整更新 AP。
- 不会在 CPU 的 `_binary_precision_recall_curve_update_loop` 长时间卡住。
- `nvidia-smi pmon` 在 binned AP update 时应看到 GPU 短时活动。

## 最终建议

最稳妥的根修方案是：

1. **恢复但泛化 `3b5adfe` 的核心思想**：binned ligand AP 用 GPU，不是只给旧的 binary ligand 分支打补丁，而是覆盖 binary、multiclass class AP、macro AP 的整条链路。
2. **保留 CPU 化的初衷**：非 binned AP 继续用 CPU，避免保存所有 dense voxel 预测导致显存膨胀。
3. **让 tuner 暴露真实问题**：batch size tuner 期间也完整计算 validation metrics，不用跳过逻辑掩盖设备或性能问题。
4. **不要简单回退**：因为当前多分类 AP、macro AP、monitor_metric、warmup plateau 和设备 mismatch 修复都是真实需求，直接回退会丢功能或复现旧 bug。

一句话版本：

> 非 binned AP 的瓶颈是存储，所以放 CPU；binned AP 的瓶颈是阈值扫描，所以放 GPU；batch size tuner 也完整跑 AP，用来提前暴露真实验证链路问题。
