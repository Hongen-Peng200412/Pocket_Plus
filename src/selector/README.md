# AdaLigand Stage1 Selector

Selector 对单个 Stage1 producer 独立训练和推理。Find 使用真实 V/P/A；`unet_c1` 只使用真实 V。所有输入先经过统一 `residual_swiglu` 融合，V5 与经过 `raw_clipnorm_none` 的原始密度上下文共同调制最终 V48；DensityMUNetLite 通道为 `[32,64,64,128]`，只在 10³ bottleneck 使用一层 Transformer。

## 训练

复制并修改 `configs/selector/Find_0.yaml`、`Find_1.yaml` 或 `unet_c1.yaml` 中的路径，然后执行：

```bash
python -m src.selector.train --config /absolute/path/to/selector.yaml
```

训练启动时只扫描一次已发布且可消费的 `CLG_centered`，按固定顺序把精确样本集合写入当前 `selector_run_dir/input_CLG_list.json`。之后 Dataset 严格使用该清单，不追逐后来新增的样本；需要公平比较的方案应显式复用同一清单。`formal_run=false` 只产生 `TRIAL_BEST.ckpt`；只有固定 validation 清单完整覆盖且显式设为 `true` 时才产生 `BEST.ckpt`。

正式 batch size 不写死：先用当前硬件做显存 preflight，再固定进该次 resolved config。训练期间不动态改变 Dataset 样本集合。

## 打分、校准和选择

```bash
# 对冻结清单写 scores.npz
python -m src.selector.inference scores \
  --checkpoint /absolute/path/to/BEST.ckpt \
  --input-clg-list /absolute/path/to/input_CLG_list.json \
  --stage1-outputs-root /absolute/path/to/stage1_outputs \
  --upstream-root /absolute/path/to/Ori_Data \
  --selector-run-dir /absolute/path/to/selector_run \
  --split calibration --device cuda

# 在 calibration 的实际唯一 p_G 候选上升序扫描，冻结 first maximum 的 tau_G
python -m src.selector.inference calibrate \
  --selector-run-dir /absolute/path/to/selector_run \
  --stage1-outputs-root /absolute/path/to/stage1_outputs \
  --input-clg-list /absolute/path/to/input_CLG_list.json \
  --stage1-model-name Find_0 --lambda-count 0.05

# 对一个 PDB 的 forest/CLG 解码精确 gated antichain，并发布 selection.npz
python -m src.selector.inference selection \
  --scores /absolute/path/to/scores.npz \
  --forest /absolute/path/to/forest.npz \
  --clg /absolute/path/to/clg.npz \
  --calibration /absolute/path/to/calibration.json \
  --output /absolute/path/to/selection.npz
```

`partition` 是当前 CLG 所有非空反链的预测能量 `exp(score)` 之和，用于训练期归一化；预测 MAP 与 GT oracle 都在同一棵专属树结构上做精确动态规划。`p_G < tau_G` 时返回空集；通过 gate 后必须返回非空反链。最终 `selection.npz` 由 `src.inference.cli selected-refined` 消费，生成与来源全图节点可追溯对应的 `Selected_Refined_Centered`。
