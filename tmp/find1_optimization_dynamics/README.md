# Find_1 优化动力学验收

本目录保存 Find_1 正式训练前的一次性验收代码, 不参与训练主入口. 生产配置位于 `configs/`, 分组梯度裁剪位于 `src/wrappers/voxel_point_stage1.py`. 本目录只在实现分支和服务器验收副本中留证; AUTO 与完整 Find_1 的实际轨迹通过后, 它会在最终学习端点前退出活动代码树.

## 阅读与执行顺序

1. `optimization_trace.py`: 在一个独立 Python 进程中运行 AUTO B_trunk 或完整 Find_1 的一条轨迹.
2. `compare_optimization_traces.py`: 比较相同轨迹类型的两个 JSON 文件.
3. `tests/test_compare_optimization_traces.py`: 验证受控轨迹、自然随机轨迹和硬失败边界.

AUTO 与完整模型必须使用两个独立 Python 进程, 防止两个项目中同名 `src` 包共享模块缓存. 两个进程必须先后使用同一张 GPU, 同一个 checkpoint、随机种子、物理 batch、梯度累积数和 optimizer step 数.

## 轨迹命令

下面十二条命令生成两份受控轨迹、两份自然随机轨迹和四份反事实重放轨迹，再完成三组逐张量比较与一份自然轨迹归因比较。`FULL_PROJECT_ROOT` 必须替换为本轮验收 release 的绝对路径；`OUTPUT_ROOT` 必须是本次验收专属目录。每次正式执行所用的展开后命令和文件 SHA-256 另行写入执行记录。

```bash
AUTO_PROJECT_ROOT="/home/penghongen/Feedback/AUTO/Pocket_Plus-v3-macro/releases/Pocket_Plus-v3-macro_2eec850361c7/Pocket_Plus-v3-macro"
FULL_PROJECT_ROOT="<本轮正式验收 release 的绝对路径>"
OUTPUT_ROOT="/home/penghongen/Feedback/Pocket_Plus_Find1/validation/<run-id>"
FULL_SOURCE_COMMIT="<本轮生产实现提交>"
TOOL_ROOT="${FULL_PROJECT_ROOT}/tmp/find1_optimization_dynamics"
CHECKPOINT="/storage/penghongen/tmp/AUTO/TUNE--Find-v3-macro/trials/baseline-7aae1f126ee0-20260828T034121771-b6/training/logs/Pocket_Plus_AUTO_tune_stage1_v3_macro/baseline-7aae1f126ee0-20260828T034121771-b6____baseline-7aae1f126ee0-20260828T034121771-b6/checkpoints/BEST.ckpt"
TRUNK_PROJECT_SOURCE_SHA256="<已核验 AUTO release 的 src 与 configs 摘要>"
FULL_PROJECT_SOURCE_SHA256="<已核验生产 release 的 src 与 configs 摘要>"

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${AUTO_PROJECT_ROOT}" \
  --experiment CPC1/Find_1_trunk --role trunk --track controlled \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/trunk_controlled.json" \
  --source-identity 7aae1f126ee0c1636a3f14455f8239a072ac28c9

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${FULL_PROJECT_ROOT}" \
  --experiment CPC1/Find_1 --role full --track controlled \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/full_controlled.json" \
  --source-identity "${FULL_SOURCE_COMMIT}"

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${AUTO_PROJECT_ROOT}" \
  --experiment CPC1/Find_1_trunk --role trunk --track natural \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/trunk_natural.json" \
  --source-identity 7aae1f126ee0c1636a3f14455f8239a072ac28c9

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${FULL_PROJECT_ROOT}" \
  --experiment CPC1/Find_1 --role full --track natural \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/full_natural.json" \
  --source-identity "${FULL_SOURCE_COMMIT}"

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${AUTO_PROJECT_ROOT}" \
  --experiment CPC1/Find_1_trunk --role trunk --track replay \
  --recycle-sequence-from "${OUTPUT_ROOT}/trunk_natural.json" \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/trunk_replay_trunk_sequence.json" \
  --source-identity 7aae1f126ee0c1636a3f14455f8239a072ac28c9

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${FULL_PROJECT_ROOT}" \
  --experiment CPC1/Find_1 --role full --track replay \
  --recycle-sequence-from "${OUTPUT_ROOT}/trunk_natural.json" \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/full_replay_trunk_sequence.json" \
  --source-identity "${FULL_SOURCE_COMMIT}"

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${AUTO_PROJECT_ROOT}" \
  --experiment CPC1/Find_1_trunk --role trunk --track replay \
  --recycle-sequence-from "${OUTPUT_ROOT}/full_natural.json" \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/trunk_replay_full_sequence.json" \
  --source-identity 7aae1f126ee0c1636a3f14455f8239a072ac28c9

python "${TOOL_ROOT}/optimization_trace.py" \
  --project-root "${FULL_PROJECT_ROOT}" \
  --experiment CPC1/Find_1 --role full --track replay \
  --recycle-sequence-from "${OUTPUT_ROOT}/full_natural.json" \
  --batch-size 6 --accumulate-steps 8 --optimizer-steps 2 --seed 3407 \
  --checkpoint "${CHECKPOINT}" --output "${OUTPUT_ROOT}/full_replay_full_sequence.json" \
  --source-identity "${FULL_SOURCE_COMMIT}"

python "${TOOL_ROOT}/compare_optimization_traces.py" \
  --trunk "${OUTPUT_ROOT}/trunk_controlled.json" \
  --full "${OUTPUT_ROOT}/full_controlled.json" \
  --output "${OUTPUT_ROOT}/controlled_comparison.json" \
  --expected-full-source-identity "${FULL_SOURCE_COMMIT}" \
  --expected-trunk-project-source-sha256 "${TRUNK_PROJECT_SOURCE_SHA256}" \
  --expected-full-project-source-sha256 "${FULL_PROJECT_SOURCE_SHA256}"

python "${TOOL_ROOT}/compare_optimization_traces.py" \
  --trunk "${OUTPUT_ROOT}/trunk_replay_trunk_sequence.json" \
  --full "${OUTPUT_ROOT}/full_replay_trunk_sequence.json" \
  --output "${OUTPUT_ROOT}/trunk_sequence_replay_comparison.json" \
  --expected-full-source-identity "${FULL_SOURCE_COMMIT}" \
  --expected-trunk-project-source-sha256 "${TRUNK_PROJECT_SOURCE_SHA256}" \
  --expected-full-project-source-sha256 "${FULL_PROJECT_SOURCE_SHA256}"

python "${TOOL_ROOT}/compare_optimization_traces.py" \
  --trunk "${OUTPUT_ROOT}/trunk_replay_full_sequence.json" \
  --full "${OUTPUT_ROOT}/full_replay_full_sequence.json" \
  --output "${OUTPUT_ROOT}/full_sequence_replay_comparison.json" \
  --expected-full-source-identity "${FULL_SOURCE_COMMIT}" \
  --expected-trunk-project-source-sha256 "${TRUNK_PROJECT_SOURCE_SHA256}" \
  --expected-full-project-source-sha256 "${FULL_PROJECT_SOURCE_SHA256}"

python "${TOOL_ROOT}/compare_optimization_traces.py" \
  --trunk "${OUTPUT_ROOT}/trunk_natural.json" \
  --full "${OUTPUT_ROOT}/full_natural.json" \
  --output "${OUTPUT_ROOT}/natural_comparison.json" \
  --expected-full-source-identity "${FULL_SOURCE_COMMIT}" \
  --expected-trunk-project-source-sha256 "${TRUNK_PROJECT_SOURCE_SHA256}" \
  --expected-full-project-source-sha256 "${FULL_PROJECT_SOURCE_SHA256}" \
  --counterfactual-replay-comparison "${OUTPUT_ROOT}/trunk_sequence_replay_comparison.json" \
  --counterfactual-replay-comparison "${OUTPUT_ROOT}/full_sequence_replay_comparison.json"
```

`optimization_trace.py` 默认使用 BF16、物理 batch size 6、8 次梯度累积、2 个 optimizer step、AdamW 和 0.5 梯度裁剪. 可用同名命令行参数覆盖 batch、累积数、step 数和 seed; 正式验收不得覆盖冻结值.

## 轨迹 JSON 契约

顶层字段包括:

- `schema_version: int`: 当前为 4。
- `role: str`: `trunk` 或 `full`.
- `track: str`: `controlled`、`natural` 或 `replay`.
- `source_identity/project_root/experiment: str`: 命令指定的提交身份、自动解析的项目根和 Hydra experiment.
- `process_nonce/process_pid`: 独立 Python 进程身份；nonce 在模块导入时生成一次，比较器同时拒绝相同 nonce 和相同 PID.
- `project_source_sha256/resolved_config_sha256`: 自动计算的 `src`、`configs` 内容摘要和完整解析配置摘要；比较器要求项目源码摘要等于正式执行前从两个只读 release 独立核验并记录的预期值.
- `checkpoint/checkpoint_sha256/checkpoint_global_step/checkpoint_epoch`: checkpoint 路径、摘要和训练坐标.
- `shared_state_count: int`: 从共同 checkpoint 载入且形状匹配的模型状态数量.
- `batch_size/accumulate_steps/optimizer_steps/seed/input_channels: int`: 本次动力学参数.
- `trainable_parameter_tensors/voxel_parameter_tensors/other_parameter_tensors: int`: 参数张量计数.
- `voxel_parameter_names_sha256: str`: 排序后体素参数名称的 SHA-256.
- `dense_tensor_store: object`: 本条轨迹的 float32 稠密旁车文件名、格式、字节数、捕获签名数和去重后张量数。
- `point_to_voxel_gradient_probe: object`: 首批 `atom` 与 `pseudo` 两项监督各自对体素组产生的梯度范数；两项都必须存在且严格为零.
- `recycle_sequence_source_sha256/recycle_sequence`: replay 使用的自然轨迹摘要和本次实际 recycle 序列.
- `microbatches: list[object]`: 每个 microbatch 的请求位置、完整请求身份、recycle 数、总损失、各损失项、共同体素输出和逐体素参数累计梯度摘要.
- `optimizer_step_records: list[object]`: 每次更新前后的学习率、两组梯度范数、体素裁剪系数，以及体素组和其他组的逐参数原值、裁剪前后梯度、参数增量、更新后参数、AdamW `exp_avg/exp_avg_sq` 和逐参数 state step。参数原值、参数增量和更新后参数不得为空；当前 step 没有梯度的参数以 `null` 记录梯度，尚未建立 AdamW state 的参数以 `null` 记录一阶矩、二阶矩与 state step。

张量摘要统一包含 `shape`、`dtype`、`numel`、原始 dtype 全张量字节的 `sha256`、`all_finite`、`l2`、`mean`、`max_abs` 和固定位置 `samples`。需要跨模型比较的共同体素输出、累计梯度、参数、参数增量与 AdamW 状态还包含 `dense_float32`：不超过 64 个元素的数组直接以 Base64 保存；较大数组按 float32 小端字节顺序写入该轨迹唯一的 `*.dense-float32.bin` 旁车，JSON 记录字节偏移、长度与内容 SHA-256。相同字节内容在一条轨迹内只存一份。

比较器先核对旁车文件规模、每个数组区间、字节长度和 SHA-256，再以 1,048,576 个元素为一块读取两侧数组，计算真正的逐元素最大绝对差。任一位置超过对应 `element_absolute_max` 都会使该 microbatch 或 optimizer step 失败；该判定不依赖抽样、线性投影或张量长度，因此交换未抽样位置也不能绕过。原始 dtype 的 `sha256` 仍用于证明共同初始参数严格相同，并记录其余张量是否发生逐字节差异。任一 NaN/Inf 都会失败。程序先原子发布旁车，再原子发布引用它的 JSON；计算或发布失败时不会留下可被当作完整轨迹读取的 JSON。

## BF16 数值包络的事实依据

共同体素路径使用 CUDA `scatter_add_` 累加同一体素的多个原子贡献。该原子累加在 H100 上不保证逐字节确定性。服务器 attempt a6 的八条真实轨迹位于：

`/home/penghongen/Feedback/Pocket_Plus/validation/find1_optimization_dynamics/Find_1_job366071_20260901T044525_a6`

在该次运行中，同一个模型使用相同 checkpoint、请求、recycle 序列和初始参数重复执行时，输出与梯度的 SHA-256 仍会变化。跨角色比较与同角色重复基线处于同一数量级，且没有出现贯穿输出、损失、梯度和更新的系统性放大：

- 共同输出 `l2` 相对差最大值为 `1.67e-4`，同角色重复基线最大值为 `1.83e-4`。
- 共同损失绝对差最大值为 `3.57e-4`，同角色重复基线最大值为 `8.05e-4`。
- 累计体素梯度的逐参数 `l2` 相对差 P95 最大值为 `2.45e-2`，同角色重复基线最大值为 `2.99e-2`。
- 累计体素梯度范数相对差最大值为 `3.36e-2`，同角色重复基线最大值为 `3.07e-2`；二者的绝对差最大值分别为 `5.41e-3` 和 `6.57e-3`。
- 两个 optimizer step 的参数更新、AdamW 一阶矩与二阶矩差异也与同角色重复基线处于同一数量级。

因此，比较器不再把共同路径的逐字节相等当作科学契约。它仍要求第一个 optimizer step 前的 353 个共同参数逐字节相同，并把后续共同输出、损失、梯度、裁剪系数、参数更新和 AdamW 状态限制在 `compare_optimization_traces.py` 中冻结的字段化包络内。相对差和传统绝对差上限以 attempt a6 的观测上界为基准保留有限余量；schema 4 新增的逐元素绝对差尚须由新的 H100 轨迹标定。attempt a6 不含稠密旁车，只能说明逐字节相等不是可行契约，不能被追认为最终通过。

## 比较结果 JSON 契约

`compare_optimization_traces.py` 输出:

- `schema_version: int`: 当前为 4。
- `passed: bool`: 所有应比较字段均满足容差且硬边界成立.
- `track/atol/rtol`: 轨迹类型，以及学习率等确定性标量使用的通用容差；BF16 共同路径使用下述字段化包络。
- `first_recycle_divergence: int | null`: 自然随机轨迹首次 recycle 数不同的 microbatch；受控与 replay 轨迹必须为空.
- `eligible_microbatches: int`: recycle 分离前纳入等价判定的 microbatch 数量.
- `numerical_limits: object`: 共同输出、共同损失、累计梯度、梯度范数、裁剪标量、参数更新和 AdamW 状态各自的冻结上限。
- `numerical_diagnostics: object`: 本次比较跨全部合格 microbatch 和 optimizer step 汇总的张量数量、SHA-256 不同数量、传统统计差异和逐元素最大绝对差。该字段只用于报告；是否通过由每个 microbatch 和每个 optimizer step 各自与 `numerical_limits` 比较决定，后续批次不能稀释局部异常。
- `trace_identity: object`: 两条输入轨迹的 role、提交身份、项目根、项目源码摘要、解析配置摘要、checkpoint 路径与摘要、batch、梯度累积数、step 数、随机种子、输入通道和参数边界等冻结身份。
- `mismatch_count: int` 与 `mismatches: list[str]`: 差异总数和最多 200 条定位信息.

比较器还固定核对 AUTO/生产 role、experiment、AUTO 提交、生产提交、两个受信 release 的项目源码摘要、不同项目根、不同进程 nonce 与 PID、353 个体素参数、1,594 个完整模型参数、1,241 个其他参数及体素名称摘要。共同初始参数不是逐字节相同、完整模型没有且仅有 `atom/pseudo` 两个额外损失、任一总损失或损失项不在 `[0, 10]`、任何张量或标量非有限、点监督向体素组产生非零梯度、任一组裁剪后范数超过 0.5、受控/replay recycle 分离、AdamW state 迁移不同或任一局部数值诊断超过冻结包络，都会使程序以退出码 1 结束。损失名称与 `[0, 10]` 区间先对两条完整轨迹分别检查，再决定 natural 轨迹可进行跨角色数值比较的共同前缀，因此第 0 个 microbatch 的 recycle 已经分离也不能隐藏损坏的损失。AdamW state step 只在参数本步存在梯度时加一；本步没有梯度时允许保留前一步数值。`[0, 10]` 是排除损坏轨迹的宽松安全区间；attempt a6 的总损失为 `0.0872–0.5899`，单项损失为 `0.0002608–0.3157`，该边界不承担优化等价判定。

自然轨迹若发生 recycle 分离，最终比较必须同时读取两份已通过的 replay 比较：一份让两模型共同重放 AUTO 自然序列，另一份共同重放完整模型自然序列。两份 replay 的 `recycle_sequence_source_sha256` 必须分别等于两份自然 JSON 的真实 SHA-256；两份比较还必须使用 schema 4，携带与当前自然轨迹完全相同的 `trace_identity`，并记录四条不同且非空的 replay 轨迹 SHA-256。因此即使第 0 个 microbatch 就分离，也不能靠零个可比较 microbatch、旧 schema 的比较或错误 checkpoint 的 replay 假通过。

## 记录内容

轨迹记录以下事实:

- 每个 microbatch 的请求身份、recycle 次数、共同体素输出、共同体素损失与累积体素梯度范数.
- 每个 optimizer step 裁剪前后的逐参数梯度摘要.
- 第一个真实 batch 中 `atom/pseudo` 点监督对体素参数产生的梯度范数; 该值必须为零.
- 每个 optimizer step 的体素裁剪系数、参数原值、实际参数增量、更新后参数、`exp_avg`、`exp_avg_sq`、step 计数和学习率.
- 项目身份、checkpoint SHA-256 与体素参数名称摘要.

正式输出统一写入服务器反馈根的独立验收目录, 不进入训练 checkpoint 目录.
