# Find_1 训练与历史续训计划

## 目标

本计划统一管理三项互相关联的工作，但只维护一套当前生产实现：

1. 为 Find_1 建立两份显式的 PDB 中心采样配置。PDB-centric-1 与 PDB-centric-2 均进入正式训练；后者也是三种 Stage1 采样方式中的方式三，代码名保持 `pdb_centric_2`。
2. 在当前生产实现中，以一个 AdamW 优化器分别裁剪体素组与其余可训练参数组。每组梯度范数上限均为 0.5，不增加第二个优化器，不改变学习率或调度器。
3. 从历史提交 `1315d301c867c99e2dc0736feffde86e9cd7fa0a` 建立隔离的续训实现，只增加完整 checkpoint 恢复和首轮已完成 batch 跳过能力。历史续训继续使用原来的全局梯度裁剪，不吸收当前 Dataset 或分组裁剪。

资源执行遵循同一份计划与执行记录，不为 A800、H100 或两份采样配置建立多个并行生产工作树。只有历史续训因代码基点不同而允许使用一个短期隔离分支或 release。

## 已冻结的科学配置

### PDB-centric-1

- Dataset 训练请求：每个 PDB 目标 bias BOX 数量 25，bias 目标比例 0.5，单个 occurrence 的 bias BOX 上限 25。
- validation：`validation_selection_pdb_centric.npz`，150 个 PDB，3,750 个 bias BOX 与 3,750 个 context BOX。
- 训练：物理 batch size 6，全局 batch size 48，70 个 epoch，每个 epoch 12 次 validation，学习率 `5e-5`。
- 每个训练 rank 使用 24 个 Dataset workers；H100 作业的 CPU 申请与实际 GPU 数另由提交命令记录。
- 梯度裁剪：体素组 0.5，其他可训练参数组 0.5。

### PDB-centric-2

- Dataset 训练请求：每个 PDB 目标 bias BOX 数量 50，bias 目标比例 25/33，单个 occurrence 的 bias BOX 上限 1。
- validation：`validation_selection_pdb_centric_v2.npz`，200 个 PDB，3,305 个 bias BOX 与 3,200 个 context BOX。
- 训练：物理 batch size 6，全局 batch size 48，110 个 epoch，每个 epoch 8 次 validation，学习率 `5e-5`。
- 每个训练 rank 使用 30 个 Dataset workers；当前双卡 H100 作业合计使用 60 个 workers，并申请 64 CPU。
- 梯度裁剪与 PDB-centric-1 相同。
- 已获正式训练授权。单卡 Job `367928` 在 hnode01 启动后由用户按既定“先取得双卡、后取消单卡”顺序取消；当前权威训练为 Job `368455`，使用同一 hnode01 的 H100×2、CPU×64，从头训练。
- 用户把 `stop_after_lr_reductions` 从 2 明确改为 3。第二次实际学习率衰减后的完整 checkpoint 仍保留，可在需要时作为“2 次衰减”版本的等价选择端点；当前运行不改回 2。

### AUTO 体素组边界

AUTO 仓库 `C:\Users\15919\Desktop\AUTO\Pocket_Plus-v3-macro` 当前保留最优是 B_trunk，`macro_joint_score=1.6670933207345788`。在完整 Find_1 与 B_trunk 的真实 Hydra 配置中：

- 完整 Find_1 有 1,594 个可训练参数张量；
- B_trunk 有 353 个可训练参数张量；
- B_trunk 的 353 个张量与两模型可训练参数名称的交集完全相同；
- 下列三个前缀在完整 Find_1 中选中的 353 个张量与该交集完全相同：
  - `backbone.embed_head.voxel_input_proj.`：6 个；
  - `backbone.embed_head.voxel_out_proj_with_offset.`：4 个；
  - `backbone.voxel_backbone.`：343 个。

生产实现只使用这三个已核实前缀划分体素组；所有其余 `requires_grad=true` 参数属于另一组。两组必须互斥、非空并穷尽当前优化器使用的全部可训练参数。

## AUTO 优化动力学对照

参数名称相等不是最终验收。正式训练前还需使用 AUTO 的实际配置、AdamW、BF16、batch size 6、8 次梯度累积、学习率与调度器执行动力学对照：

1. 从相同的体素参数状态开始，使用相同的物理 batch 序列与 recycle 次数。
2. 比较体素输出、逐项体素损失、裁剪前逐参数梯度、体素组梯度范数、裁剪系数和裁剪后梯度。
3. 连续执行多个 optimizer step，比较参数增量、AdamW `exp_avg`、`exp_avg_sq`、step 计数和学习率。
4. 受控对照固定每个 microbatch 的 recycle 次数与随机状态；体素更新应只存在可解释的浮点或 CUDA 非确定性差异。
5. 自然随机对照只统一初始 seed，不重置完整模型点分支额外消耗的随机数。若后续 recycle 序列分离，必须明确记录首次分离位置，并证明体素差异由该随机序列变化引起，而不是点分支梯度泄漏。
6. 若 AUTO 在正式对照前保留了优于 B_trunk 的新候选，先记录新保留端点，并额外核对其结构变化；没有解释清楚的体素梯度或参数更新差异会阻止 PDB-centric-1 启动。

## 历史续训

- 历史代码基点：`1315d301c867c99e2dc0736feffde86e9cd7fa0a`。
- checkpoint：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/checkpoints/last.ckpt`。
- checkpoint SHA-256：`d633de0555f5ad46e76a36bfd83d5c4b7ab5a8cb09652918c2d6dfff225afd4c`。
- 恢复语义：恢复模型、优化器、调度器、epoch、global step 和 Lightning 循环状态；第一个恢复 epoch 跳过该 checkpoint 已完成的 45,150 个 rank-local microbatches。
- 接受 checkpoint 未保存的两个累积梯度丢失，不通过补算或改 global step 修正。
- 保持历史 Dataset、损失、模型和全局 0.5 梯度裁剪；卡型、CPU、workers 和跨节点启动方式可以变化。

## 资源接管顺序

### 两节点 A800

“合适时机”只按 `nvlink` 分区没有任何等待作业判断。窗口出现时：

1. 重新核对 `350302`、`356946`、`gnode09`、`gnode10` 和锁文件。
2. 预测取消两个明确目标作业后各节点可用 CPU。令 `F` 为两节点预测空闲 CPU 的较小值，每节点申请 `C=min(31,F-1)`，每个 rank 使用 `C-1` 个 Dataset workers。
3. 先提交并核验新作业：两节点、每节点 1×A800、每节点 C CPU、`pre_lock`、`after_lock`、跨节点 DDP。
4. 只有新作业提交成功且资源、节点、任务根、锁语义均正确，才取消仍存在的 `350302` 与 `356946`；不触碰其他作业。
5. 等待新作业运行并出现实体 `pre_lock`。历史续训代码和 smoke 通过后才删除该锁。

### H100

- `366071` 专用于 PDB-centric-1。若它在优化动力学门禁完成前获得单卡资源并开始旧命令，只创建该 Job 控制根中的 `kill_lock_366071`；核实旧训练进程退出、`kill_lock` 被消费且根级 `try_lock_366071` 出现后保留 allocation。生产实现、部署哈希和动力学门禁全部通过后，才删除该 `try_lock` 启动正式 PDB-centric-1。
- 只有同一 H100 节点存在 Slurm 实际可分配的第二张 H100、且 H100 没有等待作业时，才先提交双卡 H100 作业并核验 `pre_lock`，之后取消单卡作业。
- 节点级 `GRES_USED` 是资源判断依据；不得根据逐作业 GPU 请求数推测“名义空卡”。
- PDB-centric-2 已获训练授权。Jobs `367906`、`367917` 因 hnode02 的故障 GPU 已退出；Job `367928` 后来在 hnode01 获得单 H100 并启动，用户取得同节点双 H100 Job `368455` 后才取消该单卡 Job。当前固定守护 `368455`，不得再对已取消的 `367928` 执行资源动作。

## 实施与验收顺序

1. 固化并部署跨节点 DDP 控制层。
2. 增加两套 Dataset、训练与 experiment 配置，以及不含条件分支的明确 Shell 入口。
3. 最小实现两组梯度裁剪，并完成配置、参数边界与数值测试。
4. 完成 AUTO 受控与自然随机动力学对照。
5. 建立历史续训隔离分支，完成 checkpoint 恢复与 batch 跳过测试。
6. 对生产代码进行一轮表达/结构全面审查和一轮科学逻辑全面审查；之后只窄口径复核已报告问题。
7. 同步已验收代码，记录文件哈希、release、launch、run_cmd、Slurm Job、W&B 与产物目录。
8. 启动并综合监视历史续训、PDB-centric-1、PDB-centric-2、采样方式三 `unet_c1` 与 occurrence-centric `unet_diff`；出现可恢复问题时在各自科学边界内修复并记录。
9. 实现历史与学习历史等价后推进 `Learn/CUMULATIVE`，确保其重新成为按提交者时间形成的唯一最新提交。

## 当前状态

- [x] 跨节点 DDP 控制层完成本地验收并提交到生产实现分支。
- [x] 控制层部署到隔离服务器任务根，远端哈希与 Shell 语法核验通过。
- [x] A800 与 H100 只读资源监视已启动。
- [x] 两套 PDB 中心配置与分组裁剪实现。
- [x] AUTO 优化动力学对照；schema 4 受控、自然随机和双向 replay 门禁已由 H100 attempt a10 通过。
- [x] 历史续训恢复实现。
- [x] 正式代码审查、服务器 smoke 与训练启动；H100 PDB-centric-1 Job `366071` 已完成第七次正式 validation，配体体素 PRAUC 新高为 `0.5385515`，plateau best 已按 `expected_threshold=0.003` 更新且学习率仍为 `5e-5`。A800 a1/a2 已冻结为单 batch 伪 validation 故障证据；历史续训 attempt a3 已完成三次真实续训 validation，每次均为每 rank 551 batches、`num_gt=14,802,569`。第二次的配体体素 PRAUC `0.6297938` 仍是最佳；第三次为 `0.6125641`，plateau bad epochs 为 1，学习率仍为 `5e-5`。恢复后的 plateau、candidate cache、checkpoint 与停止计数继续按完整 validation 推进。
- [x] PDB-centric-2 正式训练已获授权；科学配置与 V2 验证选择核验通过。Job `367928` 已在 hnode01 启动单卡版本，随后按用户操作在同节点双卡 Job `368455` 获得资源后取消。`368455` 使用 H100×2、CPU×64、每 rank batch 6、每 rank 30 workers、全局 batch 48 从头训练；第三次完整 validation 已完成，每 rank 543 batches、累计 1,629 batches、`num_gt=13,835,754`、配体体素 PRAUC `0.5468278`。当前仍处于 warmup，实际学习率衰减 0 次；训练保持 `stop_after_lr_reductions=3`，并把第二次实际衰减后的完整 checkpoint 作为可选的“2 次衰减”端点。
- [x] 采样方式三 `unet_c1` Job `358384` 已于 2026-09-03 正常完成训练，随后于 2026-09-04 06:03 完成独立 calibration 与 held-out 推理并写回 `try_lock`。08:35:30 Slurm 将该 Job 记录为外部取消，本守护没有执行资源写操作；09:41 调度包装器已完成锁和 allocation 活动文件清理。未经新授权不得自行重提任务或占用空出的 H100。Occurrence-centric `unet_diff` Job `350305` 已于 2026-09-05 05:01 正常结束训练：第 3 次实际学习率衰减把 checkpoint 中的优化器学习率降至 `8e-7`，并按 `stop_after_lr_reductions=3` 停止。最终 validation 的配体体素 PRAUC 为 `0.6330414`；全程原始最高 PRAUC 仍为 `0.6343954`。最终 `last.ckpt` 与 `TOP_epoch_01_score_0.6330.ckpt` 的 SHA-256 均为 `48ad3b37...7dbfd`，原始最佳 `BEST.ckpt` 的 SHA-256 为 `8777e7a5...51409`。训练产物验收完成后，用户已释放 Job `350305` 的 held allocation，相关活动锁均已清理。
- [x] 固定醒来检查已经更新：每次守护 Jobs `368455`、`366071`、`366277` 的训练、验证、checkpoint、错误与资源状态。常规 validation、epoch 结束和普通 checkpoint 只用于判断训练健康度，不触发 handoff；提交或替换任务、训练完成、故障修复与重启、资源交接、科学契约变化等阶段事件才更新 handoff。Jobs `350305`、`358384` 均已结束并完成资源清理，不再属于固定检查清单。
- [ ] 训练完成、执行记录收口、handoff 与双线 Git 收口。
