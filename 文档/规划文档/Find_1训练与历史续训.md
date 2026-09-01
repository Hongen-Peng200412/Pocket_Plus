# Find_1 训练与历史续训计划

## 目标

本计划统一管理三项互相关联的工作，但只维护一套当前生产实现：

1. 为 Find_1 建立两份显式的 PDB 中心采样配置。PDB-centric-1 完成训练并提交；PDB-centric-2 只完成配置与测试，未经用户新授权不提交训练。
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
- 每个训练 rank 使用 24 个 Dataset workers。
- 梯度裁剪与 PDB-centric-1 相同。
- 本轮只完成配置组合、Shell 入口和测试，不提交训练。

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
- PDB-centric-2 未经新授权不提交。

## 实施与验收顺序

1. 固化并部署跨节点 DDP 控制层。
2. 增加两套 Dataset、训练与 experiment 配置，以及不含条件分支的明确 Shell 入口。
3. 最小实现两组梯度裁剪，并完成配置、参数边界与数值测试。
4. 完成 AUTO 受控与自然随机动力学对照。
5. 建立历史续训隔离分支，完成 checkpoint 恢复与 batch 跳过测试。
6. 对生产代码进行一轮表达/结构全面审查和一轮科学逻辑全面审查；之后只窄口径复核已报告问题。
7. 同步已验收代码，记录文件哈希、release、launch、run_cmd、Slurm Job、W&B 与产物目录。
8. 启动并监视历史续训与 PDB-centric-1，出现可恢复问题时在各自科学边界内修复并记录。
9. 实现历史与学习历史等价后推进 `Learn/CUMULATIVE`，确保其重新成为按提交者时间形成的唯一最新提交。

## 当前状态

- [x] 跨节点 DDP 控制层完成本地验收并提交到生产实现分支。
- [x] 控制层部署到隔离服务器任务根，远端哈希与 Shell 语法核验通过。
- [x] A800 与 H100 只读资源监视已启动。
- [x] 两套 PDB 中心配置与分组裁剪实现。
- [x] AUTO 优化动力学对照；schema 4 受控、自然随机和双向 replay 门禁已由 H100 attempt a10 通过。
- [x] 历史续训恢复实现。
- [x] 正式代码审查、服务器 smoke 与训练启动；A800 历史续训 Job `366277` 与 H100 PDB-centric-1 Job `366071` 均已进入真实 step。
- [ ] 训练完成、执行记录收口、handoff 与双线 Git 收口。
