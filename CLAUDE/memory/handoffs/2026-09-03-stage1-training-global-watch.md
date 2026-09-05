# Handoff：Stage1 三条活动训练线综合守护

Date: 2026-09-04
Updated: 2026-09-05

## Current State

当前持续目标统一守护三条活动训练线：Find_1 PDB-centric-1、Find_1 PDB-centric-2 双卡训练和 Find_1 两节点历史续训。Occurrence-centric `unet_diff` Job `350305` 已完成训练并由用户释放，不再属于活动训练或资源守护清单。代码名 `pdb_centric_2` 表示 Find_1 的第二套 PDB 中心配置，也是三种 Stage1 采样方式中的方式三；不另建含义重复的 `pdb_centric_3` 配置。

截至 2026-09-05 10:26 +08:00，Jobs `368455`、`366071` 和 `366277` 正常训练。Job `350305` 已在第 3 次实际学习率衰减后正常结束训练，最终 validation 配体体素 PRAUC 为 `0.6330414`，全程原始最高为 `0.6343954`；最终与最佳 checkpoint 均已核验。用户已在产物验收后释放该 Job，活动锁与动态命令均已清理。Find_1 PDB-centric-2 的配置保持 `stop_after_lr_reductions=3`，第二次实际衰减后的完整 checkpoint 必须保留，可在需要时作为“2 次衰减”版本的等价端点。

Job `358384` 的推理于 06:03 成功结束并写回 `try_lock`，但 Slurm 于 08:35:30 把该 Job 记录为外部取消；本守护没有执行取消、删除锁或其他资源写操作。它已经不再持有 hnode01 的第三张 H100；09:41 调度包装器也已删除临时遗留的 `try_lock_358384`、`after_lock_358384` 与 allocation 活动文件。未经新的用户授权，不重提任务，也不尝试接管空出的资源。

## Training Identities

| Job | 训练身份 | 资源 | 运行目录与 W&B | 当前状态 |
| ---: | --- | --- | --- | --- |
| `366071` | Find_1 PDB-centric-1 | hnode02，H100×1，CPU×32 | `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_1/Find_1-pdb_centric_1____Find_1_job366071_20260901T102343_a11_pdb_centric_1`；`wi4gcvcs` | a11 正常训练；第七次 validation 新高 `0.5385515`，plateau best 同步更新，11:50 已到 `global_step=8378` |
| `368455` | Find_1 PDB-centric-2，即全局采样方式三 | hnode01，H100×2，CPU×64 | `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_2/Find_1-pdb_centric_2____Find_1_pdb_centric_2_job368455_20260903T233745_a1_pdb_centric_2`；`g7ul4r8p` | a1 正常双卡训练；第三次 validation `0.5468278`，仍处于 warmup，18:06 已到 `global_step=3635` |
| `366277` | Find_1 指定历史 checkpoint 两节点续训 | gnode09、gnode10 各 A800×1、CPU×17 | `/home/penghongen/Feedback/Pocket_Plus_Find1/logs/AdaLigand_Stage1_resume-Find_1-CPC1/Find_1-resume_last_job351295____Find_1_job366277_20260902T182829_a3_CPC1`；`9kj96nld` | a3 正常训练；第三次真实续训 validation 每 rank 551 batches、PRAUC `0.6125641`，best 仍为 `0.6297938`，14:57 已到 `global_step=14777` |
| `358384` | 采样方式三 `unet_c1`；训练与推理均已完成 | 已于 08:35:30 被外部取消，不再持有 H100 | 训练根 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal`；`9dsi7i9w` | Slurm `CANCELLED`；09:41 已完成活动锁清理；不自行重提资源 |
| `350305` | occurrence-centric `unet_diff` 密度通道消融 | 已释放，不再占用 gnode10 资源 | `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal`；`nf93buae` | 训练已正常完成；最终 checkpoint step 46282、实际衰减 3 次；用户完成产物验收后已释放 Job，活动锁已经清理 |

## Job 367906 Evidence

- 用户提交的可读命令为 `bash 训练与运行/submit_task.sh --sh Find_1_pdb_centric_2.sh --resource h100 --gpus 1 --cpus 32`；Slurm 记录 `pre_hold=0`、`after_hold=1`、时限无限。
- release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_a8145b759454/Pocket_Plus`。
- launch：`/home/penghongen/Feedback/Pocket_Plus/launches/367906/Find_1_pdb_centric_2_job367906_20260903T134647_a1`。
- `run_cmd.sh` SHA-256：`26684e97b2f238aba7dd04bc1ecbb43c7b554924cb606d7438e1aeab7c1a3591`。
- 运行目录 `config.yaml`、`train.yaml`、`src_snapshot/manifest.json` 的 SHA-256 依次为 `d743998d0ffc2f7a5d1c7e3b8038080896737b1fc78fac355a3cd90dfc20647b`、`e64eea2cebcc00a2220dbb7311d2c6b75beef8a2bd80bb4e8dccb1b5a06101db`、`7e82ee4503848d24e416f5d028babf4f3b40b9904c22be0ff2224acf728a5d53`。
- V2 validation 文件为 `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric_v2.npz`，SHA-256 `0c92a731a7676f8083a250ff54e13dd2356e2cc462e383f6ad45c4c30de5fda2`；包含 200 个 PDB、3,305 个 bias BOX 和 3,200 个 context BOX。
- 科学配置核验通过：每个 PDB 目标 50 个前景 BOX、前景比例 25/33、occurrence 上限 1、补充 16 个 context BOX；物理 batch 6、全局 batch 48、30 workers、110 epochs、每 epoch 8 次 validation、学习率 `5e-5`、`patience=3`、实际学习率衰减 2 次后停止、voxel/point 两组各 0.5 梯度裁剪。
- a1 日志显示 Lightning 没有发现 CUDA，随后 FlashAttention 因收到 CPU 张量失败。allocation 内带 `--gres=gpu:h100:1` 的只读步骤得到 `CUDA_VISIBLE_DEVICES=0`、`SLURM_STEP_GPUS=1`，但 `nvidia-smi` 对 `0000:C5:00.0` 返回 `Unknown Error`，PyTorch 返回 `cuda.is_available() == False`、`device_count == 0`。
- 14:50 时 hnode02 的三张 H100 分别由 Jobs `360968`、`366071`、`367906` 占用，H100 分区还有其他用户 Job `367907` 等待资源；当前没有可用于双卡转换的第二张 H100。
- 本次核验没有删除 `try_lock_367906`、没有重启 a2、没有取消 Job，也没有触碰其他作业或资源。

## Job 367917 Replacement

- 执行前发现 Job `367906` 已在 14:53:03 被账号 UID 1351 取消，旧 allocation 和锁均已由包装器清理；本守护没有执行该动作。
- 14:56:56 执行 `bash 训练与运行/submit_task.sh --sh Find_1_pdb_centric_2.sh --resource h100 --gpus 1 --cpus 32 --pre_hold --after_hold`，得到 Job `367917`。
- 新 Job 于 14:56:57 在 hnode02 启动，根级 `pre_lock_367917` 与 allocation 内 `after_lock_367917` 均存在。`run_cmd_367917.sh` SHA-256 为 `26684e97b2f238aba7dd04bc1ecbb43c7b554924cb606d7438e1aeab7c1a3591`。
- allocation 内检查仍得到物理 GPU `0000:C5:00.0`、NVML `Unknown Error`、PyTorch `cuda_available=False` 和 `device_count=0`。因此没有删除 `pre_lock`，也没有产生失败的第二次训练运行。
- Job `366071` 于同一事件窗口完成第五次完整 validation：配体体素 PRAUC `0.4847679`，TOP/last SHA-256 均为 `da7ec571ca840b9036594ea6ffebaf1c400fcd674569a7be8ab5f9ace697dfac`；BEST 继续指向第四次的 `0.5078242`，训练随后恢复。
- Job `367917` 后于 15:06:43 被账号 UID 1351 取消并完成锁清理；本守护没有执行该动作。用户在 15:41:36 提交带 `--nodelist hnode01` 的 Job `367928`；15:45 时该 Job 正常等待资源，尚未创建任何运行产物。

## Job 358384 训练完成与资源保留

- 15:45—15:46 +08:00，训练在第二次实际学习率衰减后按 `stop_after_lr_reductions=2` 正常停止；包装器记录总时长 540,795.94 秒和第一次执行成功。W&B run `9dsi7i9w` 最终为 epoch 3、`trainer/global_step=36349`、学习率 `2e-5`、验证总损失 `0.1856628805`、配体体素 PRAUC `0.5956817269`、受体 PRAUC `0.6590440273`。
- 训练产物根为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal`。`BEST.ckpt`、`TOP_epoch_03_score_0.5957.ckpt` 和 `last.ckpt` 均为 499,678,098 bytes，SHA-256 均为 `8e8bd068ebbeb8b49db3ddd40c6c485bad322e75f3308b39298b8cea7bec3a`。
- 原训练 release 与 launch 分别为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_58b27dd765ed/Pocket_Plus` 和 `/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260828T093205_a1`；`launch.json` 与 `run_cmd.sh` SHA-256 分别为 `e891f5f13519f2c3e2ecc46d5af6938533ac385ce0ac6b528bf7695accb6e116` 和 `c365ce608826031abb21963f5fb51b1b90cea67cc7d83b40fa8e633e40e0240a`。
- 第一次执行成功后，另一条工作消费了 `try_lock_358384`，以 release `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_d57060538839/Pocket_Plus` 和 launch `/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260903T161305_a2` 启动采样方式三的推理；`run_cmd.sh` SHA-256 为 `67948adeff133468605e6bc1dda25921da4e37d9b0fed06cc36ea1252afdcac3`。16:13 时 calibration probability 正在写入 `/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1_pdb_centric_v2/artifacts`。
- `after_lock_358384` 保持存在。用户最新指令取代此前的释放安排：未经新的明确授权，不删除该锁、不取消 Job `358384`、不停止 allocation 内正在运行的推理，也不为了 Job `367928` 腾卡。

## Job 350305 新最佳 validation

- 2026-09-03 20:48，Job `350305` 完成一轮 827-batch validation：总损失 `0.1614618599`、配体体素 PRAUC `0.6319023967`、受体 PRAUC `0.7162516713`。
- 相比上一轮 `0.6288676858`，配体体素 PRAUC 提高 `0.0030347109`，严格超过绝对改进阈值 `0.003`。`last.ckpt` 中 plateau 的 `best` 已更新为新分数，`num_bad_epochs=0`；学习率仍为 `4.000000000000011e-06`，实际衰减计数仍为 2。
- `TOP_epoch_00_score_0.6319.ckpt` 与 `last.ckpt` 均为 499,688,614 bytes，SHA-256 均为 `269d9c2be071d73191b8a2751cb53d9128bcb789cf4e980aa86fa47f88dead89`。`BEST.ckpt` 与旧 `TOP_epoch_00_score_0.6289.ckpt` SHA-256 均为 `ce837c1e728cffe1a99b04848736d02b64e82ecf461fada09c0fc03bc5570f81`；checkpoint 状态已正确指向新 TOP，磁盘 `BEST.ckpt` 只是既知的一轮发布滞后。
- release、launch、正式运行根和 W&B 分别为 `Pocket_Plus_fdb8a30fa196`、`unet_diff_job350305_20260822T171529_a1`、`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal`、`pencounkdual-111/AdaLigand_Stage1/nf93buae`。immutable `launch.json` 与 `run_cmd.sh` SHA-256 分别为 `f5852fe2c7ec78da3f03250c2d7c47879b5708abac5f5c89fca284a4713611f0`、`29c6433271a5f46c8403d5d2a9b9d7de6c3f1ce460b12a9e4c4714b738cedfc5`。
- 21:52 时训练已恢复并推进到至少 `global_step=41915`。核验只读完成，没有操作 Job、进程、checkpoint 或锁。

## Job 350305 训练完成并释放资源

- 2026-09-05 05:00，最终 827-batch validation 得到总损失 `0.1607799530`、配体体素 PRAUC `0.6330414414`、受体 PRAUC `0.7154738903`。该分数没有超过 plateau best `0.6319023967` 加绝对阈值 `0.003`，因此第 3 次实际学习率衰减正常发生，并触发配置化停训。
- 最终 `last.ckpt` 为 epoch 1、`global_step=46282`、完整 validation 进度 827/827；优化器学习率为 `8.000000000000022e-07`，实际衰减计数为 3。标准输出明确记录总时长 `1,165,496.32` 秒、正式训练完成和第一次执行成功。
- `last.ckpt` 与 `TOP_epoch_01_score_0.6330.ckpt` 的 SHA-256 均为 `48ad3b374c32e34edad21a7e9127cf875a5bd42c21327821f84e2b4d45c7dbfd`。原始最佳 `BEST.ckpt` 与 `TOP_epoch_00_score_0.6344.ckpt` 的 SHA-256 均为 `8777e7a58ffdb6f17914329a3a2bbdc1bb478b345b6da8b570f50108c9c51409`。
- release 为 `/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_fdb8a30fa196/Pocket_Plus`；launch 为 `/home/penghongen/Feedback/Pocket_Plus/launches/350305/unet_diff_job350305_20260822T171529_a1`；正式产物根为 `/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal`，其下 `checkpoints/`、`src_snapshot/`、`wandb/run-20260822_172939-nf93buae/` 分别保存 checkpoint、冻结源码和 W&B 本地产物。
- immutable `launch.json` 与 `run_cmd.sh` SHA-256 分别为 `f5852fe2c7ec78da3f03250c2d7c47879b5708abac5f5c89fca284a4713611f0`、`29c6433271a5f46c8403d5d2a9b9d7de6c3f1ce460b12a9e4c4714b738cedfc5`。05:01:25 创建的根级 `try_lock_350305` 和 allocation 内 `after_lock_350305` 已在用户释放 Job 后由调度包装器清理；训练产物保持原位。

## PDB-centric-2 单卡到双卡转换

- Job `367928` 于 22:42:57 在 hnode01 获得单 H100、CPU×32，并停在 `pre_lock_367928`。23:28:31 按既定授权核验身份后，只删除该任务自己的根级 `pre_lock`，直接执行 immutable `run_cmd.sh`；没有增加 GPU 门禁，也没有碰其他 Job。
- 单卡 a1 的 release、launch、运行根依次为 `Pocket_Plus_d57060538839`、`Find_1_pdb_centric_2_job367928_20260903T232856_a1`、`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_2/Find_1-pdb_centric_2____Find_1_pdb_centric_2_job367928_20260903T232856_a1_pdb_centric_2`。`launch.json` 与 `run_cmd.sh` SHA-256 分别为 `4e992b6ef9f797275cfa537bf89b4be172bf1fe4b49a05baf2529a0859872719`、`26684e97b2f238aba7dd04bc1ecbb43c7b554924cb606d7438e1aeab7c1a3591`。该执行在创建 W&B 或 checkpoint 前被用户取消。
- 用户于 23:35:39 先提交双卡 Job `368455`；Slurm 在 23:37:06 为其分配 hnode01 的 H100×2、CPU×64。`sacct` 同时记录单卡 `367928` 于 23:37:00 被 UID 1351 取消，因此转换顺序满足“先取得双卡、后取消单卡”。已取消 Job 只保留历史证据，不再执行资源动作。
- 双卡 a1 的 release、launch、运行根和 W&B 分别为 `Pocket_Plus_2ca40d551603`、`Find_1_pdb_centric_2_job368455_20260903T233745_a1`、`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-Find_1-pdb_centric_2/Find_1-pdb_centric_2____Find_1_pdb_centric_2_job368455_20260903T233745_a1_pdb_centric_2`、`pencounkdual-111/AdaLigand_Stage1/g7ul4r8p`。
- 双卡 immutable `launch.json` 与 `run_cmd.sh` SHA-256 分别为 `89147969e47d4f561d88faad8732eb615ba9e4661d5ac517c853c6c72fb448ab`、`26684e97b2f238aba7dd04bc1ecbb43c7b554924cb606d7438e1aeab7c1a3591`。`config.yaml`、`train.yaml`、`src_snapshot/manifest.json` SHA-256 分别为 `b7bb25254502e6753a3239a08b8cf37a5cbad25dc123476b5dc2be9c8329ebd5`、`9a655a76e8e9efb47f9160c7065cc3a0f01d309a4fb24349e3c1a55700498f98`、`ee159ed4d9cb9083d3bdc018e9f0cf469cf29784ee67fcb474c145715f746be2`。
- 与旧单卡验收配置的完整递归比较只有三处差异：`devices 1→2`；用户手动并明确确认的 `stop_after_lr_reductions 2→3`；只影响 checkpoint 保留数量的 `save_top_k 30→50`。训练 `src/` 清单逐文件完全一致。第二次实际学习率衰减后的完整 checkpoint 可在需要时作为 2 次衰减版本的选择端点，当前运行保持 3。
- 两条 NCCL rank 已注册，`world_size=2`、每卡 batch 6、梯度累积 4、全局 batch 48、每 rank 30 workers。23:57 时 W&B 达到 `global_step=50`；GPU 0/1 显存约 80 GiB 且持续计算，没有 traceback、OOM、NCCL 或非有限值错误。`after_lock_368455` 保持存在。

## 2026-09-04 Validation And Resource Milestones

- Job `368455` 首次完整 validation：每 rank 543 batches，`num_gt=13,835,754`，配体体素 PRAUC `0.4649433`；`TOP...0.4649` SHA `c3ad5b38...945df1`，`last` SHA `565ed000...8b0dc`。checkpoint 仍处于 warmup，实际学习率衰减 0 次。
- Job `366071` 第六次完整 validation：1,250 batches，`num_gt=8,156,584`，配体体素 PRAUC `0.5093884`；`TOP...0.5094` 与 `last` SHA 均为 `7b331145...3cdcb`。plateau best 更新，LR 仍为 `5e-5`，实际衰减 0 次。
- Job `350305` 新一轮完整 validation：827 batches，原始 PRAUC `0.6343954`；相对 plateau best 的提高只有 `0.0024930`，低于阈值 `0.003`，因此 bad epoch 为 1、LR 仍为 `4e-6`、实际衰减仍为 2 次。新 TOP 与 last SHA 均为 `8777e7a5...51409`。
- Job `366277` 第二次真实续训 validation：每 rank 551 batches，`num_gt=14,802,569`，PRAUC `0.6297938`；plateau best 更新、bad epoch 归零、LR 保持 `5e-5`。新 TOP 与 last SHA 均为 `eadba47d...7964`，恢复边界继续正确。
- Job `358384` 推理于 06:03 成功结束；Slurm 于 08:35:30 将它记录为外部取消，调度包装器在 09:41 前完成活动锁清理。详细状态、哈希、解释与下一步见 `CLAUDE/memory/handoffs/2026-09-04-stage1-four-training-validation-milestones.md`。
- 11:23，Job `368455` 完成第二次完整 validation，PRAUC 提高至 `0.5264485`；`TOP...0.5264` 与 `last` SHA 均为 `63c236bd...a5625`。仍处于 warmup，实际衰减 0 次。
- 11:33，Job `366071` 完成第七次完整 validation，PRAUC 提高至 `0.5385515`；`TOP...0.5386` 与 `last` SHA 均为 `bbf388a2...e7b1a`。plateau best 合法更新，LR 保持 `5e-5`，实际衰减 0 次。
- 12:50，Job `350305` 完成下一轮完整 validation，PRAUC `0.6314629`；`TOP...0.6315` 与 `last` SHA 均为 `70150347...cc1c3`。plateau bad epochs 增至 2，LR 保持 `4e-6`，实际衰减仍为 2 次。
- 13:21，Job `366277` 完成第三次真实续训 validation，每 rank 551 batches，PRAUC `0.6125641`；`TOP...0.6126` 与 `last` SHA 均为 `5c320e83...dc065`。plateau bad epochs 为 1，LR 保持 `5e-5`，实际衰减仍为 0。
- 17:04，Job `368455` 完成第三次完整 validation，每 rank 543 batches、累计 1,629，PRAUC `0.5468278`；`TOP...0.5468` 与 `last` SHA 均为 `f85b1beb...63620b`。checkpoint 仍在 warmup，plateau 尚未启动，实际衰减仍为 0。

## Resource Decisions

- PDB-centric-2 的单卡到双卡转换已经完成。活动权威 Job 固定为 `368455`；已取消的 `367928` 仅作为历史证据，不再提交、恢复或操作其锁。
- Job `368455` 使用 hnode01 的 H100×2、CPU×64，从头训练。未经新的明确授权，不停止该 Job、不删除 `after_lock_368455`，也不把诊断门禁或条件分支加入正式入口。
- Job `358384` 已完成训练和推理，并被外部取消；本守护没有执行该动作，调度包装器已经完成活动锁清理。未经用户新的明确指令，不得自行重提 Job 或占用目前空出的第三张 H100。
- Job `350305` 已正常结束训练，并由用户在产物验收后释放；活动锁与动态命令均已清理，不重新提交该消融训练。
- Jobs `366071`、`366277` 继续遵守各自既有 after-hold 和故障恢复契约。
- 每次醒来固定检查 Jobs `368455`、`366071`、`366277` 的训练、错误与资源状态。Jobs `350305`、`358384` 均已结束并完成资源清理，不属于固定训练检查清单。稳定阶段采用由多个独立 300 秒命令组成的 60 或 90 分钟静默等待，不创建 heartbeat。普通 validation、epoch 完成和常规 checkpoint 只用于健康判断，不更新 handoff；任务提交或替换、训练完成、故障修复与重启、资源交接、科学契约变化等阶段事件才更新 handoff。

## Logs Updated

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/mapping/计划执行映射.md`
- `ops/stage1_data_preparation/EXECUTION.md`
- `文档/exec_plan/2026-08-21_unet密度通道消融守护.md`
- `CLAUDE/memory/handoffs/2026-08-28-stage1-pdb-centric-v2-unet-c1-running.md`
- `CLAUDE/memory/handoffs/2026-09-03-stage1-pdb-centric-v2-unet-c1-complete-held.md`
- `CLAUDE/memory/handoffs/2026-08-21-unet-density-ablation-guard.md`
- `CLAUDE/memory/handoffs/2026-09-03-unet-diff-validation-new-best.md`
- `CLAUDE/memory/handoffs/2026-09-03-find1-pdb-centric2-dual-h100-running.md`
- `CLAUDE/memory/handoffs/2026-09-04-stage1-four-training-validation-milestones.md`

## Next Actions

1. 每次醒来固定检查 Job `368455` 的双 rank 训练进度、GPU 活动、validation、checkpoint、错误与锁状态；只在明确事件发生时集中记录。
2. 同一次醒来检查 Jobs `366071` 与 `366277`；普通 validation、epoch 与 checkpoint 不写 handoff，只在故障、训练完成、重启、资源变化或科学契约变化时集中记录。
3. 对已经外部取消并完成清理的 Job `358384` 只做冲突防护，不自行重提任务。若用户以后授权重新申请资源，再按当时状态拟定提交动作。
4. Job `368455` 在第二次实际学习率衰减时保留并核验对应完整 checkpoint；训练继续遵循用户确认的三次实际衰减停训配置。
5. 剩余三条活动训练最终完成后更新对应执行记录与 handoff，再统一完成 Find_1 的 Git 双线收口。

## Files To Reopen

- `文档/规划文档/Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-31_Find_1训练与历史续训.md`
- `文档/exec_plan/2026-08-21_unet密度通道消融守护.md`
- `ops/stage1_data_preparation/EXECUTION.md`
- `CLAUDE/memory/handoffs/2026-09-02-find1-a800-timeout-recovered.md`
- `CLAUDE/memory/handoffs/2026-09-03-stage1-pdb-centric-v2-unet-c1-complete-held.md`
- `CLAUDE/memory/handoffs/2026-08-21-unet-density-ablation-guard.md`
- `CLAUDE/memory/handoffs/2026-09-04-stage1-four-training-validation-milestones.md`
