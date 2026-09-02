

我们必须找准当下的实现坐标、发现、进展与问题，这是我们决定接下来怎么做的基础。
我说的以下内容是基于我的观察,它们应该在handoff、相应执行日志里面有记录————请以实际产物、运行代码等最基础事实为准。

# 小提示
AI agent应该自己先仔细了解我们整体项目的坐标：已经做了什么、正在做什么、还有什么没做。包括但不限于阅读执行日志、handoff记忆系统等等。

各个仓库在：
- "C:\Users\15919\Desktop\AdaLigand" (数据&契约)
- "C:\Users\15919\Desktop\Pocket_Plus"(stage1)
- "C:\Users\15919\Desktop\Matcher"(stage2)
- "C:\Users\15919\Desktop\Builder"(stage3)


——————————————————————————————————————————————————— 以下写于 8.26 日, 注意日期，注意可能的过时风险 ———————————————————————————————————————————————————


# Stage1:

目前训练对应三种采样方式。


## 训练采样方式一： occurrence-centric式采样，对应 Pocket_Plus仓库 8561d2790614a0d92f0ad88ebce241a9f1c77ba3 这个commit里面的dataset采样逻辑。

这一部分的实验清单如下(wandb 的 run path)：
(1).unet_c1: pencounkdual-111/AdaLigand_Stage1/hqumqkex         (训完了)
- 训练运行目录：`/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/runtime/mainchain_official_346737/logs/AdaLigand_Stage1-unet_c1-mainchain/unet_c1_mainchain____tmp_stage1_mainchain_job346737_20260818T035126_a4_formal`
- W&B 本地运行目录：`/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/runtime/mainchain_official_346737/logs/AdaLigand_Stage1-unet_c1-mainchain/unet_c1_mainchain____tmp_stage1_mainchain_job346737_20260818T035126_a4_formal/wandb/run-20260818_035841-hqumqkex`
- 产物：
  - 最佳训练检查点：`/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/runtime/mainchain_official_346737/logs/AdaLigand_Stage1-unet_c1-mainchain/unet_c1_mainchain____tmp_stage1_mainchain_job346737_20260818T035126_a4_formal/checkpoints/BEST.ckpt`
  - complete-map 推理与评估产物根目录：`/storage/penghongen/AdaLigand_stage1_inference/UNET/unet_c1-mainchain-ligand_PRAUC_0.602950`
- 训练所用的源码快照：`/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/runtime/mainchain_official_346737/logs/AdaLigand_Stage1-unet_c1-mainchain/unet_c1_mainchain____tmp_stage1_mainchain_job346737_20260818T035126_a4_formal/src_snapshot/src`
- 训练所用的 release：`/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/feedback/releases/Pocket_Plus_8139f528eebc/Pocket_Plus`
- 启动留证：`/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/feedback/launches/346737/tmp_stage1_mainchain_job346737_20260818T035126_a4/launch.json`
- 训练所用的配置：`/storage/penghongen/tmp/stage1_v3_ablation_replacement_20260817T1845/runtime/mainchain_official_346737/logs/AdaLigand_Stage1-unet_c1-mainchain/unet_c1_mainchain____tmp_stage1_mainchain_job346737_20260818T035126_a4_formal/config.yaml`

unet_diff: pencounkdual-111/AdaLigand_Stage1/nf93buae         (正在训)
- 训练运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal`
- W&B 本地运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal/wandb/run-20260822_172939-nf93buae`
- 产物目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal/checkpoints`
- 训练所用的源码快照：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal/src_snapshot/src`
- 训练所用的 release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_fdb8a30fa196/Pocket_Plus`
- 启动留证：`/home/penghongen/Feedback/Pocket_Plus/launches/350305/unet_diff_job350305_20260822T171529_a1/launch.json`
- 训练所用的配置：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_diff/unet_diff____unet_diff_job350305_20260822T171529_a1_formal/config.yaml`

unet_base: pencounkdual-111/AdaLigand_Stage1/z6ncwfag       (正在训)
- 训练运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_base/unet_base____unet_base_job350302_20260821T084104_a1_formal`
- W&B 本地运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_base/unet_base____unet_base_job350302_20260821T084104_a1_formal/wandb/run-20260821_085232-z6ncwfag`
- 产物目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_base/unet_base____unet_base_job350302_20260821T084104_a1_formal/checkpoints`
- 训练所用的源码快照：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_base/unet_base____unet_base_job350302_20260821T084104_a1_formal/src_snapshot/src`
- 训练所用的 release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_ea0aca3a357c/Pocket_Plus`
- 启动留证：`/home/penghongen/Feedback/Pocket_Plus/launches/350302/unet_base_job350302_20260821T084104_a1/launch.json`
- 训练所用的配置：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-unet_base/unet_base____unet_base_job350302_20260821T084104_a1_formal/config.yaml`

Find_1: pencounkdual-111/AdaLigand_Stage1/a87fe1ef     (训练到一半临时中断)
- 训练运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1`
- W&B 本地运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/wandb/run-20260823_153253-a87fe1ef`
- 产物目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/checkpoints`
- 训练所用的源码快照：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/src_snapshot/src`
- 训练所用的 release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_9a4533e3bf4c/Pocket_Plus`
- 启动留证：`/home/penghongen/Feedback/Pocket_Plus/launches/351295/Find_1_job351295_20260823T152130_a2/launch.json`
- 训练所用的配置：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/config.yaml`





## 训练采样方式二： pdb-centric-1，使一个PDB中的前景box数目固定(25)，同时尽可能均分给各个occurrence，对应Pocket_Plus的 f806f12007e2cef521f71588103edf17318e8b4a 这个commit。


unet_c1: pencounkdual-111/AdaLigand_Stage1/6uuzdbgs
- 训练运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal`
- W&B 本地运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal/wandb/run-20260826_111809-6uuzdbgs`
- 产物目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal/checkpoints`
- 训练所用的源码快照：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal/src_snapshot/src`
- 训练所用的 release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_064fd1e46dcc/Pocket_Plus`
- 启动留证：`/home/penghongen/Feedback/Pocket_Plus/launches/356953/unet_c1_job356953_20260826T110721_a1/launch.json`
- 训练所用的配置：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric-unet_c1-mainchain/unet_c1_mainchain____unet_c1_job356953_20260826T110721_a1_formal/config.yaml`

### 方式二训练收口（2026-09-02 核验）

Job `356953` 确实是 `unet_c1` 的采样方式二 `pdb-centric-1` 正式训练，并且训练主体已经正常完成。Slurm 最终执行事实为 `nvlink` partition、`h200g4` QOS、`gnode09`、1 张 A800 和 24 CPU，提交契约为 `pre_hold=0`、`after_hold=1`。任务于 2026-08-26 11:09:47 +08:00 获得资源；训练包装器于 2026-09-01 06:48:18 +08:00 创建 `try_lock_356953`，标准输出明确记录两次实际学习率衰减触发正常停止、`unet_c1 mainchain` 正式训练完成以及第 1 次执行成功。

运行目录中的最终 `config.yaml` 而不是当前工作树配置是本次训练的权威配置。该文件确认输入只有 `exp_clipnorm_nopost` 一个实验密度通道；训练请求使用 `pdb_foreground_box_num=25`、`pdb_foreground_fraction_target=0.5`、`pdb_occurrence_foreground_box_cap=25`，验证读取 V1 `validation_selection_pdb_centric.npz`。每个 epoch 共有 685,850 个训练 BOX；V1 验证文件固定包含 150 个 PDB、3,750 个 bias BOX 和 3,750 个 context BOX。训练使用每卡 batch 8、全局 batch 48、梯度累积 6、24 个 `DataLoader` workers、学习率 `1e-4`、`max_epochs=70`、`val_per_epoch=12`、BF16 混合精度，以及配体区域、配体体素、蛋白主链、核酸主链和配体反距离五项损失，权重分别为 `0.1/1.0/0.05/0.05/0.3`。

训练共保存 20 个 TOP checkpoint。`last.ckpt` 的内部状态为 epoch 1、`global_step=23814`，已记录第二次学习率衰减并把优化器学习率降至约 `4e-6`；W&B 最终摘要记录的相邻日志步为 `trainer/global_step=23813`，最后一次验证总损失为 `0.1979677`、配体体素 PRAUC 为 `0.4891676`、受体 PRAUC 为 `0.5830685`。最高配体体素 PRAUC 为 `0.4918`，对应 `TOP_epoch_01_score_0.4918.ckpt`；`BEST.ckpt` 与该 TOP 文件的 SHA-256 均为 `0bb6bd5231c638c13dee5144bfb361b6bd20f5268bd10337a040c8486efa9b5a`，因此最终最佳别名已经闭合。





## 训练采样方式三：pdb-centric-2（V2 产物与训练实施中）

这里主要的思路是，在一个epoch内，每个pdb截断50个occurrence，每个occurrence只采样一个bias_box。这意味着每个occurrence的采样是等权的，但是不同于采样方式一，因为context box是PDB均分的，也不同于采样方式二，因为	PDB中occurence较少时也不会在epoch内占据相对更高的采样权重。对于bias和context box的比例，我不想用0.5描述"预期它们相等"———我想经过统计，设立一个（大于0.5的）参数，使得一个epoch内bias_box和context box仍然尽可能相同。

注意 ：

1. `C:\Users\15919\Desktop\Pocket_Plus\ops\stage1_data_preparation\freeze_validation_selection_pdb_centric_2.py` 使用全部 200 个 validation PDB，并采用方式三的三个采样参数。脚本只生成独立的 `validation_selection_pdb_centric_v2.npz`，不读取、删除或覆盖方式二的 V1 文件。

2. `C:\Users\15919\Desktop\Pocket_Plus\configs\dataset\stage1_unet_c1.yaml` 与 `C:\Users\15919\Desktop\Pocket_Plus\训练与运行\sh\unet_c1.sh` 中的方式三参数已经确定。正式训练 Job `358384` 于 2026-08-28 09:31:26 +08:00 在 `hnode01` 启动，当前处于稳定训练状态；实际路径和运行证据见下文。

3.总体想法很简单：先用unet_c1训完三种采样模式，在complete map级别表现最好的采样模式，用来训Find_1.

### 服务器数据坐标与固定统计

以下路径和数量在 2026-08-26 通过只读命令核对：

- Stage1 V3 数据准备根目录：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3`
- BOX pool 根目录：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool`
- BOX pool manifest：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/manifest.json`
- 训练 PDB pool：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/train`
- 验证 PDB pool：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation`
- BOX pool 构建统计来源：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/run_state/box_pool/shard_000_of_012.json` 至 `shard_011_of_012.json`
- 方式二 V1 验证选择：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric.npz`
- 方式三 V2 验证选择：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric_v2.npz`
- PDB 划分目录：`/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/split`

方式二 V1 文件固定保存 150 个 PDB、3,750 个 bias BOX 和 3,750 个 context BOX，SHA-256 为 `546ebd3a1f07b230af42911b6740f466c6af289c8bff91a387c4eb8b1d69dd8e`，继续服务 Find、unet_base 与 unet_diff。方式三使用独立 V2 文件，固定保存全部 200 个 PDB、3,305 个 bias BOX 和 3,200 个 context BOX；两份文件长期共存。

### 采样方式三已确定的训练参数

训练划分包含 13,717 个 PDB。按照 `pdb_foreground_box_num=50` 和 `pdb_occurrence_foreground_box_cap=1`，一个 epoch 实际产生 216,739 个 bias BOX。当前采样框架只能为每个 PDB 指定同一个整数 context BOX 数量；取 16 个时，一个 epoch 产生 219,472 个 context BOX，比 bias BOX 多 1.26%，是可选整数中最接近全局等量的结果。为得到每个 PDB 16 个 context BOX，`pdb_foreground_fraction_target` 固定为 `25/33` 的十进制值 `0.7575757575757576`。

采样方式二每个 epoch 有 685,850 个训练 BOX，`val_per_epoch=12`，相邻两次验证之间约有 57,154 个训练 BOX；采样方式三每个 epoch 有 436,211 个训练 BOX，取 `val_per_epoch=8` 后，相邻两次验证之间约有 54,526 个训练 BOX，相差 4.60%。采样方式二在 `max_epochs=70`、`warmup_ratio=0.005` 下约有 240,048 个 warmup BOX；采样方式三固定 `max_epochs=110`、`warmup_ratio=0.005` 后约有 239,916 个 warmup BOX，相差 0.05%。

若方式三只冻结 150 个 PDB，预计共有 4,871 个验证 BOX，比方式二的 7,500 个少 35.05%。因此方式三改用 validation manifest 的全部 200 个 PDB：冻结 3,305 个 bias BOX 和 3,200 个 context BOX，共 6,505 个 BOX，仍比方式二少 13.27%。候选选择固定使用 `request_seed=3407`。

正式训练由 `训练与运行/sh/unet_c1.sh` 启动，固定使用 1 张 H100、32 CPU、30 个 DataLoader workers、每卡 batch 8、全局 batch 48、学习率 `1e-4`、`max_epochs=110`、`val_per_epoch=8` 和 `warmup_ratio=0.005`。提交时使用 `--after_hold`，不使用 `pre_hold`。三种 unet_c1 采样实验最终只在同一 complete-map 数据与指标契约下比较，不直接以各自 BOX validation 分数决定胜负。

### 方式三正式训练坐标

- Slurm Job：`358384`，单节点、1 张 H100、32 CPU；`pre_hold=0`、`after_hold=1`。
- 启动留证：`/home/penghongen/Feedback/Pocket_Plus/launches/358384/unet_c1_job358384_20260828T093205_a1/launch.json`
- 训练所用 release：`/home/penghongen/Feedback/Pocket_Plus/releases/Pocket_Plus_58b27dd765ed/Pocket_Plus`
- 正式运行目录：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal`
- 训练所用源码快照：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal/src_snapshot/src`
- 训练所用配置：`/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal/config.yaml`
- W&B run：`pencounkdual-111/AdaLigand_Stage1/9dsi7i9w`

启动验收已从 release 的生产请求入口重新确认每个 epoch 共有 436,211 个训练 BOX，V2 共有 6,505 个验证 BOX。前二十三次完整 validation 均已结束，任务处于 epoch 2；第二十三次验证总损失创新低至 `0.1895426`，配体体素 PRAUC 创新高至 `0.5819030`，受体 PRAUC 创新高至 `0.6446361`。`TOP_epoch_02_score_0.5819.ckpt` 与更新后的 `last.ckpt` 已生成；配体体素 PRAUC 较旧基准提高约 `0.00549`，超过绝对改进阈值 `0.003`，调度器已在学习率 `2e-5` 下把新基准更新为 `0.5819030`。`BEST.ckpt` 暂时保持此前的 `TOP_epoch_02_score_0.5764.ckpt`，符合已确认的一次 validation 回调滞后。第一次实际学习率衰减计数保持不变；第二次实际衰减仍将触发正常停训。任务继续训练且未发现显式错误。







# Stage2-Matcher 
- Stage2 一上来就选择了多模态的策略以预测配体和候选的匹配关系: (配体; A, P, PP-额外采样原子, Map-用来调制&摘要)，目前只用了10%不到的"可能可用数据"就达到了约70%的精度(e2e_f1)，我认为作为本科研项目的组件，我判断：这在科学性上已经可以算作大致成功了，但是仍然需要适配新的stage1推理数据并用更多的数据去训练。
- 我通过 “# NOTE”的形式记录下了我对Matcher仓库处理&适配数据的看法。Matcher需要据此修改。
- "C:\Users\15919\Desktop\Matcher" 是本地仓库。
- codex://threads/019ff669-ea33-73d1-8048-9204ef5a9c11 (Matcher的端到端实现、训练&测评)这个 AI agent 负责了代码的撰写和端到端的调参与评估。






# Stage3-Builder
- Stage3已经完成了: 当前项目的 ligand_object ————> PocketXmol 能吃的数据的转化, 并依此进行了前向/推理等价的测试。仓库在 "C:\Users\15919\Desktop\Builder"。
- 以上任务由 codex://threads/019fdafa-7f7f-7e53-af06-503846d2ad20 (严格忠于 PocketXmol 的适配) 这个 AI agent 完成。

总体上，有四类值得注意的 issue：
1.PocketXmol 受体只能是蛋白, 我们需要扩展到核酸。
2.PocketXmol 的配体基本上是小分子 small molecule和多肽, 我们需要扩展到金属、糖类配体、核苷酸类配体。 
3.要让 PocketXmol 能直接利用密度图。
4.要让 PocketXmol 能利用 stage1 推理所得的多模态 (配体; A, P, PP-额外采样原子, Map-用来调制&摘要)。
