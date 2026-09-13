

我们必须找准当下的实现坐标、发现、进展与问题，这是我们决定接下来怎么做的基础。
我说的以下内容是基于我的观察,它们应该在handoff、相应执行日志里面有记录————请以实际产物、运行代码等最基础事实为准。

# 小提示
AI agent应该自己先仔细了解我们整体项目的坐标：已经做了什么、正在做什么、还有什么没做。包括但不限于阅读执行日志、handoff记忆系统等等。

各个仓库在：
- "C:\Users\15919\Desktop\AdaLigand" (数据&契约)
- "C:\Users\15919\Desktop\Pocket_Plus"(stage1)
- "C:\Users\15919\Desktop\Matcher"(stage2)
- "C:\Users\15919\Desktop\PocketXMol"(stage3)


——————————————————————————————————————————————————— 以下写于 9.12 日, 注意日期，注意可能的过时风险 ———————————————————————————————————————————————————

首先请注意, C:\Users\15919\Desktop\Pocket_Plus\talk\global 实际上是一串随时间变化的global规划或关键事实清单。我可能随着项目进展删掉上一版global_xxx.md的部分内容，但这不意味着错误，而是为了突出新版的重点(如C:\Users\15919\Desktop\Pocket_Plus\talk\global\global_8.26.md 中记录的训练事实仍然没错)，工程师、AI agent仍然可以查看历史版本作为额外信息。

# Stage1-Find
## 去冗余 & 测试集的构建

1. 首先声明基础设施。

Stage1的两个有效模型已经训练完毕:
unet_c1模型(只用密度图训练): /home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_2-unet_c1-mainchain/unet_c1_mainchain_pdb_centric_2____unet_c1_job358384_20260828T093205_a1_formal/checkpoints/TOP_epoch_03_score_0.5957.ckpt
Find模型(使用密度图+真实受体结构训练, 事实上是Find_1): /home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1_pdb_centric_resume-Find_1-pdb_centric_2/Find_1-pdb_centric_2_resume____Find_1_pdb_centric_2_job368455_20260909T071535_a2_pdb_centric_2_resume/checkpoints/TOP_epoch_04_score_0.6654.ckpt

我们已经有了2个独立测试集: 
/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_0.json
/storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_1.json
虽然前者包含后者(这意味着推理产物可以复用), 但它们在逻辑上本质上仍然是独立的。

我们有CryoAtom2在测试集/校准集上的预测结构:
/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06
/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration

这对于Stage1本身的测评, 以及后续Stage2/Stage3的"输入基础设施", 已经足够了。


2. unet_c1的推理
(1). 校准集(cal)上产生概率图 ——————> 选择最优的语义F_1截断概率阈值、"basic打分"参数使得beta=1的三项和(语义F_beta + covF_beta + 1to1F_beta) ——————> 在测试集上评分


3. 推理与测评————————使用真实受体结构。

它分为三条线:
(1). 第一类打分: 校准集(cal)上产生概率图 ——————> 选择最优的语义F_1截断概率阈值、"basic打分"参数使得beta=1的三项和(语义F_beta + covF_beta + 1to1F_beta) ——————> 在测试集上评分
(2). 第二类打分: 校准集(cal)上产生概率图 ——————> 选择最优的语义F_2截断概率阈值———>进行centered推理———>选"高斯打分"参数使得beta=1的三项和(语义F_beta + covF_beta + 1to1F_beta) ——————> 在测试集上评分
(3). 对第二类打分的进一步扩展: 利用选好的参数在验证集(val)、校准集(cal)上均完成"全套流程": 划窗推理、centered推理、高斯打分并在centered.npz里面落盘"是否被高斯打分选中"。
# TODO: (4). 利用现有的基础设施和官方入口，对部分训练样本以真实受体进行"全套流程"，产物放到 /storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/train 。 也就是说，和之前的验证集val一样，进行划窗推理、centered推理、高斯打分并在centered.npz里面落盘"是否被高斯打分选中"。默认把训练样本划分为50片（1,..,50），实际推理过程中会"一片片地"进行推理：比如我们首先推理第1、2片————————我们可能下次会要求"使用这个卡完成第3~5片的全套推理"，实际运行时，必须满足这个要求且保证延续性！

它们事实上先后线性执行, 前两步已经完成了"已知GT-受体结构时Find模型的性能测试", 其余步骤为了给Stage2、Stage3准备训练/验证/校准/测评数据时所有的

值得注意的是: 上述写法只是清晰展示实质逻辑, 真实产物会存在而且必须存在"复用"的情况。例如: 统一模型的概率图可通用、第三步对校准集的操作只需简单地"重算高斯分数"而不需要forward。






# Stage2-Matcher
## 第一阶段(如果只论效果则已合格)
- Stage2 一上来就选择了多模态的策略以预测配体和候选的匹配关系: (配体; A, P, PP-额外采样原子, Map-用来调制&摘要)，目前只用了10%不到的"可能可用数据"就达到了约70%的精度(e2e_f1)，我认为作为本科研项目的组件，我判断：这在科学性上已经可以算作大致成功了，但是仍然需要适配新的stage1推理数据并用更多的数据去训练——————这一部分见第二阶段。
- "C:\Users\15919\Desktop\Matcher" 是本地仓库。
- codex://threads/019ff669-ea33-73d1-8048-9204ef5a9c11 (Matcher的端到端实现、训练&测评)这个 AI agent 负责了代码的撰写和端到端的调参与评估。

## 第二阶段
- 我通过 “# NOTE”的形式记录下了我对Matcher仓库处理&适配数据的看法。Matcher需要按照这些建议, 以及最新的stage1推理管线产物进行修改。简而言之：删除冗余功能、重整与简化代码、对模型做少量受控的扩展。

- 模型本身可以进行一系列强化：
(1). 配体图的点特征构造的有点少了;
(2). 在对candidate进行分类时, 可以制作两个分类头: 一个专门分类"stage1推理打分超过阈值(score >= score_threshold)"的那些候选; 另一个分类头专门分类没有超过阈值的那些候选。这两个分类头当然按情况赋予损失, 且进行单独的阈值校准。
(3). 再次重申, 分类头的唯一作用是过滤假阳性。根据这个中心目的，有2点推论：
    [1]选择阈值时，"F1"时需要某种"全局的计算方式", 有很多可能的选择可以体现这一点, 但最简单粗暴的2种是：和stage1一样优化三项 macro F1 + macro covF1_0.3 + 1to1F1_0.3; 或者如果算起来麻烦, 只把stage1漏测的那部分算上(应该相当于优化 1to1F1_0.3)也可。 
    [2]我们实际上只关心"哪些是假阳性", 5种配体类别只算作中间的辅助标签, 所以"按照5个正类截断阈值"或"只按照假阳性截断阈值", 还是"把前两者结合起来"都可以。

- 如同我评论中反复强调的那样, 当前的代码过于复杂，且很不符合我的口味, 我读起来很困难，需要大幅删改——————甚至很多内容可以删了重写。目的很简单：一方面把没必要的逻辑/防御都删掉, 另一方面把"诊断性的逻辑/统计性的逻辑"等从代码出现位置的层面严格和主线代码分开————这样我可以选择不看而只给AI agent看，这样我就能省很多事且痛苦小得多。

- 我认为，如果满足一系列条件，就可以把当前的(去手性smile, ligand_object)定义的身份系统简化"单SMILES"的身份定义系统:
(0). SMILES定义的身份系统和当前的(去手性smile, ligand_object)定义的身份系统相比，"对具体配体occurrencec"派生的身份等价类很相似。这意味着在新的身份系统下可以基本保持当前结果不恶化。
(1). 需要修改的代码不多(我猜测, 多身份系统简化为单身份甚至可能简化代码);
(2). 需要修改的文件不多, 不需要修改或重构重构已有仓库的文件/科学契约。




# Stage3-Builder
- Stage3已经完成了: 当前项目的 ligand_object ————> PocketXmol 能吃的数据的转化, 并依此进行了前向/推理等价的测试。仓库在 "C:\Users\15919\Desktop\Builder"。
- 以上任务由 codex://threads/019fdafa-7f7f-7e53-af06-503846d2ad20 (严格忠于 PocketXmol 的适配) 这个 AI agent 完成。

总体上，有四类值得注意的 issue：
1.PocketXmol 受体只能是蛋白, 我们需要扩展到核酸。
2.PocketXmol 的配体基本上是小分子 small molecule和多肽, 我们需要扩展到金属、糖类配体、核苷酸类配体。
3.要让 PocketXmol 能直接利用密度图。
4.要让 PocketXmol 能利用 stage1 推理所得的多模态 (配体; A, P, PP-额外采样原子, Map-用来调制&摘要)。
