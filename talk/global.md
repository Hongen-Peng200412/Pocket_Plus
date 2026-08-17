

我们必须找准当下的实现坐标、发现、进展与问题，这是我们决定接下来怎么做的基础。
我说的以下内容是基于我的观察,它们应该在handoff、相应执行日志里面有记录————请以实际产物、运行代码等最基础事实为准。

# 小提示
AI agent应该自己先仔细了解我们整体项目的坐标：已经做了什么、正在做什么、还有什么没做。包括但不限于阅读执行日志、handoff记忆系统等等。

各个仓库在：
- "C:\Users\15919\Desktop\AdaLigand" (数据&契约)
- "C:\Users\15919\Desktop\Pocket_Plus"(stage1)
- "C:\Users\15919\Desktop\Matcher"(stage2)
- "C:\Users\15919\Desktop\Builder"(stage3)


——————————————————————————————————————————————————— 以下写于 8.17 日, 注意日期，注意可能的过时风险 ———————————————————————————————————————————————————


# Stage1:

- (清理旧版) 注意 a0f19f2e87e1ce3eee6b97e20d20d2b0172dfd2e 到 45b43d3203044abddd3f80e2ced49e879c440110 的学习分支————这是"按照PDB分桶训练"的实现。这么做的初衷是为了让原本是训练瓶颈的 I/O 得到节省，但是我们已经有充足的证据了: "按照PDB分桶训练" 会显著降低模型本身的性能，所以我们建议保留 codex/stage1-pdb-grouped-loading 作为历史实现，然后回退学习分支，先退回到原本的正常 box_pool 采样与加载。

- (清理旧版) 对于 C:\Users\15919\Desktop\Pocket_Plus\processedPDB_EMDB_binder 、 C:\Users\15919\Desktop\Pocket_Plus\Make_Data 、 C:\Users\15919\Desktop\Pocket_Plus\Docking、 C:\Users\15919\Desktop\Pocket_Plus\Bundle_of_Maps 等代码，记忆中它们不在现有的活跃主线。建议删掉，由 git 提供历史。

- (清理旧版&突出推理代码功能) 我没有精力搞 selector 了(C:\Users\15919\Desktop\Pocket_Plus\src\selector)。我的意图是: 删去当前最新代码中的 selector 和推理程序中构造的不必要产物(如CLG)，但是 probability map、 F1-centered、F_alpha-centered(这里的alpha不要固定为原本的7个)、Li-centered 等产物仍然需要且必须保证契约完全不变——————即推理代码仍然能够以同样的表现为给现有的 Stage2-Matcher 代码库提供数据。同时，精简主线代码、加快运行速度，保证CPU与GPU高效并行工作、GPU在推理时总能有效利用而不等待。

- (基础建设————数据) 
(1).codex://threads/019fe775-2e19-76f0-8217-dc8a7358e308 分析了历史上 I/O 瓶颈的原因并给出了可能的解决方案。我考虑做这一项：
"中到大把 exp、ligand_dist、union_mask 转为可切块读取的 HDF5/Zarr-sharded 格式，或独立 .npy 加 mmap；避免从 NPZ 中物化整图单请求读取可从平均 276 MB 降到十几 MB；应避免在 Lustre 上制造海量小 Zarr 文件"。
我考虑直接把npz里面的大数组复制一份到相同位置的npy，然后删除npz原本的大数组。我想让这些操作原子进行，并做到机械化、不复杂、低破坏性:
for npz_name, key, npy_name in (
    ("exp.npz", "grid", "exp.npy", ·对sim.npz同样如前操作·),
    ("ligand_dist.npz", "distance", "ligand_dist.npy"),
    ("ligand_area.npz", "union_mask", "union_mask.npy"),
):
    path = density_dir / npz_name
    with np.load(path, allow_pickle=False) as data:
        np.save(density_dir / npy_name, data[key], allow_pickle=False)
        remaining = {name: data[name] for name in data.files if name != key}
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **remaining)
    os.replace(temporary, path)


- (基础设施————划分) 构建新的 split 清单、box_pool 。
(1).split规则为：先按照 EMDB 的发布时间顺序分成 held-out 和非 held-out (时间界线是 2026.1.1)两部分。非held out集合使用筛选规则：好于5埃(或4埃)，cc好于0.6(或0.65,按照实际数据决定)进行筛选，之后按照: 200val、100 cal、剩余都是训练集的规则即可。做到这里可以清理两部分历史代码: fraction 小样本构造逻辑、原本的 split 逻辑。
(2).box_pool 的训练请求为 0:5:5，冻结验证请求为 0:1:1；偏移规则仍然复用经验半径偏转 + 0~3 埃偏转。
额外主意，不要删除目前旧的 _v2 的 split&box_pool 版本。

- (基础设施————模型)
对于 embed head, 它相当于分别为体素分支和点云分支提供输入。我原本的设计意图是想让点云分支和体素分支的输入享有完全独立的参数(所以trunk不存在是符合预期的), 但是  input_proj 却违背了这一原则——————我们需要修改模型代码，需要让点云分支和体素分支享有参数独立的  input_proj 。
(2).把当前原子的49维特征变成50维特征(加入主链原子的标识), 主链原子的标识直接来自 receptor.npz ，运行时拼接即可。 




———————————————————————————————— 尽快按照新的 split 清单、box_pool 和原样配方依次训练 Find_1、unet_c1、Find_0 ———————————————————————————————— 

- (基础设施) 
优化推理程序的性能: 严重怀疑"乘以hardmask"反而有害, 建议直接去掉; 建议尝试用偏向recall的F_alpha阈值 + 高斯打分 + 把过滤的最低体素数阈值综合地调一调, 如果参数很多不好调, 可以把高斯打分的sigma固定为上次的最优经验值; 
改善推理程序的可读性：当前代码多嵌套、不直观，值得重写。

- (检验已有产物) 考虑这个原本训练到一半的任务, wandb曲线对应: pencounkdual-111/AdaLigand_Stage1/es683hq5 。使用最后一个ckp，用新的推理管线和 _v2和_v3 数据评估一下它的质量。

- 检验上面训练的新模型。




# Stage2-Matcher 
- Stage2 一上来就选择了多模态的策略以预测配体和候选的匹配关系: (配体; A, P, PP-额外采样原子, Map-用来调制&摘要)，目前只用了10%不到的"可能可用数据"就达到了约70%的精度(e2e_f1)，我认为作为本科研项目的组件，我判断：这已经可以算作成功了。
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
