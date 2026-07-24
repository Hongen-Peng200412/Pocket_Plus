# AdaLigand 数据管线代码、Git 与 Skill 治理讨论稿

> 状态：首轮只读审计后的讨论稿，尚未得到用户确认，不是实施授权，也不是最终规范。
>
> 本轮边界：未修改 AdaLigand 的代码、测试、计划、执行日志、mapping、契约 README、项目记忆或 Git 历史。本文只汇总已核验事实、初步判断、建议顺序和待讨论分歧。

## 1. 先给结论

你的三个层次的目标都合理，但不能按“先整理文件，再补注释，再整理 Git，最后顺手改 skill”的顺序机械执行。AdaLigand 当前的问题不是单纯的文件太多，而是四种边界同时失真：

1. 稳定科学逻辑、外部工具适配、数据契约、执行控制和一次性恢复代码混在同一活跃目录。
2. clean spec、当前契约、学习指南和历史执行证据之间已有明确漂移。
3. 学习注释曾直接进入 `main`，但项目尚无正式 Learn 分支；现有 Git 规则又对“以后从 main 还是 Learn 开发”给出互相不兼容的答案。
4. 三个 skill/准 skill 分别从函数结构、注释表达、源码学习角度规定规则，却没有一个规则真正负责“包、目录、模块、生命周期和依赖方向”。

因此，正确的总顺序应是：

```text
事实与契约收敛
  -> 冻结行为基线和历史证据
  -> 给每个文件判定生命周期与所有权
  -> 先分离主线/运维/历史脚手架
  -> 再按模块职责重构科学代码
  -> 重写当前契约与阅读入口
  -> 构造生产线/学习线
  -> 把经实践验证的规则提炼为多个边界清晰的 skill
  -> 全套测试、真实 smoke、等价性检查与治理收口
```

这里最重要的原则是：**目录重组是契约收口的结果，不是开始时凭文件名做的分类动作；注释是已理解结构的表达，不是用来掩盖结构混乱的补丁。**

## 2. 已核验的项目事实

### 2.1 管线已经完成，维护任务尚未完成

- 项目记忆、最新 handoff、mapping 和 A-G ExecPlan 一致表明：正式 run `adaligand_ag_20260711T154658` 已于 2026-07-20 完成 A-G analyze-only 目标。
- G 只运行 analyze，未生成 `keep_list.jsonl`；这是明确边界，不是管线失败。
- 最新 handoff 明确把“代码可读性重组”和“run-specific 脚手架物理归档”留作下一维护任务。
- ExecPlan 已初步把文件分为长期生产主线、默认关闭的可复用运维、本 run 专用历史脚手架，但尚未物理落实。

所以，本次工作应被定义为：**在不改变已经验收的科学产物契约的前提下，对完成后的实现做结构、文档、Git 和 agent 工作方式的系统收口。**

### 2.2 当前复杂度不是主观错觉

`Data_Preprocessing/Ori_Data` 当前大致包含：

| 区域 | 文件数 | 代码行数 | 观察 |
|---|---:|---:|---|
| `code/*.py` | 42 | 17,265 | 科学、适配、契约、审计、迁移、恢复混放 |
| `scripts/*.py` | 27 | 4,995 | 正式 CLI、审计 CLI、迁移 CLI、一次性入口混放 |
| `tests/*.py` | 38 | 13,040 | 覆盖广，但大量测试绑定当前平面路径和实现哈希 |
| `sbatch/*` | 24 | 2,373 | 正式 DAG、smoke、supplement 和多版 `resume_*` 混放 |

最大的正式科学模块包括 `density.py`、`parse.py`、`quality.py`，都约一千行以上；最大的运行脚手架甚至更大，例如 `f_signal11_exclusion_transition.py` 约 1,892 行。问题不仅是某个科学文件过长，更是恢复历史的代码体量已经足以遮蔽正式主线。

### 2.3 文档存在可证实的当前性问题

- `code/readme.md` 开头仍写“服务器全量产物尚需本轮正式运行生成”，末尾却记录 A-G 已完成。
- 同一 README 既包含稳定字段/shape 契约，也包含具体 Job ID、SHA、恢复步骤、阶段停点和事故历史。
- `learn.md` 记录 `filtering.py` 仍是 schema v1、只支持 Q-score 与 resolution；当前代码实际已经是 map-level schema v2，并包含 CC、配体 Q、口袋 Q 和比例规则。
- `learn.md` 自己也承认旧 `pair_pass` 口径和当前代码不一致。这说明它已从“学习入口”退化成“当前说明 + 历史差异 + 旧阅读尝试”的混合物。

因此，重写契约不是润色，而是一次正式的 plan/code/contract/learning drift 审计。按 AdaLigand 的治理规则，在用户确认前不能静默把代码现实回填为 clean spec。

### 2.4 当前 Git 状态也需要纳入设计

- 仓库当前只有 `main`，没有 Learn 分支。
- `main` 比 `origin/main` ahead 1，并有与本任务无关的已修改/未跟踪用户文件；后续分支和提交必须在独立 worktree 或精确路径操作中完成。
- 2026-07-13 的提交 `50d068e` 已把四行“学习导航”注释加入大量 Python 文件，并直接进入 `main`。
- 这意味着当前仓库并不满足“main 无学习注释、Learn 才有学习注释”的理想模型。未来不能假装从零设计 Git，必须明确如何处置这段既有历史。

### 2.5 目录移动具有真实契约风险

当前测试、waiver、process audit、source manifest 和若干恢复模块会绑定：

- `code/...`、`scripts/...` 的具体路径；
- 实现文件 SHA-256；
- `Path(__file__)` 推导出的相对位置；
- 固定脚本名和命令行字符串；
- 已完成 run 的证据目录与特定实现身份。

所以不能只用 `git mv` 后修 import。需要先决定：历史 run 的再验证是通过旧 Git commit 重放，还是要求当前工作树继续兼容旧路径。我的建议是前者：把完成 run 的实现身份永久绑定到冻结 commit/tag 和服务器证据，不让历史兼容性继续污染活跃生产目录。

## 3. 对“一个文件只能有一两个中心函数”的判断

这个直觉抓住了“模块需要中心”的本质，但不适合作为硬性计数规则。

更稳妥的规则是：

1. 一个模块只能有一个可以用一句话说清的主要职责。
2. 模块应有很小的公开 API，通常 1-3 个中心入口；其余函数默认私有。
3. 是否拆函数由独立语义、复用、易错副作用、可测试边界和主流程可读性决定，不按行数或形式强拆。
4. 是否拆文件由“变化原因是否相同”和依赖方向决定，而不只看函数数量。
5. 超过约 400-600 行应触发结构审查，超过约 800 行必须解释为何仍是一个内聚模块；这是 review trigger，不是自动失败线。

例如，Stage E 的 E1/E2/E3 分别产生三类 artifact，它们共享网格契约但变化原因不同。把它们强行压成一个中心函数并不会更清楚；更合理的是形成一个 `density` 子包，由三个 artifact builder 和一个薄的 stage service 组成。相反，一个 80 行文件若同时负责网络请求、科学计算和 Slurm 状态，也仍然应该拆。

建议把新的代码组织规则概括为：**一个包对应一个领域或生命周期，一个模块对应一个职责，一个公开函数对应一个可命名的操作；辅助函数是否存在服从可读性，不服从数量美学。**

## 4. 建议的代码分层

这里先给职责结构，不在首轮锁死最终目录名：

```text
稳定 CLI
  -> Stage 应用层（A-G 编排一个样本或一个 stage）
      -> 领域/科学层（结构、标签、密度、质量、过滤）
      -> 外部适配层（RCSB、EMDB、MRC、Chimera、MapQ）
      -> Artifact 契约层（schema、校验、序列化、状态）
  -> 运行基础层（并行、报告、锁、release gate）

显式运维入口
  -> 可复用审计/修复/恢复工具
  -> 可以调用稳定生产层
  -> 稳定生产层绝不能反向依赖运维或历史代码

历史 run 脚手架
  -> 从活跃 import/CLI/sbatch 路径退出
  -> 由 Git checkpoint、ExecPlan、manifest 和证据目录保存历史
```

一个可能的物理落点是：

```text
Data_Preprocessing/Ori_Data/
  src/adaligand_preprocessing/
    stages/          # A-G 应用服务，不放 argparse
    structure/       # occurrence、LigandObject、receptor、atom labels
    density/         # E1/E2/E3 与 MRC 几何契约
    quality/         # CC、Q-score、pocket Q、G filter
    adapters/        # RCSB/EMDB/Chimera/MapQ/model CIF/MRC IO
    artifacts/       # schema、validator、serialization、status/report
    runtime/         # 并行、锁、release gate 共用机制
  cli/               # 稳定 A-G 命令入口
  operations/        # 默认关闭的可复用 repair/audit/recovery
  slurm/
    production/
    operations/
  archive/           # 若确需保留可执行历史脚本；默认更推荐只保留索引到 Git/证据
  tests/
```

这只是方向。最终落点要在依赖图和历史证据策略确定后再锁。尤其不建议继续创建一个无边界的 `utils.py`；共享函数应按 `artifacts/io.py`、`runtime/parallel.py`、`density/geometry.py` 这类语义命名。

## 5. 脚手架的三分法

现有 ExecPlan 的三分法是正确起点，但还需要给每个文件补齐判据：

### 5.1 长期生产代码

判据：未来 clean run 默认需要；不绑定单一 run/Job/PDB allowlist；其行为属于当前科学或 artifact 契约；有稳定测试与 CLI。

### 5.2 可复用但默认关闭的运维工具

判据：能处理一类未来可能再次出现的问题；输入必须显式给出 run、manifest 和身份；不会偷偷进入默认科学路径；有自己的操作契约、测试、退出条件和负责人可读入口。

### 5.3 run-specific 历史脚手架

判据：硬编码本次 run、Job、ID、时间窗或一次性迁移目标；退出条件已经满足；其价值主要是解释历史而不是服务未来运行。

这类文件不应因为“以后也许能参考”继续留在活跃路径。Git、ExecPlan、服务器 evidence 和一个简短 archive index 已经能保存参考价值。若保留可执行副本，必须集中隔离并明确 `not a production entry point`，且生产代码和正式 sbatch 不得 import/call 它。

## 6. 注释体系应怎样设计

你提出的两层注释是正确的，但建议扩成三个层次，并区分生产线与学习线的密度。

### 6.1 模块 Docstring：回答“这个文件为什么存在”

每个正式模块开头用一个真实模块 Docstring，至少说明：

- 单一职责与非目标；
- 公开入口/中心函数；
- 主要输入来源；
- 主要返回值与落盘 artifact 路径；
- 外部工具、副作用和失败边界；
- 当前属于生产、运维还是历史生命周期。

现有四行 `# 学习导航` 与随后 Docstring 有重复，而且没有稳定说明公开 API、写盘原子性和非目标，不能作为最终方案。

### 6.2 中心函数 Docstring：回答“它消费和产出什么契约”

对真正产生 artifact 的函数，必须展开：

- 参数类型、shape/长度、单位、坐标系和来源；
- 返回结构的每个字段；
- 写盘文件的每个 key、dtype、shape、轴序、空值语义；
- 原子写/幂等/overwrite 行为；
- known/unknown failure 如何上浮。

稳定 artifact 契约仍应由 code-near contract 文档统一承载；Docstring 是代码附近的局部完整契约，不能复制几百行运行历史。

### 6.3 学习型变量注释：回答“数据在这一行变成了什么”

建议为 Learn 分支采用高密度 profile：

- 每个非标量变量在当前作用域首次获得关键语义时，至少注释一次类型、当前 shape/长度、意义和近源。
- shape、坐标系、单位、dtype、mask/索引语义发生变化时再次注释。
- 数组当前 shape 用圆括号；单行变换用方括号箭头。
- 坐标必须写明 world/voxel/voxel-index/linear-index、XYZ/ZYX、连续/离散、corner/center、local/full-map。
- 允许一行代码对应一行甚至多行注释，但不机械翻译 Python 语法。

“所有非标量变量至少注释一次”很适合你的私人 Learn 分支，但不宜强制进入生产 `main`。生产线保留契约、领域语义和非显然变换；学习线提供更密集的数据流脚手架。

## 7. 契约 README 应怎样拆

当前 `code/readme.md` 不应继续同时承担四种角色。建议拆为：

1. `README.md`：当前入口、主路径、目录角色、最小运行方式和文档导航。
2. artifact contract：只写当前字段、dtype、shape、单位、主键、空值、原子性、验证规则。
3. operations contract：只写默认关闭的 audit/repair/recovery 如何显式调用、边界和安全门。
4. run history：继续留在 ExecPlan、handoff、Git 和服务器 evidence，不复制回当前契约。
5. learning guide：按最终目录和数据流写阅读顺序，不夹带已经失效的实现口径。

README 重写前必须先做正式 drift 分类。已发现至少包括：

- 当前代码/计划已经是 G map-level schema v2，但 `learn.md` 仍描述旧 v1：这是有害的文档漂移。
- README 开头的“尚未全量运行”已经过时：这是有害的状态漂移。
- run-specific waiver、cutoff、supplement 的历史事实本身有价值，但放在 artifact contract 内是中性内容放错生命周期。
- `keep_list` 尚未生成是明确未完成的独立后续范围，不应被重写成 A-G analyze 未完成。

是否把实现现实回填到 clean spec，仍须按项目治理规则由你确认。

## 8. Git 双线方案：合理之处与核心矛盾

“实现线保留真实开发过程，学习线按逻辑依赖重建最终正确代码”是合理且有价值的。学习线笔直、实现工作从其上叉出，也非常符合人的阅读直觉。

但现有两套规则有三个直接冲突：

1. `source-reading-annotation-workflow` 把 `main` 定义为生产基线，并要求 main 更新后 cascading rebase Learn。
2. `注释规则.md` 要求下一轮实现从 `Learn/model-cumulative` 出发。
3. 前者允许独立 module branch 后 cherry-pick 到 cumulative；后者要求 milestone branch 只是同一线上的指针，不产生重复 commit。

最关键的问题是：**Learn 到底是 main 的派生阅读镜像，还是未来开发的规范基线？** 两者不能只靠口号同时成立。

我的初步建议是保留两条长期线，并引入“投影”而不是普通 merge 的概念：

```text
main
  只含生产代码、正式文档和必要生产注释，是可发布基线。

Learn/model-cumulative
  是 main 的注释增强镜像；非注释 Python AST、配置值和 artifact 接口必须与 main 等价。

Work/<topic>
  可以按你的偏好从 Learn 分出，以继承既有学习注释。
  收口时把功能性差异投影为 main 的语义提交；再让 Learn 基于新 main 重建/更新学习注释。
```

这套方案能保留“学习树干、实现分枝”的使用体验，但代价是每轮必须做双目标收口，不能直接 merge。需要一个确定性检查器验证：

- Python 去除普通注释和 Docstring 后 AST 等价；
- YAML 去除注释后配置值等价；
- 白名单外文件 blob 等价；
- 删除/新增清单一致；
- main 与 Learn 的允许差异只有学习注释、Docstring、YAML 注释和学习文档。

另一个更简单的方案是所有实现分支只从 main 出发，Learn 永远只是阅读镜像。它工程风险更低，但不满足你“维护也从学习分支出发”的明确偏好。这个分歧值得后续重点 grill，而不是由 agent 偷偷替你决定。

## 9. 当前 AdaLigand 的 Git 重建建议

讨论收敛后，建议按下列阶段建历史，而不是把当前 7 月份的运行事故提交全部重写掉：

1. 保留当前 `main` 历史作为真实实现史，不 rebase、不 squash。
2. 在干净 worktree 冻结本次维护的共同基点；先处理现有无关工作区资产，不能卷入提交。
3. 建实现分支完成目录分层、模块重构、契约更新和脚手架退出。
4. 实现分支按真实工作过程保留必要的诊断/修复提交。
5. 从约定共同基点构造本轮学习线，按最终逻辑依赖重建：基础契约 -> Stage C -> D -> E -> F -> G -> 运维边界 -> 文档。
6. 学习线不复现已经失败的中间设计；它直接呈现最终正确代码，并加入学习注释。
7. 用机器检查证明实现终点与学习终点除白名单注释/文档外等价。
8. 再决定 `Learn/model-cumulative`、各里程碑指针和是否推送。

当前已经进入 `main` 的四行学习导航注释不必改写旧历史；可在本次结构重组时决定哪些升级为生产级模块 Docstring，哪些只在新 Learn 线保留。

## 10. Skill 层面的建议

不建议把所有规则继续塞进一个更大的 `source-reading-annotation-workflow`。Skill 设计应按职责拆分，并用脚本承担脆弱的机械检查。

### 10.1 保留并收窄 `python-writing-style-cn`

继续负责：函数是否拆分、参数显式性、默认值、回退、raise/error 边界。

不要让它独自负责整个项目目录。可以增加一条指针：涉及包/模块/生命周期组织时调用新的结构治理 skill。

### 10.2 新建或提炼“Python 项目结构治理”skill

负责：

- package/module/function 三层职责；
- 依赖方向；
- 生产、适配、契约、运维、脚手架生命周期；
- CLI 与核心逻辑分离；
- 文件拆分 review trigger；
- 禁止无边界 `utils.py`；
- 重构时的兼容 shim、测试和 artifact 等价门。

这是当前 skill 体系真正缺失的一层。

### 10.3 把 `code-comment-style-cn` 变成基础表达规范 + profile

它负责中文表达、Docstring 字段展开、shape/类型/语义格式。详细规则可放在 references 中，按场景选择：

- production profile；
- learning profile；
- data-pipeline/artifact profile；
- low-level math profile；
- YAML profile 继续由 `yaml-config-style-cn` 管理。

这样 `source-reading` 不再以“发生冲突时全部覆盖”为解决办法，而是选择 learning profile 并追加阅读链规则。

### 10.4 收窄 `source-reading-annotation-workflow`

它应负责：阅读边界、数据流顺序、近源追踪、低层数学自包含、学习完成标准。Git 双线的完整机械操作应移出，避免源码阅读 skill 同时拥有 rebase、main、cumulative、module branch 的全部政策。

### 10.5 独立“双线 Git 工作流”skill

它负责：

- main/Learn/Work 的角色；
- 共同基点；
- 实现线与学习线的投影；
- 用户 staged/unstaged 资产保护；
- 提交粒度与 milestone 指针；
- main 更新如何同步 Learn；
- 哪些情况允许重写已发布历史；
- 等价性检查与 worktree 操作。

现有 `注释规则.md` 的标题已经不能覆盖它实际承担的职责，建议在讨论后拆成 Git 规则和注释 profile，而不是整体“转正”为一个大 skill。

### 10.6 Skill 应附带确定性工具

建议随 skill 提供小型脚本，而不是只写自然语言：

- Python 注释/Docstring 剥离后的 AST 比较；
- YAML 注释剥离后的语义比较；
- main/Learn 文件白名单差异检查；
- 生产包依赖方向检查；
- 活跃生产目录中的 hard-coded run/job/PDB ID 扫描；
- 历史脚手架分类清单校验。

这部分建议直接来自本次事实：单靠文档，现有 `learn.md` 和 README 已经发生漂移；脆弱不变量必须尽量机器验证。

## 11. 建议的完整执行层次

讨论收敛后，一次性完成本任务时建议设以下验收门：

### 阶段 A：事实与规格收敛

- 逐 Stage 对照 plan、ExecPlan、当前代码、README、真实完成证据。
- 列出 beneficial/neutral/harmful/unfinished drift。
- 由用户决定哪些回填 clean spec。
- 冻结当前实现 commit、CLI、artifact schema、外部工具 lineage 和历史证据入口。

### 阶段 B：文件生命周期清单

- 每个 Python/script/sbatch/test 判定所有者、生命周期、默认入口、消费者、测试和退出条件。
- 未完成分类前不移动文件。

### 阶段 C：先隔离脚手架，再重构科学代码

- 先让 run-specific 文件退出活跃主路径。
- 再建立正式 package 和依赖方向。
- 保留必要 CLI 兼容入口；兼容层必须薄且有退役条件。
- 按 Stage/领域逐个重构大模块，每步保持测试可运行。

### 阶段 D：契约与注释

- 重写 code-near artifact contract。
- 把运行历史移回 ExecPlan/Git/evidence。
- 为生产代码补模块和中心函数契约。
- 在 Learn 线补高密度变量/shape/坐标注释和阅读文档。

### 阶段 E：Git 双线重建

- 保留真实实现史。
- 构造逻辑学习史。
- 自动验证代码/配置等价。
- 建 cumulative 与 milestone 分支指针。

### 阶段 F：Skill 收口

- 用 AdaLigand 的真实操作反向验证规则。
- 拆分结构、注释、源码阅读和 Git 职责。
- 运行 skill validator 和至少一轮无泄漏的前向测试。

### 阶段 G：最终验收

- 运行契约专项、完整本地测试和必要真实 Chimera/MapQ/MRC smoke。
- 检查 CLI、artifact schema、数值 lineage、源哈希策略和历史证据可追溯性。
- 更新 plan/log/mapping/contract/memory，并完成脚手架最终分类。

## 12. 首轮最需要讨论的分歧

以下问题会改变实施方案，建议后续 grill 优先围绕它们展开：

1. `Learn/model-cumulative` 是派生阅读镜像，还是未来开发的规范基线？若是后者，是否接受每轮双目标投影到 `main` 的额外成本？
2. run-specific 脚手架是从当前树删除、只由 Git 保存，还是集中移入 `archive/` 保留可执行副本？我倾向前者，只保留精确索引；确有未来审计需求的少数工具再保留。
3. 正式 A-G CLI 路径是否必须长期兼容当前 `scripts/*.py`？我倾向先保留薄 shim，等服务器/文档消费者迁移完成后再退役。
4. 学习注释的默认密度是否采用“每个非标量变量每作用域至少一次”，以及测试代码是否也进入学习范围？我倾向生产测试不逐变量注释，只在专门学习测试里解释关键 fixture 和断言语义。
5. 本轮是否只做行为保持的结构重组，还是同时修复已经确认但未影响 A-G 完成的实现/文档问题？这决定 drift 审计后的 scope 边界。
6. 最终代码 package 名、目录名和旧路径兼容期多长，应在依赖清单完成后锁定，不建议现在凭审美决定。

## 13. 当前建议

我建议下一轮先不要直接开始移动文件，也不要立刻发动全量 grill。先围绕第 12 节六个分歧确认大方向，尤其先解决 main/Learn 的权威关系和脚手架的保存策略。两点一旦锁定，其余目录、提交和 skill 设计会明显收敛。

首轮事实已经足够说明：你感到“即使整理完仍缺了什么”是准确的。缺失的不是更多注释，而是一套贯穿 **代码生命周期、artifact 契约、Git 双线和 agent 执行方式** 的共同模型。这个模型应先在 AdaLigand 上被实际验证，再提炼成 skill；不应先写一套看似完整的抽象规则，再让项目被规则反向束缚。
