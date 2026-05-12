# Pocket Plus CLAUDE Workspace Plan

本文档用于约定本项目中与 agent skill、计划文档、项目记忆相关的目录结构、语言规则和后续安装/改造方向。

当前阶段的目标是把共享 skill 交给 CC Switch 管理，把项目产物统一落到本仓库的 `CLAUDE/` 目录中。

## 1. 总体原则

### 1.1 Skill 本体语言

* skill 的 `SKILL.md`、命令说明、说明性规则，默认使用英文撰写。
* 如采用社区 skill 或官方 skill，尽量保留发布者原文和原有表达，只做必要修改。
* 必要修改仅限于：
  * 本项目的本地路径约定
  * 本项目是否要求产出中文内容
  * 与当前工作流直接冲突的少量规则

### 1.2 Skill 产物语言

* 保存到本项目 `CLAUDE/` 目录下的产物，默认使用中文。
* 这里的“产物”包括但不限于：
  * implement plan
  * remember 保存的记忆条目
  * handoff 文档
  * 其他为当前项目持续沉淀的 markdown 文档

### 1.3 路径原则

* 所有项目级计划、记忆、handoff、项目状态文件，统一放在当前 VSCode 打开的项目根目录下的 `CLAUDE/` 文件夹中。
* 不使用上游默认的 `.claude/` 路径。
* 可以在既定目录下创建必要的子文件夹。

### 1.4 Skill 管理原则

* 共享 skill 统一放在 `C:\Users\15919\.codex\skills`，由 CC Switch 负责启用、停用和管理。
* 本项目目录不保存共享 skill 的副本。
* 项目专属规则保留在 `.agents/rules/` 或已有的 `pocket-plus-dev-workflow` skill 中。
* 本项目的 `CLAUDE/` 目录只保存计划、记忆、handoff 和项目状态产物。

## 2. 目录约定

当前项目根路径：

`C:\Users\15919\OneDrive\My_Project\Pocket_Plus`

本项目中约定使用以下目录：

```text
CLAUDE/
├── readme.md
├── plans/
│   ├── implement/
│   └── archive/
└── memory/
    ├── handoffs/
    ├── learnings/
    ├── projects/
    └── index.json
```

说明：

* `CLAUDE/plans/implement/`
  * 存放实现计划。
  * 主要对应 `implement-plan-writer` 或 `grill-then-plan` 最终生成的正式计划。
* `CLAUDE/plans/archive/`
  * 预留给过期、废弃或历史计划。
  * 当前阶段可先不主动使用。
* `CLAUDE/memory/learnings/`
  * 存放通过 `/remember` 或类似机制沉淀的长期记忆条目。
  * 条目类型可参考 Primeline 的 `solution / pattern / decision / gotcha`。
* `CLAUDE/memory/handoffs/`
  * 存放阶段性交接文档。
  * 这里替代 Primeline 上游常见的 `.claude/handoffs/` 路径。
* `CLAUDE/memory/projects/`
  * 存放项目状态 JSON 文件。
* `CLAUDE/memory/index.json`
  * 存放当前 active project 指针和全局入口信息。

## 3. Skill 规划

### 3.1 全局共享 skill

#### `grill-me`

* 保持独立存在。
* 用于在制定计划前先澄清目标、约束、边界条件、非目标和分支决策。
* 原则上尽量保留原始 skill 的英文写法和原有职责边界。

#### `implement-plan-writer`

* 保持独立存在。
* 用于把已经较为明确的方案写成正式、具体、可执行的实现计划。
* 产物保存到 `CLAUDE/plans/implement/`，并使用中文输出计划内容。

#### `grill-then-plan`

* 新建一个很薄的 orchestrator skill。
* 它不替代 `grill-me` 和 `implement-plan-writer`，只做编排。
* 预期流程：
  1. 先按 `grill-me` 风格澄清问题。
  2. 当关键决策基本稳定后，切换到计划生成阶段。
  3. 最终输出正式 implement plan。
* 该 skill 的正文仍然用英文写，尽量短、尽量聚焦。
* 该 skill 的最终产物仍然用中文。

#### `claude-remember`

* 全局共享 skill，存放在 `C:\Users\15919\.codex\skills\claude-remember`。
* 用于把可复用信息写入当前工作区的 `CLAUDE/memory/learnings/`。
* 默认产出中文；如果用户明确要求“用英文”，则产出英文。

#### `claude-handoff`

* 全局共享 skill，存放在 `C:\Users\15919\.codex\skills\claude-handoff`。
* 用于把阶段性交接写入当前工作区的 `CLAUDE/memory/handoffs/`。
* 默认产出中文；如果用户明确要求“用英文”，则产出英文。

#### `claude-project-status`

* 全局共享 skill，存放在 `C:\Users\15919\.codex\skills\claude-project-status`。
* 用于读取、初始化或轻量更新当前工作区的 `CLAUDE/memory/index.json` 与 `CLAUDE/memory/projects/*.json`。

#### `skill-router`

* 全局共享 skill，存放在 `C:\Users\15919\.codex\skills\skill-router`。
* 用于简要说明已安装 skill 的用途和触发方式。
* 它只负责路由和说明，不替代其他 skill 执行具体任务。

### 3.2 Primeline 相关规划

计划借鉴 `primeline-ai/claude-code-starter-system`，但做最小化项目适配。

适配原则：

* 保留其核心机制：
  * `remember`
  * `handoff`
  * 项目状态 JSON 文件
* 尽量保留其英文原文和原始结构。
* 仅做必要修改：
  * 把 `.claude/...` 改到 `CLAUDE/...`
  * 把 handoff 存储路径改到 `CLAUDE/memory/handoffs/`
  * 对所有 skill 的落盘产物，默认使用中文；如果用户明确简要要求“用英文”，则改为使用英文
  * 去掉或放宽会强制“所有输出都必须是英文”的规则

当前拟采用的映射：

* Upstream `.claude/memory/index.json`
  -> `CLAUDE/memory/index.json`
* Upstream `.claude/memory/projects/{active}.json`
  -> `CLAUDE/memory/projects/{active}.json`
* Upstream `.claude/memory/learnings/*.md`
  -> `CLAUDE/memory/learnings/*.md`
* Upstream `.claude/handoffs/*.md`
  -> `CLAUDE/memory/handoffs/*.md`

### 3.3 Scientific Agent Skills 规划

当前只考虑“生物分子结构预测与建模”方向，不做整包安装。

当前纳入 CC Switch 管理或按需启用的集合：

* `Database Lookup`
* `BioPython`
* `gget`
* `ESM`
* `Molecular Dynamics`
* `DiffDock`
* `RDKit`

当前理解：

* 上述 skill 更适合做“专项能力补充”，而不是项目总控 skill。
* 这些 scientific skill 可以由用户显式指定触发，例如用户直接点名某个 scientific skill 或明确提示使用 scientific skill。
* 它们保持各自独立，不与项目总控类 skill 混写。
* 这些 skill 本体继续保持英文说明；若它们输出项目记忆或计划，则相关落盘文档应遵守本项目中文产出规则。

## 4. 产物命名与内容建议

### 4.1 Implement Plan

建议存放路径：

* `CLAUDE/plans/implement/`

建议命名格式：

* `YYYY-MM-DD-topic.md`

内容语言：

* 中文

用途：

* 记录准备实施的功能、修改点、验证方式、非目标等。

### 4.2 Memory Learnings

建议存放路径：

* `CLAUDE/memory/learnings/`

建议按类型分文件名或子目录组织，例如：

* `decision-YYYY-MM-DD-topic.md`
* `solution-YYYY-MM-DD-topic.md`
* `pattern-YYYY-MM-DD-topic.md`
* `gotcha-YYYY-MM-DD-topic.md`

内容语言：

* 中文

用途：

* 记录已经查明、以后仍可能复用的信息。

### 4.3 Handoffs

建议存放路径：

* `CLAUDE/memory/handoffs/`

建议命名格式：

* `YYYY-MM-DD-session-handoff.md`

内容语言：

* 中文

用途：

* 在一次工作会话结束时，总结当前已完成事项、未完成事项、阻塞点和下一步建议。

## 5. 面向用户：如何查看这些文件

### 5.1 查看计划文件

如果要看当前或历史 implement plan：

1. 打开 `CLAUDE/plans/implement/`
2. 按文件名日期或 topic 查找对应条目
3. 直接阅读 markdown 文件内容

如果某个计划已废弃或不再使用，可后续移到 `CLAUDE/plans/archive/`

### 5.2 查看 memory 里的条目

如果要看通过 `remember` 沉淀下来的长期记忆：

1. 打开 `CLAUDE/memory/learnings/`
2. 根据文件名前缀判断条目类型：
   * `decision-...` 表示决策
   * `solution-...` 表示解决方案
   * `pattern-...` 表示可复用模式
   * `gotcha-...` 表示坑点或注意事项
3. 进入对应 markdown 文件查看细节

### 5.3 查看 handoff

如果要看某次工作交接内容：

1. 打开 `CLAUDE/memory/handoffs/`
2. 根据日期找到对应会话的 handoff 文件
3. 阅读其中的“已完成 / 未完成 / 下一步”内容

### 5.4 查看当前项目状态

如果要看当前项目的机器可读状态：

1. 打开 `CLAUDE/memory/index.json`
2. 查看当前 active project 指向
3. 再进入 `CLAUDE/memory/projects/` 中对应的 JSON 文件

这部分主要是给 agent 工作流使用的，但用户也可以直接打开查看当前记录的项目状态字段。

## 6. 当前已完成的落地项

* 删除了本项目 `.agents/skills/` 中误建的未完成共享 skill 骨架。
* 在 `C:\Users\15919\.codex\skills` 中安装了 `grill-me`。
* 在 `C:\Users\15919\.codex\skills` 中新增了 `grill-then-plan`、`claude-remember`、`claude-handoff`、`claude-project-status` 和 `skill-router`。
* 将生物分子结构预测与建模方向的 scientific skills 纳入 CC Switch 管理和按需启用策略。
* 初始化了本项目的 `CLAUDE/plans/` 与 `CLAUDE/memory/` 目录骨架。

## 7. 当前仍不做的事

* 不复制 Primeline 全部内容。
* 不把共享 skill 放回本项目 `.agents/skills/`。
* 不安装 `mattpocock/skills` 或 `superpowers` 的完整集合。

## 8. 本次新增 skill 的来源

本节只记录本次新增或本次引入的 skill。此前已经存在的 `implement-plan-writer`、`execplan`、`python-writing-style-cn`、`code-comment-style-cn`、`yaml-config-style-cn`、`pocket-plus-dev-workflow` 不在这里重复说明。

### 8.1 `grill-me`

来源：

* 仓库：`mattpocock/skills`
* 原始路径：`skills/productivity/grill-me`
* 网页：<https://github.com/mattpocock/skills/blob/main/skills/productivity/grill-me/SKILL.md>

用途：

* 在开始写计划或实现前，追问需求、边界、非目标和设计分支。
* 适合用户说“grill me”“先问清楚”“先帮我把方案拷打一遍”这类请求。

### 8.2 `grill-then-plan`

来源：

* 本地自定义 skill。
* 设计参考：
  * `grill-me`: <https://github.com/mattpocock/skills/blob/main/skills/productivity/grill-me/SKILL.md>
  * 本地已有 `implement-plan-writer`

用途：

* 先按 `grill-me` 风格澄清，再按 `implement-plan-writer` 风格写正式 implement plan。
* 计划产物默认写入 `CLAUDE/plans/implement/`。

### 8.3 `claude-remember`

来源：

* 本地自定义 skill。
* 设计参考：`primeline-ai/claude-code-starter-system` 的 `/remember` 命令。
* 网页：<https://github.com/primeline-ai/claude-code-starter-system/blob/main/commands/remember.md>

用途：

* 把以后还会复用的信息保存到 `CLAUDE/memory/learnings/`。
* 默认使用中文写入记忆条目；如果用户明确说“用英文”，则使用英文。

### 8.4 `claude-handoff`

来源：

* 本地自定义 skill。
* 设计参考：`primeline-ai/claude-code-starter-system` 的 `/handoff` 命令。
* 网页：<https://github.com/primeline-ai/claude-code-starter-system/blob/main/commands/handoff.md>

用途：

* 在一次工作会话结束、暂停或准备下次继续时，保存交接文档。
* 本项目约定 handoff 存放到 `CLAUDE/memory/handoffs/`，而不是上游默认的 `.claude/handoffs/`。

### 8.5 `claude-project-status`

来源：

* 本地自定义 skill。
* 设计参考：`primeline-ai/claude-code-starter-system` 的项目状态记忆机制。
* 相关网页：
  * <https://github.com/primeline-ai/claude-code-starter-system>
  * <https://github.com/primeline-ai/claude-code-starter-system/blob/main/skills/system-boot/SKILL.md>

用途：

* 读取、初始化或轻量更新 `CLAUDE/memory/index.json` 和 `CLAUDE/memory/projects/*.json`。
* 这部分是机器可读状态，供后续 remember/handoff 和会话续接使用。

### 8.6 `skill-router`

来源：

* 本地自定义 skill。

用途：

* 简要说明当前已安装 skill 的用途和触发方式。
* 当用户问“我该用哪个 skill”“这些 skill 有什么区别”“注意 scientific skill”时，用它帮助选择。
* 它只负责路由说明，不替代其他 skill 执行具体任务。

### 8.7 生物分子结构预测与小分子相关 scientific skills

来源：

* 仓库：`K-Dense-AI/scientific-agent-skills`
* 网页：<https://github.com/K-Dense-AI/scientific-agent-skills>
* 技能索引：<https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/docs/scientific-skills.md>

本次选择性引入或规划引入的方向：

* `database-lookup`
* `biopython`
* `gget`
* `esm`
* `molecular-dynamics`
* `diffdock`
* `rdkit`

上游说明中，`scientific-agent-skills` 是一个大型集合，包含 135 个 scientific/research skills，并覆盖 100+ 科学数据库、70+ 优化 Python package skills 等内容。上游也明确建议不要一次安装所有 skill，应只安装实际需要的 skill，并在安装前阅读对应 `SKILL.md`。

本项目当前只关注生物分子结构预测、蛋白语言模型、分子动力学、蛋白-配体对接和小分子处理相关方向，不把整个 `scientific-agent-skills` 仓库全部打开。

## 9. Claude 记忆机制说明

本项目采用 `CLAUDE/` 目录保存项目级长期产物。skill 本体由 CC Switch 管理，存放在 `C:\Users\15919\.codex\skills`；项目记忆、计划和 handoff 则保存在当前项目根目录下。

### 9.1 记忆文件放在哪里

长期记忆：

* 路径：`CLAUDE/memory/learnings/`
* 典型文件名：`decision-YYYY-MM-DD-topic.md`、`solution-YYYY-MM-DD-topic.md`、`pattern-YYYY-MM-DD-topic.md`、`gotcha-YYYY-MM-DD-topic.md`

会话交接：

* 路径：`CLAUDE/memory/handoffs/`
* 典型文件名：`YYYY-MM-DD-topic.md`

项目状态：

* 入口：`CLAUDE/memory/index.json`
* 项目文件：`CLAUDE/memory/projects/*.json`

### 9.2 记忆条目的类型

* `decision`：记录已经做出的项目决策。
* `solution`：记录某个问题的解决方式。
* `pattern`：记录以后可复用的工作模式或实现模式。
* `gotcha`：记录容易踩坑、容易误解或需要特别注意的事项。

### 9.3 常用说法

保存长期记忆：

* “记住这个结论。”
* “把这个作为 decision 记到 memory。”
* “这个坑以后还会遇到，保存一下。”
* “把新版数据路径记下来。”

保存会话交接：

* “写一个 handoff。”
* “这次先到这里，帮我留一个交接。”
* “总结当前进度，方便下次继续。”

查看或初始化项目状态：

* “查看当前 CLAUDE memory 状态。”
* “初始化这个项目的 CLAUDE memory。”
* “当前 active project 是什么？”

指定输出语言：

* 默认：保存到 `CLAUDE/` 的产物使用中文。
* 如果用户明确说“用英文”，则该次产物使用英文。

### 9.4 使用建议

* 小到一次性 debug 且没有后续价值的内容，不必保存为 memory。
* 涉及路径、数据位置、服务器约定、实验约定、长期决策、容易重复踩坑的问题，适合保存。
* handoff 适合在一轮较长工作结束、准备暂停、换模型或下次继续前使用。

## 10. 关于 skill 数量和 CC Switch 启用策略

当前把共享 skill 放在 `C:\Users\15919\.codex\skills` 是为了让 CC Switch 统一管理。这个目录里文件夹多一点不是根本问题，关键在于“当前启用”的 skill 是否过多、description 是否互相重叠。

建议长期启用：

* `implement-plan-writer`
* `grill-me`
* `grill-then-plan`
* `claude-remember`
* `claude-handoff`
* `claude-project-status`
* `skill-router`
* `pocket-plus-dev-workflow`
* `python-writing-style-cn`
* `code-comment-style-cn`
* `yaml-config-style-cn`

建议按需启用：

* `database-lookup`
* `biopython`
* `gget`
* `esm`
* `molecular-dynamics`
* `diffdock`
* `rdkit`

原因：

* scientific skills 是专项工具，适合在做生物分子、小分子、对接、MD、蛋白语言模型任务时启用。
* 上游 `scientific-agent-skills` 本身是大型集合，不建议全量安装、全量启用。
* 如果长期同时启用太多 scientific skill，模型可能会在普通代码任务中被无关科学工作流干扰。
* 如果只是在文件夹中存在、但由 CC Switch 关闭，一般不会干扰当前会话。

因此，推荐策略是：通用工作流长期启用，scientific skills 按任务打开。需要做生物分子结构预测或小分子任务时，再打开相关 skill；任务结束后可以关闭。
