# Handoff: Stage1 冲突整理与学习历史重建

Date: 2026-07-24

## Current State

Pocket_Plus 的 Stage1 最新实现事实保存在 `codex/stage1-consolidated@ebadbeac882faea40779f31e441988b118f7c6fa`。本轮从 `de6a89f38e48553edfd440c76bf61823229d9205` 重建学习历史，并在学习线补充 11 个生产 Python 文件的中文注释和 Docstring。主工作树已经干净地位于本地 `Learn/CUMULATIVE`，仓库只登记这一个工作树。

重建前的 `Learn/CUMULATIVE@218f6d3f3a02fa0f24bf6a7707a0e6293a77a9a7` 与实现端点代码树完全相同，但 13 个提交混合了 Markdown、测试、配置和实现，也没有增加学习注释。临时候选分支和工作树已经在核验后删除。精确本地端点应通过 `git rev-parse Learn/CUMULATIVE` 读取；不要再使用写入本 handoff 前的候选哈希。

## Completed

- 此前 56 项工作区冲突已经收口。两条实现线 `aa1b9e92a95e346b9a8139270854392a981ae4d7` 与 `324cb6f` 合并为双亲实现提交 `ebadbea`。
- 两条实现线相对 `de6a89f` 一共涉及 88 个不同文件；合并端点恰好包含这 88 个文件。`aa1b9e9` 独有的 32 个文件和另一条实现线独有的 39 个文件保持原 blob，其余 17 个重叠文件按逐文件决定合并。
- 独立审计确认 Find_1 Gaussian 受体散射、Find_2 的 56 通道密度调整、辅助监督范围、checkpoint 快照恢复、Selector 惰性输入、CPU/Gloo 指标和 Stage1 producer 契约均保留。
- 本轮补充了请求比例冻结、Dataset 三项辅助标签、批次 offsets、体素输出头、全体素损失、多分类 PRAUC、训练协调和源码快照说明。
- 重建后的学习提交依次说明文档与执行背景、辅助标签与训练请求、体素头与损失指标、源码快照恢复、推理与 selector，最后集中提交全部测试文件。

## Decisions

- 测试文件统一放在学习历史最后一个提交。
- 纯注释和不参与运行的 Docstring 可以只存在于学习线。
- 配置、测试和所有可执行逻辑必须与 `ebadbea` 一致。
- 未经用户明确允许，不推送本地分支或改写远端 GitHub 历史。

## Validation

- 学习端点与实现端点只有 11 个 Python 文件存在文本差异。
- 11 个文件剥离 Docstring 后的 AST 完全一致。
- `configs/` 和 `tests/` 在两个端点逐字一致。
- 两个端点对本轮 15 个可在当前 Windows 环境加载的测试文件分别得到 `96 passed`。
- `tests/model/test_online_pdb_feature.py` 和完整测试收集需要当前机器没有安装的 `torch_cluster`；该环境缺失同时影响两个端点，不是学习注释造成的失败。

## Open Questions

- 是否把本地重建后的 `Learn/CUMULATIVE` 推送到远端，必须由用户另行授权。

## Next Actions

本轮本地整理动作均已完成：

- `Learn/auxiliary-supervision-find1` 已指向重建历史中的快照恢复提交。
- `Learn/stage1-consolidated` 与 `Learn/CUMULATIVE` 已指向通过核验的最终学习端点。
- 临时候选分支、注释源分支和两个临时工作树已经删除。

唯一仍需用户另行决定的事项，是是否把本地重建后的学习历史推送到远端。

## Files To Reopen

- `src/datasets/stage1_requests.py`
- `src/datasets/stage1_dataset.py`
- `src/model/stage1_voxel_backbone.py`
- `src/wrappers/voxel_point_stage1_losses.py`
- `src/wrappers/voxel_point_stage1_metrics.py`
- `CLAUDE/memory/projects/pocket-plus.json`
