# Handoff: 桌面仓库迁移与冻结 eval 审查

Date: 2026-06-22

## Current State

当前主开发目录已切换到 `C:\Users\15919\Desktop\Pocket_Plus`，不再建议在 OneDrive 原目录 `D:\OneDrive\My_Project\Pocket_Plus` 中继续做 Git 提交、拉取或日常开发。OneDrive 原目录此前出现过 `.git/index.lock`、GitHub Desktop/Git 进程占用、OneDrive 并发同步导致的索引读取风险；桌面仓库现在是推荐使用的主工作区。

桌面仓库当前分支为 `大重构`，HEAD 与远端 `origin/大重构` 均为 `f4c3e5c 跟踪项目文档与记忆目录`。该提交把 `CLAUDE/` 和 `docs/` 纳入版本控制，并调整 `.gitignore`，保留项目记忆与文档目录可跟踪；`Ligand/logs/` 与 `Ligand/**/__pycache__/` 仍忽略。

已用 SHA256 逐文件比对确认：排除 `.git` 后，OneDrive 原目录与桌面目录的 1877 个项目文件内容完全一致，只有 Git 历史/索引状态不同。OneDrive 原目录仍停在旧提交视角，因而会显示 `.gitignore` 修改、`CLAUDE/` 与 `docs/` 未跟踪；这不是内容差异，而是 Git 状态差异。

当前桌面仓库仍有一组未提交代码改动，集中在冻结模块 train/eval 语义：

- `src/train.py`
- `src/wrappers/voxel_point_stage1.py`
- `src/utils/module_freeze.py`（新增）

## Completed

完成了从 OneDrive 工作区到桌面工作区的迁移核查与 Git 修复：

- 修复本机 Git 代理到 `http://127.0.0.1:7897`，`git ls-remote` 可访问远端。
- 将桌面仓库切到 `大重构` 并对齐 `origin/大重构`。
- 在桌面仓库提交并推送 `f4c3e5c 跟踪项目文档与记忆目录`，使 `CLAUDE/`、`docs/` 进入 Git。
- 解释了另一台机器按钮从 Push 变 Pull 的原因：远端新增了本机推送的提交，另一台机器需要先拉取。
- 确认 OneDrive 原目录与桌面目录实际文件内容完全一致，但不建议继续在 OneDrive 目录操作 Git。

完成了对当前冻结 eval 改动的静态审查：

- `src/utils/module_freeze.py` 新增 `set_fully_frozen_submodules_eval(root)`，扫描所有子模块，把递归参数全部 `requires_grad=False` 的极大冻结子树切到 `eval()`，并返回冻结子树数与带 running 统计的归一化层数量。
- `src/train.py` 在 `_apply_frozen_module()` 冻结参数后调用该工具，初始固定冻结子树，并打印 `frozen→eval` 日志。
- `src/wrappers/voxel_point_stage1.py` 覆写 `train(mode=True)`，在 Lightning 每次把模型切回 train 后，重新把完全冻结子树切到 eval，避免 frozen BN running stats 漂移与 frozen Dropout 随机。

## Decisions

主开发目录采用 `C:\Users\15919\Desktop\Pocket_Plus`。OneDrive 原目录只作为过渡备份，不作为活跃 Git 仓库。

`CLAUDE/`、`CLAUDE_PLANS/`、`docs` 是工作必需内容，需要保留和版本化。此前 `.gitignore` 中忽略它们的规则已被注释掉，并已提交。

对训练冻结逻辑的当前判断：

- 普通默认训练若使用 `configs/frozen_module/none.yaml`，`frozen_module: null`，则 `_apply_frozen_module()` 不执行；目前未发现模型里默认 `requires_grad=False` 的参数，因此新增 `train()` 覆写应为 no-op。
- CPC stage1 实际使用 `configs/frozen_module/stage1.yaml`，会冻结 `backbone.atom_head.real_to_pseudo.*`、`backbone.atom_head.pseudo_to_real.*`、`backbone.sparse_refine_head.*`。静态检查这些冻结子树主要是 Linear / SiLU / LayerNorm，没有发现 BatchNorm / Dropout，因此切 eval 理论上不改变 forward 数值。
- CPC stage2/stage3 冻结 trunk/head 后，让冻结子树保持 eval 是正确方向，可避免“参数不更新但 BN running stats 仍漂移”的隐性问题。

## Open Questions

还不能仅凭静态 diff 绝对保证 stage2/stage3 指标一定良性。该改动会改变冻结模块的训练动态：被冻结部分的 Dropout 会关闭、BatchNorm running stats 会固定。这是更标准的冻结语义，但最终收益需要 smoke test 或短训练确认。

CPC stage1 是否需要做一次前向一致性验证：在同一随机种子、同一 checkpoint/初始化和同一 batch 下，对比改动前后 logits/loss 是否逐元素一致。静态判断认为应一致，但尚未实际跑。

## Next Actions

建议先不要提交冻结 eval 这组三文件，直到完成最小验证。

推荐验证顺序：

1. 在桌面仓库运行普通 stage1 / `frozen_module: null` 的最小启动或 smoke，确认不打印 `frozen→eval` 且能正常进入训练。
2. 运行 CPC stage1 小 batch / 短步数，确认冻结模块计数合理，且没有数值异常；如条件允许，做改动前后前向一致性对比。
3. 运行 stage2 或 stage3 小数据几十步，确认 `frozen→eval` 日志里冻结子树与 BN 数量符合预期，loss 不 NaN，梯度只落在预期可训练模块。
4. 验证通过后，再将 `src/train.py`、`src/wrappers/voxel_point_stage1.py`、`src/utils/module_freeze.py` 一起提交。

另一台机器建议重新使用非 OneDrive 目录 clone/checkout `大重构` 分支，避免在 OneDrive 原目录中处理那批重复的 67 个未提交改动。

## Files To Reopen

- `C:\Users\15919\Desktop\Pocket_Plus\src\train.py`
- `C:\Users\15919\Desktop\Pocket_Plus\src\wrappers\voxel_point_stage1.py`
- `C:\Users\15919\Desktop\Pocket_Plus\src\utils\module_freeze.py`
- `C:\Users\15919\Desktop\Pocket_Plus\configs\frozen_module\stage1.yaml`
- `C:\Users\15919\Desktop\Pocket_Plus\configs\frozen_module\stage2.yaml`
- `C:\Users\15919\Desktop\Pocket_Plus\configs\frozen_module\stage3.yaml`
- `C:\Users\15919\Desktop\Pocket_Plus\configs\experiment\CPC1\trunk_main.yaml`
- `C:\Users\15919\Desktop\Pocket_Plus\configs\experiment\CPC2\heads_trunk_main.yaml`
- `C:\Users\15919\Desktop\Pocket_Plus\configs\experiment\CPC3\refine_tversky_73.yaml`
