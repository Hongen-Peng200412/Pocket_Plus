# Stage1 centered batch12 直连推理交接

## 本轮目标

修复 centered 推理固定 `batch_size=1`、`FullForwardCallback` 伪解耦、正式代码重复 shape/type `raise`、Selected 吞异常并编码 `failed`，以及结构化 Docstring 没有展开字段契约的问题。同步 Pocket Plus、AdaLigand 数据契约和长期代码习惯，全部改动保持 unstaged。

## 已完成

- `src/inference/centered.py` 删除 `FullForwardCallback`、`make_model_centered_callback`、正式 `output_adapter`、单字段转换加校验的薄辅助函数，以及 CLG 单条 forward 包装。
- centered 领域函数直接接收完整 wrapper、`CenteredBatchBuilder` 和 `centered_batch_size`。`iter_model_centered_payloads` 用 `torch.inference_mode()` 按有序请求切片运行完整 forward。
- `iter_stage1_centered_batch_payloads` 从模型顶层权威 `voxel_features` 直键读取五张 V 网格：dense V 按 batch 第 0 维拆分，Find A 按 forward 后 `atom_counts` 连续段拆分，P 按 `anchor_batch_index` 归属拆分。
- `Stage1RuntimeAssembly` 的 centered materializer 接收请求序列，Stage1Dataset 逐 BOX 现场物化，训练同源 Collator 生成真实 B 批次；CLI 新增 `--centered-batch-size`，正式默认 12。
- Selected 领域状态收敛为 `success=0`、`empty=1`、`no_overlap=2`。模型 forward 或实现异常自然向上传播，当前 role 不发布 `_COMPLETE`；产物 schema 删除 `failed=3`。
- `src/inference/centered.py` 不再手写内部 shape/type/字段对齐 `raise`，也没有用 `assert` 替代；归档发布边界仍保留集中 schema 校验。
- 删除原代码中“一行数组转换 + shape 校验”的 `_single_feature_grid`、`_aligned_rows` 等薄函数；一行 resolver 转换也已内联。
- 按 `technical-expression-workflow` 展开本轮关键 Docstring：说明 wrapper/builder 的正式调用者、batch dense/ragged 契约、数组 dtype/shape/轴顺序/单位、F1/CLG/Selected 返回字段组与索引目标。
- 后续按 `code-comment-style-cn` 再次细化了结构化返回值：`iter_stage1_centered_batch_payloads` 的 8 个 dense/V 键、8 个 A 键、5 个 P 键，以及 F1/CLG/Selected entry 的共同键、角色键和条件键均逐键列出；每个字段项保持单行物理布局并写明 dtype、shape、坐标/单位和对齐目标。
- 更新 `src/inference/README.md`、`src/artifacts/readme.md`、`talk/stage1_plus的学习注释.md`、`talk/代码习惯.md`，并同步外部 `C:\Users\15919\Desktop\AdaLigand\文档\讨论\BOX-level数据契约.md`。

## 关键决策

- callback 不会消除运行时依赖；领域函数早晚需要 Dataset、Torch 和 wrapper 时，应显式传递真实依赖。
- 性能是正式契约。collator 和模型支持 B>1 时禁止硬写 1；默认 12 必须由跨边界测试证明真正形成 `12 + 尾批`。
- 测试桩实现正式 batch 与 wrapper 输出，不向生产代码注入测试 adapter。
- 受控内部函数不重复写字段、dtype、shape、参数校验；测试负责契约，正式归档发布边界负责磁盘 schema。
- 预期业务状态用状态码；异常不是数据项，不能吞掉后伪造 `failed`。
- 模型输出只读取当前权威直键，不在领域代码中多层 fallback 猜 schema。

## 验证

- 修改前基线：`18 passed`。
- 修改后定向测试：`tests/inference/test_stage1_centered.py`、`tests/inference/test_stage1_assembly_cli.py`、`tests/artifacts/test_stage1_states.py` 共 `20 passed in 4.19s`。
- 新覆盖包括：13 个请求在 batch 12 下真实形成 `[12,1]` 两次 forward、B=2 Find A/P ragged 拆分、BF16 到 NumPy 桥接、Selected 三状态、forward 异常向上传播、CLI 默认值 12。
- `py_compile` 已覆盖本轮修改的 Python 文件。
- 全仓 `pytest -q` 在测试收集阶段被仓库根目录缺少 `.project-root` 阻断；报错来自 `src/train.py` 的 `rootutils.setup_root`，发生在两个与本轮 centered 改动无关的训练测试导入阶段。

## 工作区与 Git

- 当前分支：`Learn/CUMULATIVE`，本轮没有切分支、提交或暂存。
- 用户在本轮开始前已经暂存 `src/inference/centered.py` 的既有修改；本轮对该文件的修改叠加为 unstaged，因此状态应为 `MM`。
- 其余本轮 Pocket Plus 文件均保持 unstaged。不要执行 `git add`。
- AdaLigand 契约文件位于另一个仓库/目录，不会出现在 Pocket Plus 的 `git status` 中，需要单独查看其工作区状态。

## 后续检查

1. 重新运行最新定向 20 项测试并记录精确结果。
2. 用 `git diff --check`、`py_compile` 和关键术语 `rg` 做最终静态收口。
3. 检查 Pocket Plus `git status --short`，确认本轮文件没有进入 index，`centered.py` 保持原有 staged 加本轮 unstaged 的 `MM`。
4. 若要跑全仓测试，先由用户决定是否恢复仓库根 `.project-root`；不要为本任务擅自创建该项目标记。
