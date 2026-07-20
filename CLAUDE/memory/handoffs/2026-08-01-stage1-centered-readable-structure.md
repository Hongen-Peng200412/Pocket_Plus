# Handoff: Stage1 centered 可读性结构重构

Date: 2026-08-01

## Current State

`src/inference/centered.py` 已按“冷读工具 → 分隔线 → 主流程”重排；`assembly.py` 和 `runner.py` 同步收缩了无必要的适配层。所有本轮代码改动保留在工作区，未新增 staged 内容。

## Completed

- centered forward 继续使用显式 `centered_batch_size`，默认 12；没有回退到 batch size 1。
- 删除 centered 内的单行概率包装、request 包装、一次性 Find 表包装、坐标包装、Selected 空 entry/归一化包装。
- 保留有独立语义的 NumPy/torch 转换、稳定 sigmoid、空间包络、entry 组装和完整 forward 拆分函数。
- 直接使用完整 Dataset/Collator batch 和 wrapper，不再引入 `FullForwardCallback` 或 `output_adapter`。
- runner 直接使用共享的 `centered_start_from_centroid_zyx`，并把 F1/CLG 上下文加载移出 callback 内部的嵌套函数。
- assembly 删除未被调用的 task-first occurrence 薄适配方法，并把 dataset contract/device batch 两个冷读工具移到类定义之前。
- 修正 `python-writing-style-cn`：删除“用一句话概括模块职责”的边界表述，改为概括篇幅、句数和组织形式不受限制。
- `talk/代码习惯.md` 增加函数概括和大文件布局规则。

## Decisions

- 保持 centered、artifact 和 BOX-level 数据契约不变；`A_feat_L4/P_feat_L4` 继续不落盘，这是现有契约的明确设计。
- 不围绕用户暂时撤回的“两个旧函数”做算法替换；保留等价 NumPy 操作，仅内联不具独立语义的包装。
- 正式生产代码不新增形状校验 `raise`；外部文件和发布边界的既有校验不在本轮扩大修改。

## Open Questions

- 是否将相同的“冷读工具 + 主流程”布局规则继续推广到 inference 目录的其他历史文件。
- 是否需要把全仓库正式代码的 `raise` 政策收紧为更强的禁用规则；当前只落实本轮涉及路径。

## Next Actions

- 用户审阅结构和调用链；如确认，再决定是否同步整理其他 inference 模块。
- 保持 batch=12 的跨批量测试和 centered/artifact focused 测试作为回归入口；当前 inference/artifact focused suite 为 39 passed。

## Files To Reopen

- `src/inference/centered.py`
- `src/inference/assembly.py`
- `src/inference/runner.py`
- `C:/Users/15919/.codex/skills/python-writing-style-cn/SKILL.md`
- `talk/代码习惯.md`
