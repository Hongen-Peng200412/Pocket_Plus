# Read offset 必须使用真实行号

Type: gotcha
Date: 2026-05-30
Tags: claude-tools, read, edit, workflow

## Context

用户指出我多次在 `Read` 工具里传入巨大无效 offset，例如把字符位置、旧输出编号或拼接数字当成行号，导致连续读取失败。

## Memory

`Read` 工具的 `offset` 是文件行号偏移，必须是基于当前文件实际行数的合理小整数。不要把字符位置、旧工具输出编号、搜索结果编号、拼接数字或任何未验证的大数传给 `offset`。

如果 `Edit` 报 “File has not been read yet”，正确处理方式是先用 `Grep` 定位唯一文本，或用 `Read` 从 `offset=0`、已知真实行号附近重新读取目标片段；然后再执行 `Edit`。不要连续猜 offset。

## When To Use

每次需要用 `Read` 定位编辑片段，尤其是在大文件、刚切换工具状态、或 `Edit` 提示未读取文件时。

## Related Files

- `src/wrappers/voxel_point_stage1_diagnostics.py`
