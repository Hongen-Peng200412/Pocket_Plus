# Handoff: Stage1 V3 的 PDB 级 split 清单

Date: 2026-08-21

## Current State

`ops/stage1_data_preparation/freeze_split.py` 已增量加入显式子命令 `publish-pdb-splits`。该子命令读取既有五份 occurrence 级 split JSON，仅提取、转为小写、去重并排序其中的 `pdb_id`，然后向指定目录写出五份纯 `list[str]` JSON：`train.json`、`validation.json`、`calibration.json`、`held_out.json` 和 `quarantine_missing_release.json`。

新增行为默认不执行，原有 `fetch-release-dates` 与 `freeze` 命令行为不变。服务器尚未执行该命令，Git 尚未操作。

## Completed

- 在 `freeze_split.py` 中加入简单的 `publish_pdb_splits` 函数和 `publish-pdb-splits` CLI 子命令。
- 在 CLI 定义旁用注释写明现有 Stage1 V3 产物的一次性补建命令，输出目录为 `stage1_preparation_box_pool_3/split/pdb_split`。
- 在 `ops/stage1_data_preparation/tests/test_freeze_split.py` 中加入定向测试，确认 occurrence 去重和五份输出文件的内容。
- `Pocket_Plus_windows` 环境定向测试通过：`2 passed`。

## Decisions

- PDB 级清单只是既有 occurrence split 的身份投影，不重新执行日期、分辨率、`cc_contour`、资产或 occurrence 过滤。
- 新命令只写五份 PDB ID JSON，不额外写 `summary.json`、`_COMPLETE` 或其他发布产物。
- 用户明确要求：不要过度扩展，不要过度防御。简单集合投影保持简单；没有明确要求的审计、发布协议、交叉校验和附加文件不得自行增加。
- 用户明确要求在再次允许前不操作 Git、不访问服务器。

## Next Actions

1. 等待用户审阅本地代码。
2. 得到明确许可后，才可处理 Git 或在服务器运行代码注释中的一次性补建命令。
3. 服务器执行后，只核对五份 PDB ID JSON 是否存在以及数量是否合理，不自行扩展产物契约。

## Files To Reopen

- `ops/stage1_data_preparation/freeze_split.py`
- `ops/stage1_data_preparation/tests/test_freeze_split.py`

