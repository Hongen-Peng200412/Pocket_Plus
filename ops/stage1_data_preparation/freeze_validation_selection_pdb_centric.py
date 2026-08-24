"""从正式 V3 validation BOX pool 冻结 PDB 中心验证请求。

本文件只生成服务器路径
``/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool/validation_selection_pdb_centric.npz``。
输入路径、输出路径、随机种子和三个采样参数均在模块常量中固定，不接受命令行参数。

从 Pocket_Plus 根目录运行
``python -m ops.stage1_data_preparation.freeze_validation_selection_pdb_centric``。
模块方式保证项目根目录参与 Python 包解析，同时没有向脚本增加可变参数。

读者应先看 :func:`main`。函数复用
:class:`src.datasets.stage1_requests.Stage1TrainingRequestSet` 的 epoch 0 请求，再把
每个请求还原为 validation PDB、occurrence 和候选编号。NPZ 不复制 BOX 起点，实际
ZYX 起点仍由 ``validation/{pdb_id}.npz`` 保存。

文件字段：
    - validation_pdb_id：定宽 bytes，``(P,)``；P 个 validation PDB 身份，其他 ``*_pdb_index`` 字段索引本数组第一维。
    - center_pdb_index：int32，``(0,)``；PDB 中心规则不生成 center BOX。
    - center_occurrence_id：int32，``(0,)``；与 ``center_pdb_index`` 对齐的空 occurrence 编号数组。
    - bias_pdb_index：int32，``(N_bias,)``；每个 bias BOX 所属 PDB 在 ``validation_pdb_id`` 中的编号。
    - bias_occurrence_id：int32，``(N_bias,)``；与 ``bias_pdb_index`` 对齐的真实配体 occurrence 编号。
    - bias_candidate_index：int16，``(N_bias,)``；与前两个 bias 字段对齐，数值索引对应 PDB pool 的 ``bias_start_zyx`` 第二维。
    - context_pdb_index：int32，``(N_context,)``；每个 context BOX 所属 PDB 在 ``validation_pdb_id`` 中的编号。
    - context_candidate_index：int32，``(N_context,)``；与
      ``context_pdb_index`` 对齐，数值索引对应 PDB pool 的
      ``context_start_zyx`` 第一维。
    - pdb_foreground_box_num：int32 标量；每个 PDB 的目标 bias BOX 数量，固定为 25。
    - pdb_foreground_fraction_target：float64 标量；bias BOX 占目标 bias 与 context BOX 总数的比例，固定为 0.5。
    - pdb_occurrence_foreground_box_cap：int32 标量；单个 occurrence 的 bias BOX 数量上限，固定为 5。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ops.stage1_data_preparation.atomic_io import atomic_save_npz
from src.datasets.stage1_requests import Stage1TrainingRequestSet


BOX_POOL_ROOT = Path(
    "/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3/box_pool"
)
VALIDATION_POOL_DIRECTORY = BOX_POOL_ROOT / "validation"
OUTPUT_PATH = BOX_POOL_ROOT / "validation_selection_pdb_centric.npz"
REQUEST_SEED = 3407
PDB_FOREGROUND_BOX_NUM = 25
PDB_FOREGROUND_FRACTION_TARGET = 0.5
PDB_OCCURRENCE_FOREGROUND_BOX_CAP = 5


def main() -> None:
    """冻结正式 validation 的 epoch 0 请求并原子替换目标 NPZ。

    输入：
        - ``VALIDATION_POOL_DIRECTORY``：V3 validation PDB pool 目录；由模块常量固定。
        - ``REQUEST_SEED`` 与三个 ``PDB_*`` 常量：与正式 Hydra Dataset 配置相同的确定性采样参数。

    状态变化：
        - 原子写入 ``OUTPUT_PATH``；重复运行会用同一组确定性字段替换已有文件，
          不修改原 ``validation_selection.npz``、单 PDB pool、manifest、config、
          summary 或 ``_COMPLETE``。
    """
    # Stage1TrainingRequestSet；validation manifest 的 epoch 0 PDB 中心请求集合。
    request_source = Stage1TrainingRequestSet(
        VALIDATION_POOL_DIRECTORY,
        seed=REQUEST_SEED,
        pdb_foreground_box_num=PDB_FOREGROUND_BOX_NUM,
        pdb_foreground_fraction_target=PDB_FOREGROUND_FRACTION_TARGET,
        pdb_occurrence_foreground_box_cap=PDB_OCCURRENCE_FOREGROUND_BOX_CAP,
    )
    # tuple[ResolvedStage1Crop, ...]；按 validation manifest PDB 顺序排列的 bias 与 context 请求。
    requests = request_source.requests
    # tuple[str, ...]；validation manifest 中的 PDB 身份；每个 PDB 首次出现在自己的 bias 请求中。
    pdb_ids = tuple(dict.fromkeys(request.pdb_id for request in requests))
    # dict[str, int]；PDB 身份到 validation_pdb_id 第一维编号的映射。
    pdb_index_by_id = {pdb_id: pdb_index for pdb_index, pdb_id in enumerate(pdb_ids)}
    # list[ResolvedStage1Crop]；全部 bias 请求；每项同时保存 occurrence 和 30 个 bias 候选中的编号。
    bias_requests = [request for request in requests if request.role == "bias"]
    # list[ResolvedStage1Crop]；全部 PDB 级 context 请求；occurrence_id 固定为 None。
    context_requests = [request for request in requests if request.role == "context"]
    # int；validation_pdb_id 定宽 bytes dtype 容纳最长 UTF-8 PDB 身份所需的字节数。
    max_pdb_width = max(len(pdb_id.encode("utf-8")) for pdb_id in pdb_ids)
    # dict[str, np.ndarray]；validation_selection_pdb_centric.npz 的完整字段集合。
    arrays = {
        "validation_pdb_id": np.asarray(pdb_ids, dtype=f"S{max_pdb_width}"),
        "center_pdb_index": np.empty(0, dtype=np.int32),
        "center_occurrence_id": np.empty(0, dtype=np.int32),
        "bias_pdb_index": np.asarray(
            [pdb_index_by_id[request.pdb_id] for request in bias_requests],
            dtype=np.int32,
        ),
        "bias_occurrence_id": np.asarray(
            [int(request.occurrence_id) for request in bias_requests],
            dtype=np.int32,
        ),
        "bias_candidate_index": np.asarray(
            [int(request.candidate_index) for request in bias_requests],
            dtype=np.int16,
        ),
        "context_pdb_index": np.asarray(
            [pdb_index_by_id[request.pdb_id] for request in context_requests],
            dtype=np.int32,
        ),
        "context_candidate_index": np.asarray(
            [int(request.candidate_index) for request in context_requests],
            dtype=np.int32,
        ),
        "pdb_foreground_box_num": np.asarray(
            PDB_FOREGROUND_BOX_NUM,
            dtype=np.int32,
        ),
        "pdb_foreground_fraction_target": np.asarray(
            PDB_FOREGROUND_FRACTION_TARGET,
            dtype=np.float64,
        ),
        "pdb_occurrence_foreground_box_cap": np.asarray(
            PDB_OCCURRENCE_FOREGROUND_BOX_CAP,
            dtype=np.int32,
        ),
    }
    atomic_save_npz(OUTPUT_PATH, arrays, compressed=False)
    print(
        f"wrote {OUTPUT_PATH}: pdb={len(pdb_ids)}, "
        f"bias={len(bias_requests)}, context={len(context_requests)}"
    )


if __name__ == "__main__":
    main()
