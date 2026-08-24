# -*- coding: utf-8 -*-
"""定义并展开 Stage1 V3 几何池中的 80³ BOX 请求。

主要入口是 :class:`Stage1TrainingRequestSet`、:func:`load_validation_selection` 和
:func:`build_request_source`。训练请求按 PDB 分配 bias 与 context BOX，验证请求从
``validation_selection_pdb_centric.npz`` 读取冻结索引。本模块返回带有完整图 ZYX
起点的请求，不读取密度、受体原子或监督数组。

V3 pool 的顶层文件是 ``manifest.json``、``_COMPLETE`` 和历史构建配置
``config.json``。每个 split 目录保存一个 PDB 一个 NPZ，NPZ 字段由
:class:`_PdbPool` 说明。活动训练参数由 Hydra Dataset 配置提供，不读取
``config.json::entry_ratio``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


STAGE1_BOX_SHAPE_ZYX = (80, 80, 80)
BOX_POOL_MANIFEST_FILENAME = "manifest.json"
_VALID_ROLES = {"center", "bias", "context", "sliding", "centered"}


def resolve_stage1_start(
    requested_start_zyx: Sequence[int | float],
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """把任意请求起点解析为完整图内合法的 BOX corner index。

    输入参数:
        - requested_start_zyx: Sequence[int | float]；有限候选 BOX 起点，三个分量按完整图离散体素的 ZYX 轴顺序排列，浮点值按 ``numpy.rint`` 语义取整；本函数不负责把非有限输入转换为业务错误。
        - full_shape_zyx: Sequence[int], 完整图的 ZYX 体素形状；三个轴都必须不小于 ``box_shape_zyx``。
        - box_shape_zyx: Sequence[int], BOX 的 ZYX 体素形状；默认值是 Stage1 固定的 ``(80, 80, 80)``。

    返回值:
        - start_zyx: tuple[int, int, int], 将候选起点按每个轴裁剪到 ``0`` 至 ``full_shape_zyx - box_shape_zyx`` 后的完整图零基 ZYX 起点；该起点对应的 BOX 不需要补零。

    失败语义:
        - 输入形状不是 ``(3,)`` 或完整图小于 BOX 时立即抛出 ``ValueError``，不返回隐式回退起点；非有限起点不属于本函数的有效输入契约。
    """

    requested = np.rint(np.asarray(requested_start_zyx, dtype=np.float64)).astype(np.int64)
    full_shape = np.asarray(full_shape_zyx, dtype=np.int64)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    if requested.shape != (3,) or full_shape.shape != (3,) or box_shape.shape != (3,):
        raise ValueError("BOX 起点、完整图形状和 BOX 形状都必须包含三个 ZYX 分量。")
    if np.any(full_shape < box_shape):
        raise ValueError("完整图三轴必须不小于 Stage1 BOX。")
    return tuple(int(value) for value in np.clip(requested, 0, full_shape - box_shape))


def centered_start_from_sparse_mask(
    sparse_voxel_zyx: np.ndarray,
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """根据 occurrence 非空体素的几何中心计算合法 BOX 起点。

    输入参数:
        - sparse_voxel_zyx: np.ndarray, ``(K_occ, 3)`` 的完整图零基 ZYX 体素索引；每一行代表 occurrence 的一个非空体素，``K_occ`` 必须大于零。
        - full_shape_zyx: Sequence[int], 完整图 ZYX 体素形状，传给 :func:`centered_start_from_centroid_zyx` 做边界裁剪。
        - box_shape_zyx: Sequence[int], BOX 的 ZYX 体素形状；默认值是 ``(80, 80, 80)``。

    返回值:
        - start_zyx: tuple[int, int, int], 将稀疏体素均值置于 BOX 中心后得到的完整图零基 ZYX 起点。
    """

    sparse = np.asarray(sparse_voxel_zyx, dtype=np.int64)
    if sparse.ndim != 2 or sparse.shape[1] != 3 or sparse.shape[0] == 0:
        raise ValueError("occurrence 稀疏体素必须是非空 (K,3) ZYX 数组。")
    return centered_start_from_centroid_zyx(
        sparse.astype(np.float64).mean(axis=0),
        full_shape_zyx,
        box_shape_zyx,
    )


def centered_start_from_centroid_zyx(
    centroid_zyx: Sequence[int | float],
    full_shape_zyx: Sequence[int],
    box_shape_zyx: Sequence[int] = STAGE1_BOX_SHAPE_ZYX,
) -> tuple[int, int, int]:
    """把完整图 ZYX 体素质心放到 BOX 中心并解析边界。

    输入参数:
        - centroid_zyx: Sequence[int | float], 完整图连续 ZYX 体素坐标的质心；体素中心语义由调用者提供，函数只按同一坐标约定计算 BOX corner。
        - full_shape_zyx: Sequence[int], 完整图 ZYX 体素形状。
        - box_shape_zyx: Sequence[int], BOX 的 ZYX 体素形状；默认值是 ``(80, 80, 80)``。

    返回值:
        - start_zyx: tuple[int, int, int], 质心减去半个 BOX 形状并经 :func:`resolve_stage1_start` 裁剪后的完整图零基 ZYX 起点。
    """

    centroid = np.asarray(centroid_zyx, dtype=np.float64)
    box_shape = np.asarray(box_shape_zyx, dtype=np.int64)
    requested = centroid + 0.5 - box_shape.astype(np.float64) / 2.0
    return resolve_stage1_start(requested, full_shape_zyx, box_shape_zyx)


def _string_array(values: np.ndarray) -> list[str]:
    """把 NPZ 字符串字段规范化为小写 Python 字符串列表。

    输入参数:
        - values: np.ndarray, 任意可展平的 bytes 或 Unicode 字段；每个元素表示一个 PDB identity。

    返回值:
        - strings: list[str], 按输入展平顺序解码、去除首尾空白并转为小写的字符串；不负责去重或验证 identity 是否存在。
    """

    result: list[str] = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(str(value).strip().lower())
    return result


def _load_manifest_pool_paths(
    box_pool_root: str | Path,
    split_name: str,
) -> tuple[tuple[str, Path], ...]:
    """读取已发布的 V3 manifest 并返回指定 split 的 PDB NPZ 路径。

    输入参数:
        - box_pool_root: str | Path, V3 pool 根目录；必须同时包含 ``_COMPLETE`` 和 ``manifest.json``，``manifest.json`` 的 ``splits[split_name]`` 保存 split 清单。
        - split_name: str, split 名称；函数按小写名称读取 ``train`` 或 ``validation`` 等 manifest split，不改变清单内顺序。

    返回值:
        - entries: tuple[tuple[str, Path], ...], 每项为 ``(pdb_id, pool_path)``；``pdb_id`` 是小写身份，``pool_path`` 是根目录下由 manifest ``path`` 字段解析出的 NPZ 路径，返回顺序与 manifest 一致。

    manifest 约束:
        - 每项必须提供非空 ``pdb_id`` 和相对 ``path``；绝对路径、包含 ``..`` 的路径和空 split 直接失败，不扫描 manifest 未列出的文件。
    """

    root = Path(box_pool_root)
    if not (root / "_COMPLETE").is_file():
        raise FileNotFoundError(f"BOX pool 尚未完成发布：{root}。")
    manifest_path = root / BOX_POOL_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_rows = manifest["splits"][str(split_name).lower()]
    entries: list[tuple[str, Path]] = []
    for record in split_rows:
        pdb_id = str(record["pdb_id"]).strip().lower()
        relative_path = Path(str(record["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"BOX pool path 必须位于 manifest 根目录内：{relative_path}。")
        entries.append((pdb_id, root / relative_path))
    if not entries:
        raise ValueError(f"{manifest_path}: splits.{split_name} 不能为空。")
    return tuple(entries)


@dataclass(frozen=True)
class _PdbPool:
    """保存一个 PDB pool 的 occurrence 身份和完整图 BOX 起点。

    字段:
        - pdb_id: str, manifest 中的小写 PDB identity。
        - occurrence_id: int32, ``(O,)``，该 PDB 的 occurrence 编号；与 ``center_start_zyx`` 和 ``bias_start_zyx`` 的第一维按位置对齐。
        - center_start_zyx: int32, ``(O, 3)``，每个 occurrence 的完整图零基 ZYX BOX corner index。
        - bias_start_zyx: int32, ``(O, 30, 3)``，每个 occurrence 的 30 个 bias 候选起点；第二维是候选编号，最后一维是 ZYX。
        - context_start_zyx: int32, ``(C, 3)``，该 PDB 级 context 候选池的起点；第一维不与 occurrence 数量对齐。
    """

    pdb_id: str
    occurrence_id: np.ndarray
    center_start_zyx: np.ndarray
    bias_start_zyx: np.ndarray
    context_start_zyx: np.ndarray


def _load_pdb_pool(path: Path, expected_pdb_id: str) -> _PdbPool:
    """读取并校验一个 PDB 的 V3 pool NPZ。

    输入参数:
        - path: Path, manifest 声明的 PDB pool NPZ 路径；函数只读取该文件，不搜索同目录的其他文件。
        - expected_pdb_id: str, manifest 记录的小写 PDB identity；必须与文件内 ``pdb_id`` 相等。

    文件字段:
        - ``pdb_id``: 标量 bytes 或 Unicode 字符串；当前 NPZ 所属的 PDB identity。
        - ``occurrence_id``: int32, ``(O,)``；occurrence 编号。
        - ``center_start_zyx``: int32, ``(O, 3)``；与 ``occurrence_id`` 第一维逐 occurrence 对齐的完整图零基 ZYX BOX 起点。
        - ``bias_start_zyx``: int32, ``(O, 30, 3)``；与 ``occurrence_id`` 第一维对齐的 30 个 bias 起点，第二维是候选编号，最后一维是 ZYX。
        - ``context_start_zyx``: int32, ``(C, 3)``；当前 PDB 的 context 候选起点，最后一维是 ZYX。

    返回值:
        - pool: _PdbPool, 把上述五个字段保存在内存中；函数不复制密度、受体原子或监督数组。

    校验语义:
        - occurrence 对齐维度、bias 候选数和 context 的二维形状不符合契约，或文件 identity 与 manifest 不一致时失败。
    """

    with np.load(path, allow_pickle=False) as data:
        pdb_id = _string_array(data["pdb_id"])[0]
        occurrence_id = np.asarray(data["occurrence_id"], dtype=np.int32)
        center_start_zyx = np.asarray(data["center_start_zyx"], dtype=np.int32)
        bias_start_zyx = np.asarray(data["bias_start_zyx"], dtype=np.int32)
        context_start_zyx = np.asarray(data["context_start_zyx"], dtype=np.int32)
    if pdb_id != expected_pdb_id:
        raise ValueError(f"{path}: PDB 身份 {pdb_id!r} 与 manifest 不一致。")
    occurrence_count = int(occurrence_id.shape[0])
    if (
        center_start_zyx.shape != (occurrence_count, 3)
        or bias_start_zyx.shape != (occurrence_count, 30, 3)
        or context_start_zyx.ndim != 2
        or context_start_zyx.shape[1] != 3
    ):
        raise ValueError(f"{path}: V3 BOX 起点数组形状不符合契约。")
    return _PdbPool(
        pdb_id,
        occurrence_id,
        center_start_zyx,
        bias_start_zyx,
        context_start_zyx,
    )


@dataclass(frozen=True)
class ResolvedStage1Crop:
    """描述一个已经解析到完整图位置的 Stage1 BOX 请求。

    字段:
        - pdb_id: str, 小写 PDB identity；Dataset 用它定位该 PDB 的四类密度、受体与标签文件。
        - box_start_zyx: tuple[int, int, int], 完整图零基 ZYX BOX corner index；三个分量依次为 Z、Y、X，BOX 形状固定为 ``(80, 80, 80)``。
        - require_targets: bool, 是否要求 Dataset 返回训练/验证监督字段；推理滑窗请求可设为 False。
        - role: str, 请求角色；允许值为 ``center``、``bias``、``context``、``sliding`` 或 ``centered``。
        - occurrence_id: int | None, 请求对应的 occurrence 编号；context、sliding 和 centered 请求可以没有该编号。
        - candidate_index: int | None, bias 或 context 候选池中的下标；center、sliding 和 centered 请求通常没有该下标。
    """

    pdb_id: str
    box_start_zyx: tuple[int, int, int]
    require_targets: bool
    role: str
    occurrence_id: int | None = None
    candidate_index: int | None = None

    def __post_init__(self) -> None:
        pdb_id = str(self.pdb_id).strip().lower()
        start = tuple(int(value) for value in self.box_start_zyx)
        role = str(self.role).strip().lower()
        if not pdb_id or len(start) != 3 or role not in _VALID_ROLES:
            raise ValueError("Stage1 BOX 请求缺少合法 PDB 身份、ZYX 起点或角色。")
        object.__setattr__(self, "pdb_id", pdb_id)
        object.__setattr__(self, "box_start_zyx", start)
        object.__setattr__(self, "role", role)


class Stage1TrainingRequestSet:
    """按 epoch 从一个 V3 split pool 生成确定性的 PDB 中心 BOX 请求。

    构造参数：
        - pool_directory：str | Path；``stage1_preparation_box_pool_3/box_pool`` 下的
          ``train`` 或 ``validation`` 目录；父目录提供 ``_COMPLETE`` 和
          ``manifest.json``。
        - seed：int；请求抽样种子；相同 seed、epoch 和 manifest 顺序得到相同请求。
        - pdb_foreground_box_num：int；每个 PDB 的目标 bias BOX 数量；当前正式值为 25。
        - pdb_foreground_fraction_target：float；bias BOX 占目标总 BOX 数量的比例；当前正式值为 0.5。
        - pdb_occurrence_foreground_box_cap：int；单个 occurrence 每个 epoch 最多获得的 bias BOX 数量；当前正式值为 5。

    生成规则：
        - 第 ``i`` 个 PDB 含 ``O_i`` 个 occurrence 时，实际 bias 数量为
          ``min(pdb_foreground_box_num, O_i * pdb_occurrence_foreground_box_cap)``。
        - bias 数量先整除分配给全部 occurrence，余数沿稳定 occurrence 排列按 epoch 循环移动；单个 occurrence 的数量不超过 cap。
        - context 目标数量按 ``round(pdb_foreground_box_num * (1 - fraction) / fraction)`` 计算，不因实际 bias 数量不足而减少。
        - 每个 occurrence 从 30 个 bias 候选中无放回选择分配数量；每个 PDB 从正式 context 候选池中无放回选择目标数量。
        - 请求顺序保持 manifest PDB 顺序；每个 PDB 先按 pool occurrence 顺序排列 bias，再排列 PDB 级 context。

    生命周期：
        - ``set_epoch`` 只在 epoch 改变时重建 ``requests``；训练 DataLoader 不应使用常驻 worker，否则主进程更新的 epoch 请求不会同步到旧 Dataset 副本。

    公开属性：
        - requests：tuple[ResolvedStage1Crop, ...]；当前 epoch 的不可变请求序列。
        - pdb_context_box_num：int；每个 PDB 固定抽取的 context BOX 数量；当前三个正式参数对应 25。
    """

    def __init__(
        self,
        pool_directory: str | Path,
        seed: int,
        pdb_foreground_box_num: int,
        pdb_foreground_fraction_target: float,
        pdb_occurrence_foreground_box_cap: int,
    ) -> None:
        """读取一个 manifest split 的 PDB pool，并立即生成 epoch 0 请求。

        参数：
            - pool_directory：str | Path；已发布的 V3 ``train`` 或 ``validation``
              目录；父目录必须包含 ``_COMPLETE`` 和 ``manifest.json``。
            - seed：int；请求抽样种子；与 manifest PDB 编号和 epoch 共同决定 occurrence 轮转及候选选择。
            - pdb_foreground_box_num：int；每个 PDB 的目标 bias BOX 数量。
            - pdb_foreground_fraction_target：float；bias BOX 占目标总 BOX 数量的比例。
            - pdb_occurrence_foreground_box_cap：int；单个 occurrence 每个 epoch 的 bias BOX 数量上限。

        状态变化：
            - 读取 manifest 声明的 occurrence、bias 和 context 起点数组，保存显式采样参数，并调用 :meth:`set_epoch(0)` 初始化 ``requests``。
        """
        # Path；当前请求集合对应的 train 或 validation pool 目录。
        pool_directory = Path(pool_directory)
        # tuple[_PdbPool, ...]；按 manifest 顺序保存每个 PDB 的 occurrence 与候选 BOX 起点。
        self._pools = tuple(
            _load_pdb_pool(path, expected_pdb_id=pdb_id)
            for pdb_id, path in _load_manifest_pool_paths(pool_directory.parent, pool_directory.name)
        )
        # int；每个 PDB 期望获得的 bias BOX 数量。
        self.pdb_foreground_box_num = int(pdb_foreground_box_num)
        # float；bias BOX 占目标 bias 与 context BOX 总数的比例。
        self.pdb_foreground_fraction_target = float(pdb_foreground_fraction_target)
        # int；单个 occurrence 在一个 epoch 内最多获得的 bias BOX 数量。
        self.pdb_occurrence_foreground_box_cap = int(pdb_occurrence_foreground_box_cap)
        # int；按目标 bias 数量与目标前景比例计算的 PDB 级 context BOX 数量；正式值为 25。
        self.pdb_context_box_num = int(
            round(
                self.pdb_foreground_box_num
                * (1.0 - self.pdb_foreground_fraction_target)
                / self.pdb_foreground_fraction_target
            )
        )
        # int；与 manifest PDB 编号和 epoch 共同组成 NumPy 随机种子。
        self.seed = int(seed)
        # int；当前 requests 对应的 epoch；-1 保证构造时生成 epoch 0。
        self.epoch = -1
        # tuple[ResolvedStage1Crop, ...]；当前 epoch 的全部 PDB 级 bias 与 context 请求。
        self.requests: tuple[ResolvedStage1Crop, ...] = ()
        self.set_epoch(0)

    def _build_epoch_requests(self, epoch: int) -> tuple[ResolvedStage1Crop, ...]:
        """按固定 seed、manifest PDB 编号和 epoch 展开不可变请求序列。

        参数：
            - epoch：int；当前训练周期编号；控制余数 occurrence 的循环位置和 bias/context 候选选择。

        返回值：
            - requests：tuple[ResolvedStage1Crop, ...]；按 manifest PDB 顺序排列；每个 PDB 的 bias 请求位于 context 请求之前，全部请求都要求监督目标。

        科学前提：
            - 正式 V3 pool 的每个 PDB 至少包含一个 occurrence、30 个逐 occurrence bias 候选和超过目标数量的 context 候选。
        """
        # list[ResolvedStage1Crop]；依次累积全部 PDB 的 bias 与 context BOX 请求。
        requests: list[ResolvedStage1Crop] = []
        for pdb_index, pool in enumerate(self._pools):
            # int；当前 PDB 的真实配体 occurrence 数量 O_i。
            occurrence_count = int(pool.occurrence_id.shape[0])
            # int；当前 PDB 的实际 bias BOX 数量 F_i；occurrence 上限不足时允许低于目标数量。
            foreground_box_num = min(
                self.pdb_foreground_box_num,
                occurrence_count * self.pdb_occurrence_foreground_box_cap,
            )
            # int；每个 occurrence 先获得的共同 bias BOX 数量。
            foreground_box_num_per_occurrence, rotating_occurrence_num = divmod(
                foreground_box_num,
                occurrence_count,
            )
            # int32 (O_i,)；每个 occurrence 在当前 epoch 获得的 bias BOX 数量，数值总和为 F_i。
            occurrence_foreground_box_num = np.full(
                occurrence_count,
                foreground_box_num_per_occurrence,
                dtype=np.int32,
            )
            # int64 (O_i,)；当前 PDB 的稳定 occurrence 排列，数值索引 pool.occurrence_id 的第一维。
            occurrence_order = np.random.default_rng(
                np.random.SeedSequence([self.seed, pdb_index, 0])
            ).permutation(occurrence_count)
            # int64 (R_i,)；当前 epoch 额外获得一个 bias BOX 的循环位置，R_i 是 F_i 除以 O_i 的余数。
            rotating_position = (
                int(epoch) * foreground_box_num
                + np.arange(rotating_occurrence_num, dtype=np.int64)
            ) % occurrence_count
            # int64 (R_i,)；当前 epoch 额外获得一个 bias BOX 的 occurrence 编号，数值索引 pool.occurrence_id 的第一维。
            rotating_occurrence_index = occurrence_order[rotating_position]
            occurrence_foreground_box_num[rotating_occurrence_index] += 1
            # Generator；当前 PDB 和 epoch 专属的 bias/context 候选随机流。
            candidate_rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, pdb_index, int(epoch), 1])
            )
            for occurrence_row, occurrence_box_num in enumerate(
                occurrence_foreground_box_num.tolist()
            ):
                # int；当前 occurrence 在 pool.occurrence_id 中保存的正式编号。
                occurrence_id = int(pool.occurrence_id[occurrence_row])
                # int64 (F_occ,)；当前 occurrence 从 30 个 bias 起点中无放回抽中的候选编号。
                bias_indices = candidate_rng.choice(
                    30,
                    size=int(occurrence_box_num),
                    replace=False,
                )
                for candidate_index in bias_indices.tolist():
                    requests.append(
                        ResolvedStage1Crop(
                            pool.pdb_id,
                            tuple(pool.bias_start_zyx[occurrence_row, candidate_index]),
                            True,
                            "bias",
                            occurrence_id,
                            int(candidate_index),
                        )
                    )
            # int64 (C_target,)；当前 PDB 从 context_start_zyx 第一维无放回抽中的候选编号。
            context_indices = candidate_rng.choice(
                int(pool.context_start_zyx.shape[0]),
                size=self.pdb_context_box_num,
                replace=False,
            )
            for candidate_index in context_indices.tolist():
                requests.append(
                    ResolvedStage1Crop(
                        pool.pdb_id,
                        tuple(pool.context_start_zyx[candidate_index]),
                        True,
                        "context",
                        None,
                        int(candidate_index),
                    )
                )
        return tuple(requests)

    def set_epoch(self, epoch: int) -> None:
        """切换训练 epoch，并在 epoch 变化时重建请求源。

        参数：
            - epoch：int；会先转换为 Python ``int``，用于确定性训练抽样。

        状态变化：
            - 新 epoch 调用 :meth:`_build_epoch_requests` 并替换 ``requests``；重复设置相同 epoch 不重新抽样。
            - Dataset/DataLoader 调用方应确保 worker 不持有过期的常驻 Dataset 副本，否则主进程更新不会传播。
        """
        epoch = int(epoch)
        if epoch != self.epoch:
            self.requests = self._build_epoch_requests(epoch)
            self.epoch = epoch

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> ResolvedStage1Crop:
        return self.requests[index]


def load_validation_selection(
    selection_path: str | Path,
    box_pool_root: str | Path,
) -> list[ResolvedStage1Crop]:
    """把冻结的 validation selection NPZ 展开为固定 BOX 请求。

    输入参数：
        - selection_path：str | Path；``validation_selection_pdb_centric.npz``
          路径；该文件只保存 PDB/occurrence/candidate 索引与三个 PDB 中心采样参数，
          不复制 BOX 起点数组。
        - box_pool_root：str | Path；V3 pool 根目录；函数从其 validation manifest 找到每个 PDB 的 pool NPZ。

    selection 文件字段：
        - ``validation_pdb_id``：定宽 bytes 或 Unicode，``(P,)``；PDB identity 表，其他 ``*_pdb_index`` 数组按第一维索引它。
        - ``center_pdb_index`` 和 ``center_occurrence_id``：int32，``(N_center,)``；联合定位每个 occurrence 的 center 起点。
        - ``bias_pdb_index``、``bias_occurrence_id`` 和
          ``bias_candidate_index``：int32/int16，三者均为 ``(N_bias,)``；联合定位
          occurrence 的 30 个 bias 起点中的一个候选。
        - ``context_pdb_index`` 和 ``context_candidate_index``：int32，二者均为 ``(N_context,)``；定位 PDB 级 context 起点。
        - ``pdb_foreground_box_num``：int32 标量；每个 PDB 的目标 bias BOX 数量；正式值为 25。
        - ``pdb_foreground_fraction_target``：float64 标量；bias BOX 占目标总 BOX 数量的比例；正式值为 0.5。
        - ``pdb_occurrence_foreground_box_cap``：int32 标量；单个 occurrence 每个 epoch 的 bias BOX 数量上限；正式值为 5。

    长度与边界契约：
        - 每组字段的第一维必须完全相等，且索引分别落在 ``validation_pdb_id``、对应 PDB occurrence 和候选池范围内；这些条件由冻结 selection 产物保证，本函数不修正或截断不一致数组。

    返回值：
        - requests：list[ResolvedStage1Crop]；按全部 center、全部 bias、全部 context 的固定顺序展开；函数只按已冻结索引查找，不重新随机抽样，也不读取密度或受体数组。
    """

    # Path；当前活动 validation selection NPZ 路径。
    source = Path(selection_path)
    # dict[str, Path]；validation PDB identity 到 manifest 声明 pool 文件的映射。
    manifest_paths = dict(_load_manifest_pool_paths(box_pool_root, "validation"))
    with np.load(source, allow_pickle=False) as data:
        # list[str]；selection 保存的 validation PDB identity 表。
        pdb_ids = _string_array(data["validation_pdb_id"])
        # dict[str, np.ndarray]；按角色保存 PDB、occurrence 与候选索引数组。
        arrays = {
            field_name: np.asarray(data[field_name])
            for field_name in (
                "center_pdb_index",
                "center_occurrence_id",
                "bias_pdb_index",
                "bias_occurrence_id",
                "bias_candidate_index",
                "context_pdb_index",
                "context_candidate_index",
            )
        }
    # dict[str, _PdbPool]；selection 引用的全部 validation PDB 几何候选池。
    pools = {
        pdb_id: _load_pdb_pool(manifest_paths[pdb_id], expected_pdb_id=pdb_id)
        for pdb_id in pdb_ids
    }
    # dict[str, dict[int, int]]；每个 PDB 的 occurrence identity 到 pool 第一维编号的映射。
    occurrence_rows = {
        pdb_id: {
            int(occurrence_id): row_index
            for row_index, occurrence_id in enumerate(pool.occurrence_id.tolist())
        }
        for pdb_id, pool in pools.items()
    }

    # list[ResolvedStage1Crop]；按 center、bias、context 顺序累积冻结请求。
    requests: list[ResolvedStage1Crop] = []
    for pdb_index, occurrence_id in zip(
        arrays["center_pdb_index"], arrays["center_occurrence_id"]
    ):
        # str；当前 center 索引对应的 PDB identity。
        pdb_id = pdb_ids[int(pdb_index)]
        # int；当前 occurrence 在该 PDB pool 第一维中的编号。
        row_index = occurrence_rows[pdb_id][int(occurrence_id)]
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].center_start_zyx[row_index]),
                True,
                "center",
                int(occurrence_id),
            )
        )
    for pdb_index, occurrence_id, candidate_index in zip(
        arrays["bias_pdb_index"],
        arrays["bias_occurrence_id"],
        arrays["bias_candidate_index"],
    ):
        # str；当前 bias 索引对应的 PDB identity。
        pdb_id = pdb_ids[int(pdb_index)]
        # int；当前 occurrence 在该 PDB pool 第一维中的编号。
        row_index = occurrence_rows[pdb_id][int(occurrence_id)]
        # int；当前 bias 请求在 30 个候选中的编号。
        candidate_index = int(candidate_index)
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].bias_start_zyx[row_index, candidate_index]),
                True,
                "bias",
                int(occurrence_id),
                candidate_index,
            )
        )
    for pdb_index, candidate_index in zip(
        arrays["context_pdb_index"], arrays["context_candidate_index"]
    ):
        # str；当前 context 索引对应的 PDB identity。
        pdb_id = pdb_ids[int(pdb_index)]
        # int；当前 context 请求在 PDB 级候选池中的编号。
        candidate_index = int(candidate_index)
        requests.append(
            ResolvedStage1Crop(
                pdb_id,
                tuple(pools[pdb_id].context_start_zyx[candidate_index]),
                True,
                "context",
                None,
                candidate_index,
            )
        )
    return requests


def load_split_pdb_ids(path: str | Path) -> tuple[str, ...]:
    """读取 split JSON 并按首次出现顺序提取唯一 PDB identity。

    输入参数:
        - path: str | Path, JSON split 文件；顶层必须是候选记录列表，每条记录提供 ``pdb_id`` 字段。

    返回值:
        - pdb_ids: tuple[str, ...], 逐记录读取、去除首尾空白、转为小写并按首次出现顺序去重的 PDB identity；不排序，也不检查 PDB 资产是否存在。
    """

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        dict.fromkeys(str(record["pdb_id"]).strip().lower() for record in records)
    )


def build_request_source(
    split_file: str | Path,
    mode: str,
    box_pool_root: str | Path | None,
    seed: int,
    pdb_foreground_box_num: int,
    pdb_foreground_fraction_target: float,
    pdb_occurrence_foreground_box_cap: int,
) -> Stage1TrainingRequestSet | list[ResolvedStage1Crop]:
    """根据路径形态创建 V3 动态训练请求集或固定 validation 请求列表。

    输入参数：
        - split_file：str | Path；V3 ``train`` 目录或名为 ``validation_selection_pdb_centric.npz`` 的冻结选择文件。
        - mode：str；请求模式；目录来源只允许 ``train``，冻结 validation 文件不依赖该模式重新抽样。
        - box_pool_root：str | Path | None；validation selection 所属的 V3 pool 根目录；读取冻结文件时必须提供。
        - seed：int；传给 :class:`Stage1TrainingRequestSet` 的训练抽样种子。
        - pdb_foreground_box_num：int；每个 PDB 的目标 bias BOX 数量。
        - pdb_foreground_fraction_target：float；bias BOX 占目标总 BOX 数量的比例。
        - pdb_occurrence_foreground_box_cap：int；单个 occurrence 每个 epoch 的 bias BOX 数量上限。

    返回值：
        - source：Stage1TrainingRequestSet | list[ResolvedStage1Crop]；目录来源返回按 epoch 更新的训练请求集，冻结文件来源返回固定请求列表。

    路径分支：
        - 仅接受 V3 train 目录和 ``validation_selection_pdb_centric.npz``；其他路径不执行旧版 split、fraction 或隐式搜索回退。
    """

    # Path；训练 pool 目录或当前活动 validation selection 文件。
    source = Path(split_file)
    if source.is_dir():
        if str(mode).lower() != "train":
            raise ValueError("只有 train 模式允许直接读取 BOX pool 目录。")
        return Stage1TrainingRequestSet(
            source,
            seed=seed,
            pdb_foreground_box_num=pdb_foreground_box_num,
            pdb_foreground_fraction_target=pdb_foreground_fraction_target,
            pdb_occurrence_foreground_box_cap=pdb_occurrence_foreground_box_cap,
        )
    if source.name == "validation_selection_pdb_centric.npz" and box_pool_root is not None:
        return load_validation_selection(source, box_pool_root)
    raise ValueError(
        "Stage1 请求来源必须是 V3 train 目录或 validation_selection_pdb_centric.npz。"
    )
