"""冻结 Selector 输入清单，并把已发布的 CLG 居中特征物化为模型样本。

主要入口:
    - `freeze_input_clg_list`: 扫描一次可用 PDB/CLG，或严格复用既有冻结清单。
    - `SelectorDataset`: 按冻结清单读取组件森林、居中特征和上游密度。

一个 Dataset 元素对应一个组件谱系组（CLG），而不是一个 PDB。组件、候选和实体表
均保留各自的变长 offsets；离散网格索引统一使用 ZYX，连续世界坐标统一使用 XYZ，
长度单位为 Å。训练监督由冻结的 candidate-occurrence 交集现场计算，不另存标签副本。
"""

from __future__ import annotations

import json
import math
import os
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.density_channel_builder import DensityChannelConfig, build_density_channels
from src.stage1_producers import FIND_MODEL_NAMES, STAGE1_MODEL_NAMES

from .structured.antichain_dp import CandidateTreeClosure, build_candidate_tree_closure
from .structured.oracle import build_online_oracle, compute_candidate_max_iou


@dataclass(frozen=True, order=True)
class SelectorInputRecord:
    """
    表示冻结在一次 Selector 运行中的一个组件谱系组样本身份。

    输入参数:
        - split: str, Stage1 数据划分名
        - pdb_id: str, PDB 标识
        - CLG_id: int, 当前 PDB 的 `clg.npz` 内连续局部编号
    """

    split: str
    pdb_id: str
    CLG_id: int


def _pdb_output_root(
    stage1_outputs_root: Path,
    stage1_model_name: str,
    split: str,
    pdb_id: str,
) -> Path:
    """
    返回固定模型来源、数据划分和 PDB 对应的输出目录。

    输入参数:
        - stage1_outputs_root: Path, `stage1_outputs/` 根目录
        - stage1_model_name: str, `STAGE1_MODEL_NAMES` 中的模型来源身份
        - split: str, `train/validation/calibration` 等数据划分
        - pdb_id: str, PDB 标识

    输出:
        - path: Path, 当前 PDB 正式输出目录
    """
    return stage1_outputs_root / stage1_model_name / split / pdb_id


def _is_complete_clg_centered(pdb_root: Path) -> bool:
    """
    判断 PDB 的 CLG 居中特征角色是否完整且可被下游读取。

    输入参数:
        - pdb_root: Path, 固定 producer/split/PDB 输出目录

    输出:
        - is_complete: bool，只有不存在 `_RUNNING` 和 `_BLOB_EXCEED`，并且
          `CLG_centered.npz`、`clg.npz`、`forest.npz` 及角色 `_COMPLETE`
          同时存在时才为 True
    """
    return (
        not (pdb_root / "_RUNNING").exists()
        and not (pdb_root / "_BLOB_EXCEED").exists()
        and (pdb_root / "status" / "CLG_centered" / "_COMPLETE").exists()
        and (pdb_root / "centered" / "CLG_centered.npz").is_file()
        and (pdb_root / "components" / "clg.npz").is_file()
        and (pdb_root / "components" / "forest.npz").is_file()
    )


def _read_expected_pdb_ids(path: Path) -> tuple[str, ...]:
    """
    读取正式 run 用于验证 validation 完整性的固定 PDB 清单。

    输入参数:
        - path: Path, JSON 字符串列表、含 `pdb_id` 的 JSON 对象列表，或每行一个
          PDB 标识的文本文件

    输出:
        - pdb_ids: tuple[str, ...]，去重后保持来源顺序的 PDB 标识
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        values = payload["pdb_ids"] if isinstance(payload, dict) else payload
        raw_ids = [value["pdb_id"] if isinstance(value, dict) else value for value in values]
    else:
        raw_ids = [line.strip() for line in text.splitlines() if line.strip()]
    result: list[str] = []
    seen: set[str] = set()
    for value in raw_ids:
        pdb_id = str(value)
        if pdb_id not in seen:
            result.append(pdb_id)
            seen.add(pdb_id)
    return tuple(result)


def _validate_frozen_records(
    records: Sequence[SelectorInputRecord],
    stage1_outputs_root: Path,
    stage1_model_name: str,
    pdb_ids_by_split: Mapping[str, Sequence[str]],
) -> None:
    """
    逐条验证冻结清单中的 CLG 仍属于同一 producer 且完整存在。

    输入参数:
        - records: Sequence[SelectorInputRecord], 待验证清单
        - stage1_outputs_root: Path, `stage1_outputs/` 根目录
        - stage1_model_name: str, 清单声明的唯一 Stage1 模型来源
        - pdb_ids_by_split: Mapping[str, Sequence[str]]，允许保留零 CLG PDB 的固定
          数据划分到 PDB 清单映射

    异常:
        - 任一产物缺失、身份重复、PDB 清单漂移或来源 CLG 不存在时立即抛出异常
    """
    identities = [(record.split, record.pdb_id, record.CLG_id) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("input_CLG_list.json 含重复 (split,pdb_id,CLG_id)。")
    expected_records: list[SelectorInputRecord] = []
    inventory_identities: set[tuple[str, str]] = set()
    for split, pdb_ids in pdb_ids_by_split.items():
        for pdb_id in pdb_ids:
            key = (str(split), str(pdb_id))
            if key in inventory_identities:
                raise ValueError(f"input_CLG_list.json 含重复 PDB identity: {key}")
            inventory_identities.add(key)
            pdb_root = _pdb_output_root(
                stage1_outputs_root, stage1_model_name, key[0], key[1]
            )
            if not _is_complete_clg_centered(pdb_root):
                raise FileNotFoundError(f"冻结 PDB 的 CLG_centered 未完整发布: {key}")
            with np.load(pdb_root / "components" / "clg.npz", allow_pickle=False) as source:
                clg_ids = np.asarray(source["CLG_id"], dtype=np.int64)
            if np.unique(clg_ids).size != clg_ids.size:
                raise ValueError(f"来源 clg.npz 含重复 CLG_id: {key}")
            expected_records.extend(
                SelectorInputRecord(split=key[0], pdb_id=key[1], CLG_id=int(clg_id))
                for clg_id in clg_ids.tolist()
            )
    if tuple(expected_records) != tuple(records):
        raise ValueError(
            "input_CLG_list.json.items 必须逐 PDB 完整复制来源 CLG 顺序；"
            "零 CLG PDB 只保留在 pdb_ids_by_split。"
        )
    for record in records:
        key = (record.split, record.pdb_id)
        if key not in inventory_identities:
            raise ValueError(f"冻结 CLG 不属于 pdb_ids_by_split: {record}")
        pdb_root = _pdb_output_root(stage1_outputs_root, stage1_model_name, *key)
        if not _is_complete_clg_centered(pdb_root):
            raise FileNotFoundError(f"冻结 CLG 样本未完整发布: {record}")


def _pdb_inventory_from_frozen_payload(
    payload: Mapping[str, Any],
    split_order: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """
    解析能够表示零 CLG PDB 的固定“数据划分→PDB 清单”。

    输入参数:
        - payload: Mapping[str, Any], `input_CLG_list.json` 根对象
        - split_order: Sequence[str], 配置冻结的数据划分顺序

    输出:
        - inventory: dict[str, tuple[str, ...]], 每个数据划分按来源顺序排列且
          不重复的 PDB 标识；零 CLG 的 PDB 仍保留在此映射中
    """
    source = payload.get("pdb_ids_by_split")
    if not isinstance(source, Mapping) or set(source) != set(split_order):
        raise ValueError("input_CLG_list.json.pdb_ids_by_split 必须覆盖全部 split_order。")
    result: dict[str, tuple[str, ...]] = {}
    for split in split_order:
        values = tuple(str(value) for value in source[split])
        if len(set(values)) != len(values):
            raise ValueError(f"pdb_ids_by_split[{split!r}] 含重复 PDB。")
        result[str(split)] = values
    expected_counts = {split: len(values) for split, values in result.items()}
    if payload.get("split_pdb_counts") != expected_counts:
        raise ValueError("input_CLG_list.json.split_pdb_counts 与 PDB inventory 不一致。")
    return result


def _records_from_frozen_payload(
    payload: dict[str, Any],
    stage1_model_name: str,
    split_order: Sequence[str],
) -> tuple[SelectorInputRecord, ...]:
    """
    严格解析既有 `input_CLG_list.json`，并验证模型来源与数据划分顺序。

    输入参数:
        - payload: dict[str, Any], 冻结清单 JSON 根对象
        - stage1_model_name: str, 当前 Selector 运行的 Stage1 模型来源
        - split_order: Sequence[str], 当前配置声明的固定数据划分顺序

    输出:
        - records: tuple[SelectorInputRecord, ...], 保持 JSON `items` 的原始顺序
    """
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("input_CLG_list.json.schema_version 必须为 1。")
    if payload.get("stage1_model_name") != stage1_model_name:
        raise ValueError("指定 input_CLG_list.json 来自不同 producer。")
    if tuple(str(value) for value in payload.get("split_order", ())) != tuple(split_order):
        raise ValueError("input_CLG_list.json 的 split_order 与当前 run 配置不一致。")
    records = tuple(
        SelectorInputRecord(
            split=str(item["split"]),
            pdb_id=str(item["pdb_id"]),
            CLG_id=int(item["CLG_id"]),
        )
        for item in payload["items"]
    )
    if any(record.split not in split_order for record in records):
        raise ValueError("input_CLG_list.json.items 含 split_order 之外的 split。")
    expected_counts = {
        str(split): sum(record.split == str(split) for record in records)
        for split in split_order
    }
    if payload.get("split_counts") != expected_counts:
        raise ValueError("input_CLG_list.json.split_counts 与 CLG items 不一致。")
    return records


def freeze_input_clg_list(
    stage1_outputs_root: str | Path,
    selector_run_dir: str | Path,
    stage1_model_name: str,
    split_order: Sequence[str],
    input_clg_list_path: str | Path | None,
    formal_run: bool,
    expected_validation_pdb_ids_path: str | Path | None,
) -> Path:
    """
    在 Selector 运行启动前扫描一次或严格复用既有 `input_CLG_list.json`。

    输入参数:
        - stage1_outputs_root: str | Path, `stage1_outputs/` 根目录
        - selector_run_dir: str | Path, 当前独立 Selector 运行输出目录
        - stage1_model_name: str, `STAGE1_MODEL_NAMES` 中的模型来源身份；不同模型
          来源不得混用
        - split_order: Sequence[str], 扫描与清单排序所用的固定数据划分顺序
        - input_clg_list_path: str | Path | None, 显式复用清单；None 表示本次启动扫描
        - formal_run: bool, 是否允许该运行产生正式 `BEST.ckpt`
        - expected_validation_pdb_ids_path: str | Path | None, 正式 run 的固定 validation PDB 清单

    输出:
        - frozen_path: Path, 当前 run 目录内的 `input_CLG_list.json`

    落盘与幂等性:
        - 首次扫描时按 `split_order`、PDB 名和来源 `CLG_id` 顺序原子写入冻结清单。
        - 同一运行目录已有清单时只允许内容完全一致的复用，不会吸收后来新增的
          Stage1 产物。
        - `formal_run=True` 时，固定 validation PDB 清单必须全部已完整发布；
          零 CLG PDB 仍由 `pdb_ids_by_split` 表示。
    """
    if stage1_model_name not in STAGE1_MODEL_NAMES:
        raise ValueError(f"stage1_model_name 必须属于 {STAGE1_MODEL_NAMES}。")
    outputs_root = Path(stage1_outputs_root)
    run_dir = Path(selector_run_dir)
    frozen_path = run_dir / "input_CLG_list.json"
    split_values = tuple(str(value) for value in split_order)
    if not split_values or len(set(split_values)) != len(split_values):
        raise ValueError("split_order 必须非空且无重复。")

    existing_payload: dict[str, Any] | None = None
    if frozen_path.is_file():
        existing_payload = json.loads(frozen_path.read_text(encoding="utf-8"))

    if input_clg_list_path is not None or existing_payload is not None:
        source_path = Path(input_clg_list_path) if input_clg_list_path is not None else frozen_path
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        records = _records_from_frozen_payload(
            payload,
            stage1_model_name=stage1_model_name,
            split_order=split_values,
        )
        pdb_ids_by_split = _pdb_inventory_from_frozen_payload(payload, split_values)
    else:
        records_list: list[SelectorInputRecord] = []
        pdb_ids_by_split_lists: dict[str, list[str]] = {
            split: [] for split in split_values
        }
        for split in split_values:
            split_root = outputs_root / stage1_model_name / split
            if not split_root.is_dir():
                continue
            pdb_roots = sorted(
                (path for path in split_root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
            )
            for pdb_root in pdb_roots:
                if not _is_complete_clg_centered(pdb_root):
                    continue
                pdb_ids_by_split_lists[split].append(pdb_root.name)
                with np.load(pdb_root / "components" / "clg.npz", allow_pickle=False) as source:
                    clg_ids = np.asarray(source["CLG_id"], dtype=np.int64)
                records_list.extend(
                    SelectorInputRecord(split=split, pdb_id=pdb_root.name, CLG_id=int(clg_id))
                    for clg_id in clg_ids.tolist()
                )
        records = tuple(records_list)
        pdb_ids_by_split = {
            split: tuple(values) for split, values in pdb_ids_by_split_lists.items()
        }

    _validate_frozen_records(
        records, outputs_root, stage1_model_name, pdb_ids_by_split
    )
    if formal_run:
        if expected_validation_pdb_ids_path is None:
            raise ValueError("formal_run=True 时必须显式提供固定 validation PDB 清单。")
        expected = set(_read_expected_pdb_ids(Path(expected_validation_pdb_ids_path)))
        available = set(pdb_ids_by_split.get("validation", ()))
        missing = sorted(expected - available)
        if missing:
            raise FileNotFoundError(f"正式 Selector run 的 validation 尚未全部可读: missing={missing[:20]}")

    split_counts = {
        split: sum(record.split == split for record in records)
        for split in split_values
    }
    payload = {
        "schema_version": 1,
        "stage1_model_name": stage1_model_name,
        "split_order": list(split_values),
        "split_counts": split_counts,
        "split_pdb_counts": {
            split: len(pdb_ids_by_split[split]) for split in split_values
        },
        "pdb_ids_by_split": {
            split: list(pdb_ids_by_split[split]) for split in split_values
        },
        "items": [asdict(record) for record in records],
    }
    if existing_payload is not None:
        existing_records = _records_from_frozen_payload(
            existing_payload,
            stage1_model_name=stage1_model_name,
            split_order=split_values,
        )
        if existing_records != records:
            raise ValueError("同一 selector_run_dir 的 input_CLG_list.json 已冻结，禁止替换。")
        existing_pdb_ids = _pdb_inventory_from_frozen_payload(
            existing_payload, split_values
        )
        if existing_pdb_ids != pdb_ids_by_split:
            raise ValueError("同一 selector_run_dir 的 PDB inventory 已冻结，禁止替换。")
        return frozen_path

    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = frozen_path.with_name(f".{frozen_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, frozen_path)
    return frozen_path


def _slice_offsets(offsets: np.ndarray, index: int) -> tuple[int, int]:
    """
    返回变长 offsets 中第 `index` 段的半开区间。

    输入参数:
        - offsets: np.ndarray, `(N + 1,)`，单调 int64 offsets
        - index: int, 目标条目或候选的局部编号

    输出:
        - bounds: tuple[int, int]，对应值表的 `[begin, end)` 切片边界
    """
    return int(offsets[index]), int(offsets[index + 1])


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    """
    以 allow_pickle=False 读取 NPZ 并复制全部数组，及时关闭文件句柄。

    输入参数:
        - path: Path, 盘上 NPZ 路径

    输出:
        - arrays: dict[str, np.ndarray]，字段到独立内存数组的映射；函数返回前
          已关闭 NPZ 文件句柄
    """
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]).copy() for name in source.files}


def _node_rows_by_id(forest: dict[str, np.ndarray], tree_id: int) -> dict[int, int]:
    """
    构造当前组件树的 `node_id` 到森林节点表绝对行号的映射。

    输入参数:
        - forest: dict[str, np.ndarray]，`forest.npz` 的全部字段
        - tree_id: int, 当前组件谱系组所属组件树编号

    输出:
        - rows: dict[int, int]，树内节点编号到森林节点表绝对行号的映射
    """
    selected_rows = np.flatnonzero(np.asarray(forest["tree_id"], dtype=np.int64) == int(tree_id))
    return {int(forest["node_id"][row]): int(row) for row in selected_rows.tolist()}


def _tree_depth(parent_by_id: dict[int, int], node_id: int) -> int:
    """
    沿直接父节点链计算一个森林节点相对树根的深度。

    输入参数:
        - parent_by_id: dict[int, int]，当前树的节点编号到直接父节点编号映射
        - node_id: int, 目标节点编号

    输出:
        - depth: int, root=0 的树深度
    """
    depth = 0
    current = int(node_id)
    seen: set[int] = set()
    while parent_by_id[current] != -1:
        if current in seen:
            raise ValueError("forest parent 链存在环。")
        seen.add(current)
        current = parent_by_id[current]
        depth += 1
    return depth


def _lca_node(parent_by_id: dict[int, int], depth_by_id: dict[int, int], left: int, right: int) -> int:
    """
    在一棵组件树中求两个节点的最低共同祖先。

    输入参数:
        - parent_by_id: dict[int, int]，节点编号到直接父节点编号映射
        - depth_by_id: dict[int, int]，节点编号到相对树根深度的映射
        - left: int, 第一个节点编号
        - right: int, 第二个节点编号

    输出:
        - lca: int, 最低共同祖先节点编号
    """
    a, b = int(left), int(right)
    while depth_by_id[a] > depth_by_id[b]:
        a = parent_by_id[a]
    while depth_by_id[b] > depth_by_id[a]:
        b = parent_by_id[b]
    while a != b:
        a = parent_by_id[a]
        b = parent_by_id[b]
    return a


class SelectorDataset(Dataset[dict[str, Any]]):
    """
    把冻结清单中的每个组件谱系组物化为 CCLN 所需的一个样本。

    输入参数:
        - input_clg_list_path: str | Path, 当前 Selector run 的冻结清单
        - stage1_outputs_root: str | Path, Stage1 producer 正式输出根目录
        - upstream_root: str | Path, 提供 `density/{pdb_id}/exp.npz` 的 AdaLigand Stage A 至 Stage G 数据根目录
        - split: str, 当前 Dataset 读取的唯一数据划分
        - lambda_count: float, 在线最优监督中每选择一个候选所扣除的计数惩罚
        - require_oracle: bool, 是否强制读取 overlap 并现场计算 q/S*/y_G
        - density_clip_percentile: tuple[float,float], exp_clipnorm_nopost 的 clip 分位数
        - pdb_cache_size: int, 每个 DataLoader 进程最多缓存的完整 PDB 数据包数量；
          应按实际内存测量结果调整，0 表示禁用缓存

    单样本输出:
        - 身份与几何: `split/pdb_id/CLG_id/candidate_node_id` 以及 BOX 几何；
          `box_start_zyx/box_shape_zyx` 是离散 voxel ZYX，`*_world` 是连续 XYZ Å。
        - 候选结构: `candidate_attributes` 为 `(N_c, 16)`，
          `tree_relative_feature` 为 `(N_c, N_c, 8)`，并附最小连接闭包。
        - V 模态: 稀疏 `voxel_final`、概率、世界坐标、候选成员关系和现场构造的
          `density_input`；不读取 Stage1 骨干内部的固定多尺度 V 网格。
        - Find 模型的 P/A 模态: 从 `CLG_centered.npz` 直接读取多层来源、概率、
          世界坐标和候选到 A 实体的成员关系；`A_feat_L0` 也是归档字段。
        - 在线监督: `require_oracle=True` 时增加每个候选的最大 IoU、最优反链
          局部候选下标和 CLG 是否具有正分最优非空解。
    """

    def __init__(
        self,
        input_clg_list_path: str | Path,
        stage1_outputs_root: str | Path,
        upstream_root: str | Path,
        split: str,
        lambda_count: float,
        require_oracle: bool,
        density_clip_percentile: tuple[float, float],
        pdb_cache_size: int,
    ) -> None:
        """
        读取冻结清单并初始化当前数据划分的只读样本视图和 PDB 缓存。

        输入参数:
            - input_clg_list_path: str | Path, 当前 Selector 运行的
              `input_CLG_list.json`
            - stage1_outputs_root: str | Path, 已发布 Stage1 产物根目录
            - upstream_root: str | Path, experimental density 根目录；P/A 特征不从该目录读取
            - split: str, 本实例唯一读取的数据划分
            - lambda_count: float, 在线最优反链的候选计数惩罚
            - require_oracle: bool, 是否读取 `overlap.npz` 并现场生成训练监督
            - density_clip_percentile: tuple[float, float], experimental density 的
              下、上裁剪分位数
            - pdb_cache_size: int, 当前 DataLoader 进程的最近使用 PDB 数据包容量；
              0 表示每次直接读取

        副作用:
            - 初始化阶段只读取和验证冻结清单，不写入项目产物。
            - PDB 数据与 NPZ 数组延迟到 `_load_pdb_bundle` 或 `__getitem__` 时读取。
        """
        super().__init__()
        payload = json.loads(Path(input_clg_list_path).read_text(encoding="utf-8"))
        self.stage1_model_name = str(payload["stage1_model_name"])
        split_order = tuple(str(value) for value in payload["split_order"])
        pdb_inventory = _pdb_inventory_from_frozen_payload(payload, split_order)
        self.records = tuple(
            SelectorInputRecord(str(item["split"]), str(item["pdb_id"]), int(item["CLG_id"]))
            for item in payload["items"]
            if str(item["split"]) == str(split)
        )
        self.stage1_outputs_root = Path(stage1_outputs_root)
        self.upstream_root = Path(upstream_root)
        self.split = str(split)
        self.lambda_count = float(lambda_count)
        self.require_oracle = bool(require_oracle)
        if int(pdb_cache_size) < 0:
            raise ValueError("pdb_cache_size 不能为负。")
        self.pdb_cache_size = int(pdb_cache_size)
        # 最近使用缓存的键为 `(split, pdb_id)`；每个 DataLoader 进程持有独立实例。
        self._pdb_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self.density_config = DensityChannelConfig(
            clip_percentile=(float(density_clip_percentile[0]), float(density_clip_percentile[1])),
            fit_mask_percentile=0.003,
            enabled_channels=["exp_clipnorm_nopost"],
        )
        _validate_frozen_records(
            self.records,
            self.stage1_outputs_root,
            self.stage1_model_name,
            {self.split: pdb_inventory[self.split]},
        )

    def _load_pdb_bundle(self, record: SelectorInputRecord) -> dict[str, Any]:
        """
        读取并按显式容量缓存一个 PDB 的居中、森林、CLG 和概率基础表。

        输入参数:
            - record: SelectorInputRecord, 当前冻结 CLG identity

        输出:
            - bundle: dict[str, Any]，包含 `centered/forest/clg/probability`、
              experimental density，以及训练时可选的 occurrence 交集；
              probability 为完整图离散 ZYX 网格，Find 的 A/P 特征已经位于 centered 归档
        """
        key = (record.split, record.pdb_id)
        if key in self._pdb_cache:
            bundle = self._pdb_cache.pop(key)
            self._pdb_cache[key] = bundle
            return bundle
        pdb_root = _pdb_output_root(
            self.stage1_outputs_root,
            self.stage1_model_name,
            record.split,
            record.pdb_id,
        )
        bundle: dict[str, Any] = {
            "centered": _load_npz_arrays(pdb_root / "centered" / "CLG_centered.npz"),
            "forest": _load_npz_arrays(pdb_root / "components" / "forest.npz"),
            "clg": _load_npz_arrays(pdb_root / "components" / "clg.npz"),
            "probability": _load_npz_arrays(pdb_root / "probability" / "probability_map.npz")["probability_map"],
        }
        with np.load(self.upstream_root / "density" / record.pdb_id / "exp.npz", allow_pickle=False) as source:
            bundle["experimental_density"] = np.asarray(source["grid"], dtype=np.float32).copy()
        if self.require_oracle:
            bundle["overlap"] = _load_npz_arrays(pdb_root / "components" / "overlap.npz")
        if self.pdb_cache_size > 0:
            self._pdb_cache[key] = bundle
            while len(self._pdb_cache) > self.pdb_cache_size:
                self._pdb_cache.popitem(last=False)
        return bundle

    def __len__(self) -> int:
        """
        返回当前 split 的冻结 CLG 数量。

        输出:
            - count: int, Dataset 样本数；运行中新增 Stage1 产物不会改变该值
        """
        return len(self.records)

    @staticmethod
    def collate_fn(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        保留每个 CLG 自己的变长实体表和树结构，以列表作为小批次。

        输入参数:
            - samples: list[dict[str, Any]], 长度 B，每项是一个完整组件谱系组

        输出:
            - batch: list[dict[str, Any]], 长度 B；包装器逐 CLG 前向并在损失层
              对 B 个 CLG 等权平均
        """
        return samples

    def source_dimensions(self) -> dict[str, dict[str, int]]:
        """
        从首个冻结样本解析实际 A/P 特征来源通道数，供模型配置冻结。

        输出:
            - dimensions: dict[str, dict[str, int]]，包含 `A/P` 到“字段名→末维
              通道数”的映射；unet_c1 的两组映射均为空
        """
        if not self.records:
            raise ValueError("空 SelectorDataset 无法解析 source dimensions。")
        sample = self[0]
        return {
            "A": {name: int(value.shape[-1]) for name, value in sample.get("A_sources", {}).items()},
            "P": {name: int(value.shape[-1]) for name, value in sample.get("P_sources", {}).items()},
        }

    def _load_density_crop(
        self,
        full_grid: np.ndarray,
        box_start_zyx: np.ndarray,
        box_shape_zyx: np.ndarray,
    ) -> np.ndarray:
        """
        从上游原始 experimental density 现场裁剪 BOX 并构造单通道标准化密度。

        输入参数:
            - full_grid: np.ndarray, `(1, D_full, H_full, W_full)`，完整图离散 ZYX
              voxel 网格上的原始 experimental density
            - box_start_zyx: np.ndarray, `(3,)`，完整图离散 voxel-index ZYX 的
              合法 BOX 起点
            - box_shape_zyx: np.ndarray, `(3,)`，BOX 离散 voxel 网格尺寸 ZYX

        输出:
            - density_input: np.ndarray, `(1, D, H, W)`，float32，
              `exp_clipnorm_nopost` 单通道密度
        """
        z, y, x = (int(value) for value in box_start_zyx)
        dz, dy, dx = (int(value) for value in box_shape_zyx)
        raw_crop = full_grid[0, z : z + dz, y : y + dy, x : x + dx]
        if raw_crop.shape != (dz, dy, dx):
            raise ValueError("上游 experimental density 无法提供真实 BOX crop。")
        return build_density_channels(raw_crop, None, self.density_config, None)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        物化一个冻结 CLG，并现场生成候选属性、候选对特征和可选最优监督。

        输入参数:
            - index: int, 当前数据划分过滤后的冻结样本局部下标

        输出:
            - sample: dict[str, Any]，字段集合见类 Docstring；`*_index/*_zyx`
              为离散 voxel 或变长值表局部下标，`*_world` 为连续世界 XYZ，
              长度单位为 Å，数值数组均转为 torch.Tensor
        """
        record = self.records[index]
        bundle = self._load_pdb_bundle(record)
        centered = bundle["centered"]
        forest = bundle["forest"]
        clg = bundle["clg"]
        probability = bundle["probability"]

        # int64，(1,)，当前 CLG 在聚合居中条目表中的唯一绝对行号。
        entry_rows = np.flatnonzero(np.asarray(centered["CLG_id"], dtype=np.int64) == record.CLG_id)
        # int64，(1,)，同一 CLG 在来源 `clg.npz` 主表中的唯一绝对行号。
        source_rows = np.flatnonzero(np.asarray(clg["CLG_id"], dtype=np.int64) == record.CLG_id)
        if entry_rows.size != 1 or source_rows.size != 1:
            raise ValueError(f"CLG_id 必须在 centered/source 中各出现一次: {record}")
        entry = int(entry_rows[0])
        source_clg_row = int(source_rows[0])

        # int64，(3,)，完整图离散 ZYX voxel-index 中的 BOX 起点。
        box_start_zyx = np.asarray(centered["box_start_zyx"][entry], dtype=np.int64)
        # int64，(3,)，BOX 的离散 ZYX voxel 网格尺寸。
        box_shape_zyx = np.asarray(centered["box_shape_zyx"][entry], dtype=np.int64)
        # float32，(3,)，BOX 起点和单 voxel 尺寸的连续世界 XYZ 表示，单位 Å。
        box_origin_world = np.asarray(centered["box_origin_world"][entry], dtype=np.float32)
        voxel_size_world = np.asarray(centered["voxel_size_world"][entry], dtype=np.float32)
        box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
        box_center_world = box_origin_world + 0.5 * box_shape_xyz * voxel_size_world
        box_half_size_world = 0.5 * box_shape_xyz * voxel_size_world
        full_origin_world = box_origin_world - box_start_zyx[[2, 1, 0]] * voxel_size_world

        # int64，(N_entry + 1,)，按居中条目切分候选值表。
        candidate_offsets = np.asarray(centered["candidate_offsets"], dtype=np.int64)
        candidate_begin, candidate_end = _slice_offsets(candidate_offsets, entry)
        candidate_node_id = np.asarray(centered["candidate_node_id"][candidate_begin:candidate_end], dtype=np.int64)
        candidate_threshold_grid_index = np.asarray(
            centered["candidate_threshold_grid_index"][candidate_begin:candidate_end], dtype=np.int64
        )
        source_begin, source_end = _slice_offsets(np.asarray(clg["candidate_offsets"], dtype=np.int64), source_clg_row)
        if not np.array_equal(candidate_node_id, np.asarray(clg["candidate_node_id"][source_begin:source_end])):
            raise ValueError("CLG_centered candidate 顺序与来源 clg.npz 不一致。")

        tree_id = int(centered["source_tree_id"][entry])
        node_row_by_id = _node_rows_by_id(forest, tree_id)
        tree_node_ids = tuple(node_row_by_id)
        tree_parent_ids = tuple(int(forest["parent_node_id"][node_row_by_id[value]]) for value in tree_node_ids)
        # CandidateTreeClosure，包含候选到最低共同祖先的最小连接子树；父节点先于子节点。
        closure: CandidateTreeClosure = build_candidate_tree_closure(
            node_id=tree_node_ids,
            parent_node_id=tree_parent_ids,
            candidate_node_id=candidate_node_id.tolist(),
        )
        parent_by_id = dict(zip(tree_node_ids, tree_parent_ids, strict=True))
        depth_by_id = {value: _tree_depth(parent_by_id, value) for value in tree_node_ids}

        full_shape_zyx = probability.shape
        # float32，(D_full*H_full*W_full,)，按 ZYX 网格 C-order 展平的完整图概率。
        probability_flat = np.asarray(probability, dtype=np.float32).reshape(-1)
        candidate_count = candidate_node_id.size
        # float32，(N_candidate, 16)，候选阈值、体积、概率、位置、形状和树深度属性。
        candidate_attributes = np.empty((candidate_count, 16), dtype=np.float32)
        candidate_centroid_world = np.empty((candidate_count, 3), dtype=np.float32)
        candidate_voxel_count = np.empty((candidate_count,), dtype=np.int64)
        oldest_node_id = int(centered["CLG_oldest_node_id"][entry])
        oldest_row = node_row_by_id[oldest_node_id]
        oldest_voxel_count = max(int(forest["voxel_count"][oldest_row]), 1)

        for candidate_index, node_identity in enumerate(candidate_node_id.tolist()):
            forest_row = node_row_by_id[int(node_identity)]
            voxel_begin, voxel_end = _slice_offsets(
                np.asarray(forest["node_voxel_offsets"], dtype=np.int64),
                forest_row,
            )
            global_linear = np.asarray(
                forest["node_voxel_global_linear_index"][voxel_begin:voxel_end], dtype=np.int64
            )
            coord_zyx = np.stack(np.unravel_index(global_linear, full_shape_zyx), axis=-1).astype(np.float32)
            mask_probability = probability_flat[global_linear]
            # float32，(5,)，候选 mask 内概率的均值、最大值和三个四分位数。
            probability_stats = np.asarray(
                [
                    mask_probability.mean(),
                    mask_probability.max(),
                    *np.quantile(mask_probability, [0.25, 0.5, 0.75]).tolist(),
                ],
                dtype=np.float32,
            )
            voxel_count = int(global_linear.size)
            candidate_voxel_count[candidate_index] = voxel_count
            centroid_zyx = coord_zyx.mean(axis=0)
            centroid_xyz = centroid_zyx[[2, 1, 0]]
            candidate_centroid_world[candidate_index] = full_origin_world + (centroid_xyz + 0.5) * voxel_size_world
            normalized_local_centroid = 2.0 * (
                centroid_zyx - box_start_zyx.astype(np.float32) + 0.5
            ) / box_shape_zyx.astype(np.float32) - 1.0
            normalized_coord = (coord_zyx - coord_zyx.mean(axis=0, keepdims=True)) / box_shape_zyx.astype(np.float32)
            if normalized_coord.shape[0] > 1:
                # 显式采用无偏样本协方差；单体素候选的三个特征值定义为零。
                covariance = normalized_coord.T @ normalized_coord / (normalized_coord.shape[0] - 1)
            else:
                covariance = np.zeros((3, 3), dtype=np.float32)
            covariance_eigenvalue = np.sort(np.linalg.eigvalsh(covariance)).astype(np.float32)
            candidate_attributes[candidate_index] = np.asarray(
                [
                    candidate_threshold_grid_index[candidate_index] / 32768.0,
                    float(forest["threshold_value"][forest_row]),
                    np.log1p(voxel_count),
                    voxel_count / oldest_voxel_count,
                    *probability_stats.tolist(),
                    *normalized_local_centroid[[2, 1, 0]].tolist(),
                    *covariance_eigenvalue.tolist(),
                    depth_by_id[int(node_identity)] - depth_by_id[oldest_node_id],
                ],
                dtype=np.float32,
            )

        # float32，(N_candidate, N_candidate, 8)，有序候选对的树距离、阈值差、
        # 连续世界相对 XYZ、相对距离和体积比。
        tree_relative_feature = np.empty((candidate_count, candidate_count, 8), dtype=np.float32)
        for left in range(candidate_count):
            for right in range(candidate_count):
                left_id = int(candidate_node_id[left])
                right_id = int(candidate_node_id[right])
                lca = _lca_node(parent_by_id, depth_by_id, left_id, right_id)
                relative_xyz = (
                    candidate_centroid_world[right] - candidate_centroid_world[left]
                ) / box_half_size_world
                tree_relative_feature[left, right] = np.asarray(
                    [
                        depth_by_id[left_id] - depth_by_id[lca],
                        depth_by_id[right_id] - depth_by_id[lca],
                        (
                            candidate_threshold_grid_index[right]
                            - candidate_threshold_grid_index[left]
                        )
                        / 32768.0,
                        *relative_xyz.tolist(),
                        math.sqrt(sum(float(value) ** 2 for value in relative_xyz)),
                        math.log(
                            (float(candidate_voxel_count[right]) + 1e-6)
                            / (float(candidate_voxel_count[left]) + 1e-6)
                        ),
                    ],
                    dtype=np.float32,
                )

        voxel_offsets = np.asarray(centered["voxel_offsets"], dtype=np.int64)
        voxel_begin, voxel_end = _slice_offsets(voxel_offsets, entry)
        voxel_index_local_zyx = np.asarray(
            centered["voxel_index_local_zyx"][voxel_begin:voxel_end], dtype=np.int64
        )
        # 连续世界坐标 XYZ；离散 BOX-local voxel-index ZYX 的 voxel-center 为 index+0.5。
        voxel_coord_world = box_origin_world + (
            voxel_index_local_zyx[:, [2, 1, 0]].astype(np.float32) + 0.5
        ) * voxel_size_world
        # 全局 offsets 先截取当前条目的候选段，再平移为指向本条目 V 值表的局部 offsets。
        candidate_voxel_offsets_all = np.asarray(centered["candidate_voxel_offsets"], dtype=np.int64)
        candidate_voxel_value_begin = int(candidate_voxel_offsets_all[candidate_begin])
        candidate_voxel_value_end = int(candidate_voxel_offsets_all[candidate_end])
        local_candidate_voxel_offsets = (
            candidate_voxel_offsets_all[candidate_begin : candidate_end + 1] - candidate_voxel_value_begin
        )
        local_candidate_voxel_index = np.asarray(
            centered["candidate_voxel_index"][candidate_voxel_value_begin:candidate_voxel_value_end], dtype=np.int64
        )
        if local_candidate_voxel_index.size and (
            local_candidate_voxel_index.min() < 0
            or local_candidate_voxel_index.max() >= voxel_index_local_zyx.shape[0]
        ):
            raise ValueError("candidate_voxel_index 越过所属 entry voxel 段。")

        # float32，(1, D, H, W)，与当前 BOX 对齐且不落盘缓存的密度输入。
        density_input = self._load_density_crop(bundle["experimental_density"], box_start_zyx, box_shape_zyx)
        sample: dict[str, Any] = {
            "split": record.split,
            "pdb_id": record.pdb_id,
            "CLG_id": record.CLG_id,
            "candidate_node_id": torch.from_numpy(candidate_node_id),
            "candidate_attributes": torch.from_numpy(candidate_attributes),
            "candidate_centroid_world": torch.from_numpy(candidate_centroid_world),
            "tree_relative_feature": torch.from_numpy(tree_relative_feature),
            "closure_parent_index": closure.parent_index,
            "closure_candidate_index_by_node": closure.candidate_index_by_node,
            "box_start_zyx": torch.from_numpy(box_start_zyx),
            "box_shape_zyx": torch.from_numpy(box_shape_zyx),
            "box_origin_world": torch.from_numpy(box_origin_world),
            "voxel_size_world": torch.from_numpy(voxel_size_world),
            "box_center_world": torch.from_numpy(box_center_world.astype(np.float32)),
            "box_half_size_world": torch.from_numpy(box_half_size_world.astype(np.float32)),
            "oldest_V_center_world": torch.from_numpy(voxel_coord_world.mean(axis=0).astype(np.float32)),
            "candidate_voxel_offsets": torch.from_numpy(local_candidate_voxel_offsets),
            "candidate_voxel_index": torch.from_numpy(local_candidate_voxel_index),
            "V_sources": {
                "voxel_final": torch.from_numpy(
                    np.asarray(centered["voxel_final"][voxel_begin:voxel_end], dtype=np.float32)
                )
            },
            "V_probability": torch.from_numpy(
                np.asarray(centered["centered_probability"][voxel_begin:voxel_end], dtype=np.float32)
            ),
            "V_coord_world": torch.from_numpy(voxel_coord_world.astype(np.float32)),
            "V_index_local_zyx": torch.from_numpy(voxel_index_local_zyx),
            "V_batch_index": torch.zeros((voxel_index_local_zyx.shape[0],), dtype=torch.long),
            "density_input": torch.from_numpy(density_input).unsqueeze(0),
        }

        if self.stage1_model_name in FIND_MODEL_NAMES:
            p_offsets = np.asarray(centered["P_offsets"], dtype=np.int64)
            p_begin, p_end = _slice_offsets(p_offsets, entry)
            # 连续 BOX-local voxel 坐标 XYZ，corner 语义；换算为世界坐标时不额外加 0.5。
            p_coord_local = np.asarray(centered["P_coord_local_xyz"][p_begin:p_end], dtype=np.float32)
            sample.update(
                {
                    "P_sources": {
                        name: torch.from_numpy(np.asarray(centered[name][p_begin:p_end], dtype=np.float32))
                        for name in ("P_feat_L2", "P_feat_L3")
                    },
                    "P_probability": torch.from_numpy(
                        np.asarray(centered["P_probability"][p_begin:p_end], dtype=np.float32)
                    ),
                    "P_coord_world": torch.from_numpy(
                        (box_origin_world + p_coord_local * voxel_size_world).astype(np.float32)
                    ),
                }
            )

            a_offsets = np.asarray(centered["A_offsets"], dtype=np.int64)
            a_begin, a_end = _slice_offsets(a_offsets, entry)
            a_global_index = np.asarray(centered["A_global_index"][a_begin:a_end], dtype=np.int64)
            # 连续 BOX-local voxel 坐标 XYZ，corner 语义；A 世界坐标由 BOX origin 与 voxel 间距换算。
            a_coord_local = np.asarray(centered["A_coord_local_xyz"][a_begin:a_end], dtype=np.float32)
            # 与 V 成员关系相同，把聚合归档的全局 offsets 平移到当前条目 A 值表。
            candidate_a_offsets_all = np.asarray(centered["candidate_A_offsets"], dtype=np.int64)
            candidate_a_value_begin = int(candidate_a_offsets_all[candidate_begin])
            candidate_a_value_end = int(candidate_a_offsets_all[candidate_end])
            local_candidate_a_offsets = (
                candidate_a_offsets_all[candidate_begin : candidate_end + 1] - candidate_a_value_begin
            )
            local_candidate_a_index = np.asarray(
                centered["candidate_A_index"][candidate_a_value_begin:candidate_a_value_end], dtype=np.int64
            )
            if local_candidate_a_index.size and (
                local_candidate_a_index.min() < 0 or local_candidate_a_index.max() >= a_global_index.size
            ):
                raise ValueError("candidate_A_index 越过所属 entry A 段。")
            a_coord_world = box_origin_world + a_coord_local * voxel_size_world
            sample.update(
                {
                    "A_sources": {
                        "A_feat_L0": torch.from_numpy(
                            np.asarray(centered["A_feat_L0"][a_begin:a_end], dtype=np.float32)
                        ),
                        **{
                            name: torch.from_numpy(np.asarray(centered[name][a_begin:a_end], dtype=np.float32))
                            for name in ("A_feat_L1", "A_feat_L2", "A_feat_L3")
                        },
                    },
                    "A_probability": torch.from_numpy(
                        np.asarray(centered["A_probability"][a_begin:a_end], dtype=np.float32)
                    ),
                    "A_coord_world": torch.from_numpy(a_coord_world.astype(np.float32)),
                    "oldest_A_center_world": torch.from_numpy(
                        (a_coord_world.mean(axis=0) if a_coord_world.shape[0] else box_center_world).astype(np.float32)
                    ),
                    "candidate_A_offsets": torch.from_numpy(local_candidate_a_offsets),
                    "candidate_A_index": torch.from_numpy(local_candidate_a_index),
                }
            )

        if self.require_oracle:
            overlap = bundle["overlap"]
            occurrence_offsets = np.asarray(overlap["candidate_occurrence_offsets"], dtype=np.int64)
            overlap_begin = int(occurrence_offsets[source_begin])
            overlap_end = int(occurrence_offsets[source_end])
            local_occurrence_offsets = occurrence_offsets[source_begin : source_end + 1] - overlap_begin
            # float32，(N_candidate,)，每个候选对任一真实 occurrence 的最大 IoU。
            q = compute_candidate_max_iou(
                candidate_occurrence_offsets=local_occurrence_offsets,
                overlap_occurrence_index=overlap["overlap_occurrence_index"][overlap_begin:overlap_end],
                intersection_voxel_count=overlap["intersection_voxel_count"][overlap_begin:overlap_end],
                candidate_voxel_count=candidate_voxel_count,
                occurrence_voxel_count=overlap["occurrence_voxel_count"],
            )
            oracle = build_online_oracle(
                candidate_max_iou=torch.from_numpy(q),
                parent_index=closure.parent_index,
                candidate_index_by_node=closure.candidate_index_by_node,
                lambda_count=self.lambda_count,
            )
            sample.update(
                {
                    "candidate_max_iou": oracle.candidate_max_iou,
                    "oracle_selected_candidate_index": oracle.selected_candidate_index,
                    "CLG_is_valid": oracle.is_valid.to(dtype=torch.float32),
                    "oracle_best_score": oracle.best_score,
                }
            )
        return sample
