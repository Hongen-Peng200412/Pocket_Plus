"""从已发布 CLG_centered 冻结并冷读 Selector 样本。"""

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

from .structured.antichain_dp import CandidateTreeClosure, build_candidate_tree_closure
from .structured.oracle import build_online_oracle, compute_candidate_max_iou


def recover_a_feat_l0(
    receptor_feat: np.ndarray,
    a_global_index: np.ndarray,
) -> np.ndarray:
    """按 centered ``A_global_index`` 从当前 PDB receptor 表恢复原始 49D。

    参数:
        receptor_feat: Stage C ``receptor_tokens.npz/feat``，形状 ``[N_receptor,49]``。
        a_global_index: centered archive 的 A 行身份，形状 ``[N_A]``；其行序就是
            当前 centered entry 的 A 行序。

    返回:
        ``float32[N_A,49]`` 独立数组。A_feat_L0 不在 centered NPZ 重复落盘，
        但该索引恢复必须逐行保持来源身份与顺序。
    """
    features = np.asarray(receptor_feat, dtype=np.float32)
    indices = np.asarray(a_global_index)
    if features.ndim != 2 or features.shape[1] != 49:
        raise ValueError("receptor_feat 必须为 [N_receptor,49]")
    if indices.dtype.kind not in "iu" or indices.ndim != 1:
        raise ValueError("A_global_index 必须为一维整数数组")
    indices = indices.astype(np.int64, copy=False)
    if indices.size and (
        int(indices.min()) < 0 or int(indices.max()) >= int(features.shape[0])
    ):
        raise ValueError("A_global_index 越过当前 PDB receptor_feat")
    return np.asarray(features[indices], dtype=np.float32).copy()


@dataclass(frozen=True, order=True)
class SelectorInputRecord:
    """
    表示冻结在 Selector run 中的一个 CLG 样本 identity。

    字段:
        - split: str, Stage1 split 名称
        - pdb_id: str, PDB identity
        - CLG_id: int, 当前 PDB 内来源 clg.npz 的连续本地 ID
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
    返回固定 producer/split/PDB 输出目录。

    输入参数:
        - stage1_outputs_root: Path, `stage1_outputs/` 根目录
        - stage1_model_name: str, Find_0/Find_1/unet_c1
        - split: str, train/validation/calibration 等 split
        - pdb_id: str, PDB identity

    输出:
        - path: Path, 当前 PDB 正式输出目录
    """
    return stage1_outputs_root / stage1_model_name / split / pdb_id


def _is_complete_clg_centered(pdb_root: Path) -> bool:
    """
    判断 PDB 的 CLG_centered role 是否完整且可被下游读取。

    输入参数:
        - pdb_root: Path, 固定 producer/split/PDB 输出目录

    输出:
        - is_complete: bool，只有无运行锁/超限终态且 payload 与 role marker 均存在时为 True
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
        - path: Path, JSON list、含 `pdb_id` 的 JSON records 或每行一个 PDB 的文本文件

    输出:
        - pdb_ids: tuple[str, ...]，去重后保持来源顺序的 PDB identity
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
        - stage1_model_name: str, 清单声明的唯一 producer

    输出:
        - None；任何缺失、重复、PDB inventory 漂移或来源 CLG 不存在都会 fail-fast
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
    """解析可表示零 CLG PDB 的固定 split→PDB inventory。"""
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
    严格解析既有 input_CLG_list.json，并验证 producer 与 split 顺序身份。

    输入参数:
        - payload: dict[str,Any], 冻结清单 JSON 根对象
        - stage1_model_name: str, 当前 Selector run 的 producer
        - split_order: Sequence[str], 当前配置声明的固定 split 顺序

    输出:
        - records: tuple[SelectorInputRecord,...], 保持清单原始顺序
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
    在 Selector run 启动前扫描一次或严格复用既有 `input_CLG_list.json`。

    输入参数:
        - stage1_outputs_root: str | Path, `stage1_outputs/` 根目录
        - selector_run_dir: str | Path, 当前独立 Selector run 输出目录
        - stage1_model_name: str, Find_0/Find_1/unet_c1；不同 producer 不得混用
        - split_order: Sequence[str], 扫描与清单排序所用的固定 split 顺序
        - input_clg_list_path: str | Path | None, 显式复用清单；None 表示本次启动扫描
        - formal_run: bool, 是否允许该 run 产生正式 BEST
        - expected_validation_pdb_ids_path: str | Path | None, 正式 run 的固定 validation PDB 清单

    输出:
        - frozen_path: Path, 当前 run 目录内的 `input_CLG_list.json`
    """
    if stage1_model_name not in {"Find_0", "Find_1", "unet_c1"}:
        raise ValueError("stage1_model_name 只允许 Find_0、Find_1、unet_c1。")
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
    返回 ragged offsets 第 index 段的半开区间。

    输入参数:
        - offsets: np.ndarray, (N+1,), 单调 int64 offsets
        - index: int, 目标 entry/candidate 行

    输出:
        - begin/end: tuple[int,int]，value 表的半开切片边界
    """
    return int(offsets[index]), int(offsets[index + 1])


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    """
    以 allow_pickle=False 读取 NPZ 并复制全部数组，及时关闭文件句柄。

    输入参数:
        - path: Path, 盘上 NPZ 路径

    输出:
        - arrays: dict[str,np.ndarray]，字段到独立内存数组的映射
    """
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]).copy() for name in source.files}


def _node_rows_by_id(forest: dict[str, np.ndarray], tree_id: int) -> dict[int, int]:
    """
    构造当前 tree 的 node_id 到 forest 全局行映射。

    输入参数:
        - forest: dict[str,np.ndarray]，forest.npz 全部字段
        - tree_id: int, 当前 CLG 所属 tree identity

    输出:
        - rows: dict[int,int]，tree 内 node identity 到 forest 行
    """
    selected_rows = np.flatnonzero(np.asarray(forest["tree_id"], dtype=np.int64) == int(tree_id))
    return {int(forest["node_id"][row]): int(row) for row in selected_rows.tolist()}


def _tree_depth(parent_by_id: dict[int, int], node_id: int) -> int:
    """
    沿 direct parent 计算一个 forest node 的 root depth。

    输入参数:
        - parent_by_id: dict[int,int]，当前 tree 的 node→direct parent
        - node_id: int, 目标 node identity

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
    在一棵 component tree 中求两个 node 的最低共同祖先。

    输入参数:
        - parent_by_id: dict[int,int]，node→direct parent
        - depth_by_id: dict[int,int]，node→root depth
        - left/right: int, 两个 node identity

    输出:
        - lca: int, 最低共同祖先 node identity
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
    把冻结清单中的每个 CLG 物化为 CCLN 所需的一个样本。

    输入参数:
        - input_clg_list_path: str | Path, 当前 Selector run 的冻结清单
        - stage1_outputs_root: str | Path, Stage1 producer 正式输出根目录
        - upstream_root: str | Path, AdaLigand A–G 数据根目录
        - split: str, 当前 Dataset 读取的单一 split
        - lambda_count: float, online oracle 计数惩罚
        - require_oracle: bool, 是否强制读取 overlap 并现场计算 q/S*/y_G
        - density_clip_percentile: tuple[float,float], exp_clipnorm_nopost 的 clip 分位数
        - pdb_cache_size: int, 每个 DataLoader worker 最多缓存的完整 PDB bundle 数；按总内存 profiling 调整

    单样本输出:
        - identity/geometry: split、pdb_id、CLG_id、candidate_node_id、BOX 几何
        - candidate: candidate_attributes (N_c,16)、tree_relative_feature (N_c,N_c,8) 与最小连接闭包
        - V: sparse voxel_final/probability/坐标/membership、四张原生网格及现场 density_input
        - Find only P/A: 多层来源、概率、坐标及 candidate→A membership；A_feat_L0 从 receptor 表索引
        - require_oracle=True: candidate_max_iou、oracle_selected_candidate_index、CLG_is_valid
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
        读取并按显式容量缓存一个 PDB 的聚合 centered/forest/CLG/probability 基础表。

        输入参数:
            - record: SelectorInputRecord, 当前冻结 CLG identity

        输出:
            - bundle: dict[str,Any]，包含 pdb_root、centered、forest、clg、probability 及可选 overlap
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
        if self.stage1_model_name in {"Find_0", "Find_1"}:
            with np.load(
                self.upstream_root / "parse" / record.pdb_id / "receptor_tokens.npz",
                allow_pickle=False,
            ) as receptor:
                bundle["receptor_feat"] = np.asarray(receptor["feat"], dtype=np.float32).copy()
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
        保留每个 CLG 自己的 ragged/tree 结构，以列表作为 mini-batch。

        输入参数:
            - samples: list[dict[str,Any]], 长度 B，每项是一个完整 CLG

        输出:
            - batch: list[dict[str,Any]], 长度 B；Wrapper 逐 CLG 前向并在 loss 层做 batch 平均
        """
        return samples

    def source_dimensions(self) -> dict[str, dict[str, int]]:
        """
        从首个冻结样本解析实际 A/P source 通道并供模型配置落盘。

        输出:
            - dimensions: dict，包含 "A"/"P" 到字段名→末维通道数的映射；unet 两者均为空
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
        从上游 raw experimental density 现场裁 BOX 并构造 exp_clipnorm_nopost。

        输入参数:
            - full_grid: np.ndarray, (1,D_full,H_full,W_full)，当前 PDB raw experimental density
            - box_start_zyx: np.ndarray, (3,), 合法 BOX 起点
            - box_shape_zyx: np.ndarray, (3,), 当前 BOX shape

        输出:
            - density_input: np.ndarray, (1,D,H,W), float32，clip-normalized experimental density
        """
        z, y, x = (int(value) for value in box_start_zyx)
        dz, dy, dx = (int(value) for value in box_shape_zyx)
        raw_crop = full_grid[0, z : z + dz, y : y + dy, x : x + dx]
        if raw_crop.shape != (dz, dy, dx):
            raise ValueError("上游 experimental density 无法提供真实 BOX crop。")
        return build_density_channels(raw_crop, None, self.density_config, None)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        物化一个冻结 CLG，并现场生成 candidate 属性、tree pair 特征与 oracle。

        输入参数:
            - index: int, 冻结清单中的样本行

        输出:
            - sample: dict[str,Any]，字段集合见类 Docstring；数值数组均转为 torch.Tensor
        """
        record = self.records[index]
        bundle = self._load_pdb_bundle(record)
        centered = bundle["centered"]
        forest = bundle["forest"]
        clg = bundle["clg"]
        probability = bundle["probability"]

        entry_rows = np.flatnonzero(np.asarray(centered["CLG_id"], dtype=np.int64) == record.CLG_id)
        source_rows = np.flatnonzero(np.asarray(clg["CLG_id"], dtype=np.int64) == record.CLG_id)
        if entry_rows.size != 1 or source_rows.size != 1:
            raise ValueError(f"CLG_id 必须在 centered/source 中各出现一次: {record}")
        entry = int(entry_rows[0])
        source_clg_row = int(source_rows[0])

        box_start_zyx = np.asarray(centered["box_start_zyx"][entry], dtype=np.int64)
        box_shape_zyx = np.asarray(centered["box_shape_zyx"][entry], dtype=np.int64)
        box_origin_world = np.asarray(centered["box_origin_world"][entry], dtype=np.float32)
        voxel_size_world = np.asarray(centered["voxel_size_world"][entry], dtype=np.float32)
        box_shape_xyz = box_shape_zyx[[2, 1, 0]].astype(np.float32)
        box_center_world = box_origin_world + 0.5 * box_shape_xyz * voxel_size_world
        box_half_size_world = 0.5 * box_shape_xyz * voxel_size_world
        full_origin_world = box_origin_world - box_start_zyx[[2, 1, 0]] * voxel_size_world

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
        closure: CandidateTreeClosure = build_candidate_tree_closure(
            node_id=tree_node_ids,
            parent_node_id=tree_parent_ids,
            candidate_node_id=candidate_node_id.tolist(),
        )
        parent_by_id = dict(zip(tree_node_ids, tree_parent_ids, strict=True))
        depth_by_id = {value: _tree_depth(parent_by_id, value) for value in tree_node_ids}

        full_shape_zyx = probability.shape
        probability_flat = np.asarray(probability, dtype=np.float32).reshape(-1)
        candidate_count = candidate_node_id.size
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
        voxel_coord_world = box_origin_world + (
            voxel_index_local_zyx[:, [2, 1, 0]].astype(np.float32) + 0.5
        ) * voxel_size_world
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
            "V_native_grids": {
                name: torch.from_numpy(np.asarray(centered[name][entry : entry + 1], dtype=np.float32))
                for name in ("voxel_ds_2", "voxel_ds_3", "voxel_ds_4", "voxel_c4")
            },
            "density_input": torch.from_numpy(density_input).unsqueeze(0),
        }

        if self.stage1_model_name in {"Find_0", "Find_1"}:
            p_offsets = np.asarray(centered["P_offsets"], dtype=np.int64)
            p_begin, p_end = _slice_offsets(p_offsets, entry)
            p_coord_local = np.asarray(centered["P_coord_local_xyz"][p_begin:p_end], dtype=np.float32)
            sample.update(
                {
                    "P_sources": {
                        name: torch.from_numpy(np.asarray(centered[name][p_begin:p_end], dtype=np.float32))
                        for name in ("P_feat_L2", "P_feat_L3", "P_feat_L4")
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
            a_feat_l0 = recover_a_feat_l0(
                bundle["receptor_feat"],
                a_global_index,
            )
            a_coord_local = np.asarray(centered["A_coord_local_xyz"][a_begin:a_end], dtype=np.float32)
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
                        "A_feat_L0": torch.from_numpy(a_feat_l0),
                        **{
                            name: torch.from_numpy(np.asarray(centered[name][a_begin:a_end], dtype=np.float32))
                            for name in ("A_feat_L1", "A_feat_L2", "A_feat_L3", "A_feat_L4")
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
