"""从现有 Stage1 训练准备产物中删除 16 个辅助标签缺失的 PDB 结构. 

读者应先看 :func:`materialize_filtered_preparation`. 它只做一次确定的筛除和复制: 
保留源清单原有的训练集、验证集、校准集和留出集划分, 删除 ``EXCLUDED_PDB_IDS`` 中的 PDB 编号, 再同步删除相应的 BOX 文件引用和冻结验证请求. 

固定输入根目录:
    ``/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000``

固定输出根目录:
    ``/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation``

输出文件:
    - ``final_keep_list.jsonl``: 可以进入 Stage1 训练准备流程的配体实例清单. 
    - ``split/train.json``、``validation.json``、``calibration.json`` 和
      ``held_out_pool.json``: 删除指定 PDB 后的四份既有划分. 
    - ``box_pool/manifest.json``: 训练集和验证集每个 PDB 对应的 BOX 文件路径. 
    - ``box_pool/validation_selection.npz``: 删除指定 PDB 后的冻结验证 BOX 请求. 
    - ``box_pool/train/{pdb_id}.npz`` 与 ``box_pool/validation/{pdb_id}.npz``: 
      从源目录原样复制的每个 PDB 的候选 BOX 起点. 
    - ``split/_COMPLETE`` 与 ``box_pool/_COMPLETE``: 上述目录全部写完后的完成标记. 

每种文件的全部字段、数组形状和编号关系见 ``src/datasets/readme.md``. 本脚本不重新划分 PDB, 不重新计算 BOX 起点, 不扫描辅助标签目录, 也不修改 ``box_pool/config.json`` 中的训练准备规则. 
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


# 已发布的旧版训练准备根目录; 脚本只从这里读取, 不修改其中任何文件. 
SOURCE_PREPARATION_ROOT = Path(
    "/storage/penghongen/AdaLigand/Ori_Data/"
    "stage1_preparation/adaligand_stage1_20260721T024000"
)
# 例如完整输入路径为 /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/inventory/final_keep_list.jsonl. 
SOURCE_FINAL_KEEP_LIST = (SOURCE_PREPARATION_ROOT / "inventory" / "final_keep_list.jsonl")
SOURCE_SPLIT_ROOT = SOURCE_PREPARATION_ROOT / "split"
SOURCE_BOX_POOL_ROOT = SOURCE_PREPARATION_ROOT / "box_pool"


# 新版训练直接引用的正式根目录; 运行前可以在这里修改为用户最终选定的目录. 
TARGET_PREPARATION_ROOT = Path(
    "/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation"
)
# 例如正式清单会写到 /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/final_keep_list.jsonl. 
TARGET_FINAL_KEEP_LIST = TARGET_PREPARATION_ROOT / "final_keep_list.jsonl"
TARGET_SPLIT_ROOT = TARGET_PREPARATION_ROOT / "split"
TARGET_BOX_POOL_ROOT = TARGET_PREPARATION_ROOT / "box_pool"


# 这 16 个 PDB 缺少本轮训练需要的辅助标签, 因此从所有实际含有它们的清单、
# BOX 文件引用和冻结验证请求中删除; 源产物本来就不含某个编号时只输出提示. 
EXCLUDED_PDB_IDS = frozenset(
    {
        "1q5c",
        "2w49",
        "5y6p",
        "6k0a",
        "7cbp",
        "7cth",
        "7n6g",
        "7ojf",
        "7pel",
        "7wc2",
        "7z8g",
        "8olc",
        "9hhl",
        "9v7i",
        "9wqp",
        "9yx6",
    }
)

SPLIT_FILENAMES = (
    "train.json",
    "validation.json",
    "calibration.json",
    "held_out_pool.json",
)
MANIFEST_FILENAME = "manifest.json"
VALIDATION_SELECTION_FILENAME = "validation_selection.npz"
BOX_POOL_CONFIG_FILENAME = "config.json"
COMPLETE_FILENAME = "_COMPLETE"

VALIDATION_SELECTION_FIELDS = {
    "validation_pdb_id",
    "center_pdb_index",
    "center_occurrence_id",
    "bias_pdb_index",
    "bias_occurrence_id",
    "bias_candidate_index",
    "context_pdb_index",
    "context_candidate_index",
}


def _normalize_pdb_id(value: Any, source: Path) -> str:
    """把一个 PDB 编号规范为四位小写字母数字字符串. 

    输入参数:
        - value: 从 ``pdb_id`` 或 ``validation_pdb_id`` 字段读出的单个值; 可以是 Python 字符串、NumPy 定长字节串或可转为字符串的标量. 
        - source: 提供该编号的具体文件路径, 仅用于在异常信息中定位文件. 

    返回:
        - pdb_id: 四位小写字母数字字符串, 例如输入 ``"1Q5C"`` 返回 ``"1q5c"``. 
    """
    pdb_id = str(value).strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise ValueError(f"{source}: 无效 pdb_id={value!r}。")
    return pdb_id


def _read_final_keep_list(path: Path) -> tuple[list[str], set[str]]:
    """读取 ``inventory/final_keep_list.jsonl`` 并删除指定 PDB 的配体实例. 

    输入文件:
        - ``path`` 的每个非空文本段是一条 JSON object. 
        - pdb_id: str, 四位 PDB 编号. 
        - candidate_id: int, 同一 PDB 内的候选配体编号. 
        - 其余 14 个正式字段及其含义见 ``src/datasets/readme.md``. 

    返回:
        - kept_text: ``list[str]``, 每个元素是一条未排除配体实例的原始 JSON 文本; 保持源文件顺序、字段顺序和字段值不变. 
        - found_pdb_ids: ``set[str]``, 源文件中出现过的全部小写 PDB 编号, 包括随后被删除的编号. 

    例如当前固定输入为
    ``/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/inventory/final_keep_list.jsonl``. 
    """
    # 每个元素对应输出 final_keep_list.jsonl 的一条完整 JSON 文本. 
    kept_text: list[str] = []
    # 源清单中的全部 PDB 编号, 用于报告哪些排除编号在旧版产物中本来就不存在. 
    found_pdb_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 必须是 JSON object。")
            pdb_id = _normalize_pdb_id(value.get("pdb_id", ""), path)
            found_pdb_ids.add(pdb_id)
            if pdb_id not in EXCLUDED_PDB_IDS:
                kept_text.append(text)
    if not kept_text:
        raise ValueError(f"{path}: 排除指定 PDB 后 final_keep_list 为空。")
    return kept_text, found_pdb_ids


def _read_split(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """读取一份既有数据划分并删除指定 PDB 的配体实例. 

    输入文件:
        - ``path`` 顶层是 JSON array; 实际文件名为 ``train.json``、``validation.json``、``calibration.json`` 或 ``held_out_pool.json``. 
        - 数组中的每个 JSON object 至少包含 ``pdb_id`` 和 ``candidate_id``; 两个字段共同标识一个配体实例. 

    返回:
        - kept: list[dict[str, Any]], 从 train.json、validation.json、calibration.json 或 held_out_pool.json 中保留下来的候选配体实例; 每个字典至少包含 pdb_id 和 candidate_id, 字段值与源文件一致, 数组顺序也与源 JSON array 一致. pdb_id 属于 EXCLUDED_PDB_IDS 的实例不会出现在这里. 
        - found_pdb_ids: set[str], 该 JSON array 中实际出现过的全部小写 PDB 编号, 包括后来被删除的编号; 只用于报告哪些待排除编号原本不存在, 不会直接写入输出文件. 
    """
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise TypeError(f"{path}: 顶层必须是 JSON array。")

    # 每个字典仍代表源划分中的一个配体实例, 不重新分配到其他划分. 
    kept: list[dict[str, Any]] = []
    # 当前划分的全部 PDB 编号, 包括随后被删除的编号. 
    found_pdb_ids: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise TypeError(f"{path}: 每个配体实例必须是 JSON object。")
        pdb_id = _normalize_pdb_id(value.get("pdb_id", ""), path)
        found_pdb_ids.add(pdb_id)
        if pdb_id not in EXCLUDED_PDB_IDS:
            kept.append(value)
    if not kept:
        raise ValueError(f"{path}: 排除指定 PDB 后 split 为空。")
    return kept, found_pdb_ids


def _read_manifest(
    path: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]], set[str]]:
    """读取 ``box_pool/manifest.json`` 并删除内部指定的 PDB 的 BOX 文件引用. 

    - path: /storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000/box_pool/manifest.json, manifest.json 的实际路径. 

    manifest的内部字段:
        - schema_version: int, 当前只接受 ``1``. 
        - splits: JSON object, 只读取 ``train`` 和 ``validation``. 
        - ``splits.train``、``splits.validation``: JSON array, 每个元素包含: 
          - pdb_id: str, 四位 PDB 编号. 
          - path: str, 相对于源 ``box_pool`` 的 POSIX 路径, 例如
            ``train/{pdb_id}.npz`` 或 ``validation/{pdb_id}.npz``. 

    返回:
        - filtered_manifest: ``dict``, 保留源清单其他顶层字段, 只把 ``splits`` 替换成筛除后的文件引用. 
        - filtered_splits: ``dict[str, list[dict[str, str]]]``, 键为 ``train`` 和 ``validation``; 每个元素只含 ``pdb_id``、``path``, 供中心函数逐个复制 NPZ 文件. 
        - found_pdb_ids: ``set[str]``, 源 manifest 中出现过的全部 PDB 编号. 

    ``path`` 不得是绝对路径或越过 ``box_pool``. 例如 ``train/1abc.npz`` 会解析为 ``SOURCE_BOX_POOL_ROOT / "train/1abc.npz"``
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version 必须为 1。")
    source_splits = manifest.get("splits")
    if not isinstance(source_splits, dict):
        raise TypeError(f"{path}: splits 必须是 JSON object。")

    # 每个键对应一个实际 BOX 子目录("train", "validation"), 值按源 manifest 顺序保存未排除文件引用. 
    filtered_splits: dict[str, list[dict[str, str]]] = {}
    # 源 manifest 的全部 PDB 编号, 用于统一报告排除编号是否命中. 
    found_pdb_ids: set[str] = set()
    for split_name in ("train", "validation"):
        entries = source_splits.get(split_name)
        if not isinstance(entries, list):
            raise TypeError(f"{path}: splits.{split_name} 必须是 JSON array。")

        # 每个元素固定为 {"pdb_id": <四位编号>, "path": <相对 NPZ 路径>}. 
        filtered_entries: list[dict[str, str]] = []
        # 当前 train 或 validation 内已经出现的 PDB 编号, 禁止重复引用. 
        seen: set[str] = set()
        for value in entries:
            if not isinstance(value, dict):
                raise TypeError(f"{path}: splits.{split_name} 的每个文件引用必须是 JSON object。")
            pdb_id = _normalize_pdb_id(value.get("pdb_id", ""), path)
            relative_path = Path(str(value.get("path", "")))
            if pdb_id in seen:
                raise ValueError(f"{path}: splits.{split_name} 重复包含 pdb_id={pdb_id!r}。")
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or not relative_path.parts
                or relative_path.parts[0].lower() != split_name
            ):
                raise ValueError(
                    f"{path}: {pdb_id} 的 BOX 文件必须位于 {split_name}/，"
                    f"实际为 {relative_path}。"
                )
            source_pool_path = SOURCE_BOX_POOL_ROOT / relative_path
            if not source_pool_path.is_file():
                raise FileNotFoundError(f"{path}: 找不到 {pdb_id} 的 BOX 文件 {source_pool_path}。")

            found_pdb_ids.add(pdb_id)
            seen.add(pdb_id)
            if pdb_id not in EXCLUDED_PDB_IDS:
                filtered_entries.append({"pdb_id": pdb_id, "path": relative_path.as_posix()})
        if not filtered_entries:
            raise ValueError(f"{path}: 排除指定 PDB 后 splits.{split_name} 为空。")
        filtered_splits[split_name] = filtered_entries

    filtered_manifest = dict(manifest)
    filtered_manifest["splits"] = filtered_splits
    return filtered_manifest, filtered_splits, found_pdb_ids


def _decode_pdb_array(values: np.ndarray, path: Path) -> list[str]:
    """解码冻结验证文件中的 ``validation_pdb_id`` 数组: 实质只使用 _normalize_pdb_id

    输入参数:
        - values: ``(P,)``, NumPy 定长字节串或 Unicode 字符串数组; 第 ``i`` 个值是冻结验证 PDB 表中的第 ``i`` 个 PDB 编号. 
        - path: ``validation_selection.npz`` 的具体路径, 仅用于异常定位. 

    返回:
        - pdb_ids: 长度为 ``P`` 的 ``list[str]``, 元素为四位小写 PDB 编号; 元素顺序与 ``values`` 完全一致. 

    ``P`` 表示筛除前冻结验证 PDB 的数量. 
    """
    if values.ndim != 1 or values.dtype.kind not in {"S", "U"}:
        raise TypeError(f"{path}: validation_pdb_id 必须是一维定长字符串数组。")
    decoded = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]
    pdb_ids = [_normalize_pdb_id(value, path) for value in decoded]
    if len(pdb_ids) != len(set(pdb_ids)):
        raise ValueError(f"{path}: validation_pdb_id 含重复 PDB 编号。")
    return pdb_ids


def _filter_selection_group(
    arrays: dict[str, np.ndarray],
    old_to_new_pdb_index: np.ndarray,
    pdb_index_field: str,
    aligned_fields: tuple[str, ...],
    source: Path,
) -> dict[str, np.ndarray]:
    """筛除一类冻结验证请求并重写其中的 PDB 表编号. 

    形状符号:
        - P: 筛除前 ``validation_pdb_id`` 的 PDB 数量. 
        - R: 当前请求类型在筛除前的请求数量. 
        - K: 当前请求类型在筛除后的请求数量. 

    输入参数:
        - arrays: 筛出前 ``validation_selection.npz`` 的全部数组. 
        - old_to_new_pdb_index: int64, ``(P,)``, 旧 PDB 表编号到新 PDB 表编号的映射; 值 ``-1`` 表示该 PDB 已被排除. 
        - pdb_index_field: 当前请求使用的 PDB 编号字段, 取值为 ``center_pdb_index``、``bias_pdb_index`` 或 ``context_pdb_index``. 
        - aligned_fields: 与 ``pdb_index_field`` 第一维逐请求对齐的字段名(变长). 例如 ``bias_pdb_index`` 对齐 ``bias_occurrence_id`` 和 ``bias_candidate_index``. 
        - source: 源 ``validation_selection.npz`` 路径, 仅用于异常定位. 

    返回 filtered, 内部字段:
        - pdb_index_field: 与源字段相同的整数数据类型, ``(K,)``; 每个数值索引筛除后的 ``validation_pdb_id`` 第一维. 
        - ``aligned_fields`` 中的每个字段: 保持源数据类型, ``(K,)``; 删除与被排除 PDB 对应的请求, 其余数值与顺序不变. 
    """

    # 整数数组, (R,), 每个数值索引筛除前 validation_pdb_id 的第一维. 
    old_pdb_index = np.asarray(arrays[pdb_index_field])
    if old_pdb_index.ndim != 1:
        raise ValueError(f"{source}: {pdb_index_field} 必须是一维数组。")
    if np.any(old_pdb_index < 0) or np.any(old_pdb_index >= old_to_new_pdb_index.shape[0]):
        raise IndexError(f"{source}: {pdb_index_field} 含超出 validation_pdb_id 的编号。")

    # bool, (R,), True 表示该验证请求引用的 PDB 仍保留. 
    keep_request = old_to_new_pdb_index[old_pdb_index] >= 0
    # 新 PDB 编号为 [0, P_new) 的连续整数, 仍与各请求专属字段逐项对齐. 
    filtered = {
        pdb_index_field: old_to_new_pdb_index[old_pdb_index[keep_request]].astype(
            old_pdb_index.dtype,
            copy=False,
        )
    }
    for field_name in aligned_fields:
        values = np.asarray(arrays[field_name])
        if values.ndim != 1 or values.shape[0] != old_pdb_index.shape[0]:
            raise ValueError(f"{source}: {field_name} 必须与 {pdb_index_field} 等长。")
        filtered[field_name] = values[keep_request]
    return filtered


def _read_filtered_validation_selection(
    path: Path,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """读取并筛除 ``box_pool/validation_selection.npz``. 

    输入字段:
        - validation_pdb_id: ``(P,)``, 定长字节串或 Unicode 字符串; 冻结验证请求共用的 PDB 编号表. 
        - center_pdb_index: 整数 ``(R_center,)``, 索引 ``validation_pdb_id``; 指定每个中心 BOX 请求所属的 PDB. 
        - center_occurrence_id: 整数 ``(R_center,)``, 索引对应 PDB 的 ``occurrence_id`` 数组. 
        - bias_pdb_index: 整数 ``(R_bias,)``, 索引 ``validation_pdb_id``; 指定每个偏移 BOX 请求所属的 PDB. 
        - bias_occurrence_id: 整数 ``(R_bias,)``, 索引对应 PDB 的 ``occurrence_id`` 数组. 
        - bias_candidate_index: 整数 ``(R_bias,)``, 索引对应 PDB 的 ``bias_start_zyx`` 第一维. 
        - context_pdb_index: 整数 ``(R_context,)``, 索引 ``validation_pdb_id``; 指定每个背景 BOX 请求所属的 PDB. 
        - context_candidate_index: 整数 ``(R_context,)``, 索引对应 PDB 的 ``context_start_zyx`` 第一维. 

    返回:
        - filtered: ``dict[str, np.ndarray]``, 字段名与输入完全相同; ``validation_pdb_id`` 删除指定编号, 三类请求分别删除对这些编号的引用, 并把剩余 ``*_pdb_index`` 改为新 PDB 表中的连续编号. 
        - found_pdb_ids: ``set[str]``, 筛除前 ``validation_pdb_id`` 中的全部小写 PDB 编号. 

    ``P`` 是筛除前验证 PDB 数量; ``R_center``、``R_bias`` 和 ``R_context`` 分别是三类冻结验证请求数量. 
    """

    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != VALIDATION_SELECTION_FIELDS:
            missing = sorted(VALIDATION_SELECTION_FIELDS.difference(source.files))
            unexpected = sorted(set(source.files).difference(VALIDATION_SELECTION_FIELDS))
            raise KeyError(f"{path}: validation selection 字段不符合契约；缺少={missing}，额外={unexpected}。")
        # 八个 NumPy 数组(对应输入字段); 字段集合固定, 详细形状由本函数 Docstring 定义. 
        arrays = {
            field_name: np.asarray(source[field_name])
            for field_name in source.files
        }

    # 长度 P 的 PDB 编号表; 其数组位置是三个 *_pdb_index 字段的旧编号空间. 
    pdb_ids = _decode_pdb_array(arrays["validation_pdb_id"], path)
    # bool, (P,), True 表示该 PDB 及其三类冻结验证请求都保留. 
    keep_pdb = np.asarray(
        [pdb_id not in EXCLUDED_PDB_IDS for pdb_id in pdb_ids],
        dtype=np.bool_,
    )
    # int64, (P,), 旧 PDB 表编号映射到筛除后的连续编号; 被排除位置为 -1. 
    old_to_new = np.full(len(pdb_ids), -1, dtype=np.int64)
    old_to_new[keep_pdb] = np.arange(int(keep_pdb.sum()), dtype=np.int64)
    if not np.any(keep_pdb):
        raise ValueError(f"{path}: 排除指定 PDB 后验证 PDB 表为空。")

    # 筛除后的共享 PDB 编号表; 三类请求的 *_pdb_index 都改为索引此数组. 
    filtered: dict[str, np.ndarray] = {
        "validation_pdb_id": arrays["validation_pdb_id"][keep_pdb],
    }
    filtered.update(
        _filter_selection_group(
            arrays,
            old_to_new,
            "center_pdb_index",
            ("center_occurrence_id",),
            path,
        )
    )
    filtered.update(
        _filter_selection_group(
            arrays,
            old_to_new,
            "bias_pdb_index",
            ("bias_occurrence_id", "bias_candidate_index"),
            path,
        )
    )
    filtered.update(
        _filter_selection_group(
            arrays,
            old_to_new,
            "context_pdb_index",
            ("context_candidate_index",),
            path,
        )
    )
    return filtered, set(pdb_ids)


def _require_source_complete() -> None:
    """确认固定源目录包含筛除操作所需的全部已发布文件. 

    必须存在:
        - ``inventory/final_keep_list.jsonl``; 
        - ``split/_COMPLETE``; 
        - ``box_pool/_COMPLETE``; 
        - ``box_pool/manifest.json``; 
        - ``box_pool/validation_selection.npz``; 
        - ``box_pool/config.json``. 

    两个 ``_COMPLETE`` 只证明源 split 和 BOX 池已经完成发布; 本函数不解析标记内容. 四份 split JSON 和 manifest 引用的每个 NPZ 会在后续读取时分别检查. 
    """
    required_paths = (
        SOURCE_FINAL_KEEP_LIST,
        SOURCE_SPLIT_ROOT / COMPLETE_FILENAME,
        SOURCE_BOX_POOL_ROOT / COMPLETE_FILENAME,
        SOURCE_BOX_POOL_ROOT / MANIFEST_FILENAME,
        SOURCE_BOX_POOL_ROOT / VALIDATION_SELECTION_FILENAME,
        SOURCE_BOX_POOL_ROOT / BOX_POOL_CONFIG_FILENAME,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"源 Stage1 preparation 缺少文件: {missing}。")


def _require_target_outputs_absent() -> None:
    """确认输出根目录存在, 并拒绝覆盖三个正式产物入口. 

    固定输出入口:
        - ``/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/final_keep_list.jsonl``; 
        - ``/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/split``; 
        - ``/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/box_pool``. 

    输出根目录可以预先存在, 但上述任意文件或目录已经存在时立即停止. 
    """
    if not TARGET_PREPARATION_ROOT.is_dir():
        raise FileNotFoundError(
            f"正式 Stage1 preparation 根目录不存在: {TARGET_PREPARATION_ROOT}。"
        )
    existing = [
        str(path)
        for path in (
            TARGET_FINAL_KEEP_LIST,
            TARGET_SPLIT_ROOT,
            TARGET_BOX_POOL_ROOT,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"以下正式产物已经存在，拒绝覆盖: {existing}。")


def materialize_filtered_preparation() -> None:
    """用固定路径和固定排除集合生成正式 Stage1 训练准备产物. 

    处理顺序:
        1. 确认旧版源产物完整, 且三个正式输出入口尚不存在. 
        2. 在内存中读取 final keep list、四份 split、BOX manifest 和冻结验证请求(validation_selection.npz); 任何源文件不符合字段契约时, 在创建输出目录前停止. 
        3. 汇总 ``EXCLUDED_PDB_IDS`` 中没有出现在任何源产物里的编号, 并输出提示. 
        4. 写出筛除后的清单和 split, 复制保留 PDB 的 BOX NPZ, 写出筛除后的 manifest 与冻结验证请求, 并原样复制 ``box_pool/config.json``. 
        5. 最后创建 ``split/_COMPLETE`` 和 ``box_pool/_COMPLETE``. 

    输出字段:
        - ``final_keep_list.jsonl`` 与四份 split 的每个配体实例保留源字段, 其中 ``pdb_id``、``candidate_id`` 的意义不变. 
        - ``manifest.json`` 保留 ``schema_version=1``, 其 ``splits.train`` 和 ``splits.validation`` 每个元素包含 ``pdb_id`` 与相对 NPZ 路径 ``path``. 
        - ``validation_selection.npz`` 保留八个原字段; 三类 ``*_pdb_index`` 均索引筛除后的 ``validation_pdb_id``. 
        - 单个 PDB 的 BOX NPZ 和 ``config.json`` 逐文件复制, 不改字段、数组形状、数据类型或数值. 

    安全边界:
        - 本函数直接写入 ``TARGET_PREPARATION_ROOT``, 不会自动清理失败后留下的部分目录; 没有 ``_COMPLETE`` 的目录不得用于训练. 
        - 本函数不覆盖已有正式入口, 不修改固定源目录, 也不启动训练. 

    详细产物说明见 ``src/datasets/readme.md``. 
    """

    _require_source_complete()
    _require_target_outputs_absent()

    # 未排除的 JSON 文本, 以及筛除前 final keep list 的全部 PDB 编号. 
    kept_keep_list, keep_list_pdb_ids = _read_final_keep_list(SOURCE_FINAL_KEEP_LIST)
    # 四个键为实际 split 文件名; 每个值是该划分中保留的配体实例字典. 
    filtered_splits: dict[str, list[dict[str, Any]]] = {}
    # 四份源 split 中出现过的全部 PDB 编号. 
    split_pdb_ids: set[str] = set()
    for filename in SPLIT_FILENAMES:
        values, pdb_ids = _read_split(SOURCE_SPLIT_ROOT / filename)
        filtered_splits[filename] = values
        split_pdb_ids.update(pdb_ids)

    # 筛除后的 manifest、按 train/validation 分组的文件引用, 以及源 PDB 集合. 
    manifest, manifest_splits, manifest_pdb_ids = _read_manifest(SOURCE_BOX_POOL_ROOT / MANIFEST_FILENAME)
    # 八个筛除后的冻结验证数组, 以及筛除前验证 PDB 编号集合. 
    validation_selection, selection_pdb_ids = (
        _read_filtered_validation_selection(
            SOURCE_BOX_POOL_ROOT / VALIDATION_SELECTION_FILENAME
        )
    )

    # 所有源清单的 PDB 编号并集, 用于提示哪些固定排除编号原本就不存在. 
    found_pdb_ids = (
        keep_list_pdb_ids
        | split_pdb_ids
        | manifest_pdb_ids
        | selection_pdb_ids
    )
    missing_excluded_ids = sorted(EXCLUDED_PDB_IDS.difference(found_pdb_ids))
    if missing_excluded_ids:
        print(f"提示: 以下固定排除 PDB 未出现在源 final_keep_list、split、BOX manifest 或 validation selection 中: {missing_excluded_ids}。")


    # 直到全部源产物读取和字段核对成功后, 才开始创建正式输出目录. 
    TARGET_SPLIT_ROOT.mkdir()
    (TARGET_BOX_POOL_ROOT / "train").mkdir(parents=True)
    (TARGET_BOX_POOL_ROOT / "validation").mkdir()

    TARGET_FINAL_KEEP_LIST.write_text(
        "\n".join(kept_keep_list) + "\n",
        encoding="utf-8",
    )
    for filename, values in filtered_splits.items():
        (TARGET_SPLIT_ROOT / filename).write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # manifest 中每个 path 都相对于 box_pool, 例如 train/{pdb_id}.npz; 
    # copy2 原样保留 NPZ 字段、数组数据以及可保留的文件元信息. 
    for entries in manifest_splits.values():
        for value in entries:
            relative_path = Path(value["path"])
            shutil.copy2(
                SOURCE_BOX_POOL_ROOT / relative_path,
                TARGET_BOX_POOL_ROOT / relative_path,
            )

    (TARGET_BOX_POOL_ROOT / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        TARGET_BOX_POOL_ROOT / VALIDATION_SELECTION_FILENAME,
        **validation_selection,
    )
    shutil.copy2(
        SOURCE_BOX_POOL_ROOT / BOX_POOL_CONFIG_FILENAME,
        TARGET_BOX_POOL_ROOT / BOX_POOL_CONFIG_FILENAME,
    )

    # 完成标记最后写入; 训练入口只能把存在 _COMPLETE 的目录视为完整产物. 
    (TARGET_SPLIT_ROOT / COMPLETE_FILENAME).write_text("", encoding="utf-8")
    (TARGET_BOX_POOL_ROOT / COMPLETE_FILENAME).write_text("", encoding="utf-8")


if __name__ == "__main__":
    materialize_filtered_preparation()
