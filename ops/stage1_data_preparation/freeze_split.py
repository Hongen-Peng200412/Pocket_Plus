"""按 EMDB 首次发布时间、质量阈值和资产契约冻结 Stage1 V3 PDB split。

命令入口是 ``fetch-release-dates`` 和 ``freeze``；函数入口分别为 :func:`fetch_release_dates` 与 :func:`freeze_split`。前者从 EMDB 官方 ``entry/admin`` 接口建立可续传的 ``map_release`` 日志；后者把 Stage G 候选按 PDB 聚合，先隔离 2026-01-01 及以后首次发布的 PDB，再对更早 PDB 应用 ``map_resolution < 4``、``cc_contour > 0.65``、完整图至少 80³ 和迁移资产契约。

本模块只在 ``output_root`` 写 split JSON、审计 JSONL、配置、摘要和 ``_COMPLETE``，不修改 A—G 正式资产，也不构造 BOX 起点。split JSON 的顶层是候选记录列表；审计 JSONL 每行是一个 PDB 审计对象；``config.json`` 保存阈值、日期和 seed；``summary.json`` 保存各状态计数与 split 数量。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ops.stage1_data_preparation.atomic_io import atomic_write_json, atomic_write_text


EMDB_ADMIN_URL = "https://www.ebi.ac.uk/emdb/api/entry/admin/{emdb_id}"
EMDB_USER_AGENT = "Pocket-Plus-Stage1-Data-Preparation/1.0"
RELEASE_CUTOFF = date(2026, 1, 1)
MAP_RESOLUTION_EXCLUSIVE_MAX = 4.0
CC_CONTOUR_EXCLUSIVE_MIN = 0.65
VALIDATION_PDB_COUNT = 200
CALIBRATION_PDB_COUNT = 100
SPLIT_SEED = 3407
MIN_GRID_SHAPE_ZYX = (80, 80, 80)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 记录，并在顶层类型错误时报告文件行号。

    输入参数:
        - path: Path；UTF-8 JSONL 文件；空白行跳过，其余每行必须是 JSON object。

    返回值:
        - records: list[dict[str, Any]]；按文件顺序保存的对象记录，不重排、不去重。

    失败语义:
        - JSON 语法错误或某行顶层不是 object 时抛出异常；错误消息包含具体路径和行号。
    """

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} 必须是 JSON object。")
            records.append(value)
    return records


def normalize_emdb_id(value: Any) -> str:
    """把数字或带前缀的 EMDB identity 规范为 ``EMD-<整数>``。

    输入参数:
        - value: Any；允许 ``21605``、``EMD_21605``、``EMD-21605`` 或 ``EMD21605`` 等可转字符串写法。

    返回值:
        - emdb_id: str；大写的 ``EMD-`` 前缀和无前导零整数编号。

    失败语义:
        - 去除空白、下划线和可选前缀后仍不是纯数字时抛出 ``ValueError``。
    """

    text = str(value).strip().upper().replace("_", "-")
    if text.startswith("EMD-"):
        number = text[4:]
    elif text.startswith("EMD"):
        number = text[3:].lstrip("-")
    else:
        number = text
    if not number.isdigit():
        raise ValueError(f"非法 EMDB identity：{value!r}")
    return f"EMD-{int(number)}"


def candidate_pdb_to_emdb_ids(
    candidates_path: Path,
    pair_list_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[str, ...]]]:
    """读取候选与 PDB–EMDB 对照并建立 PDB 级分组。

    输入参数:
        - candidates_path: Path；候选 JSONL；每条记录必须有非空 ``pdb_id``。
        - pair_list_path: Path；PDB–EMDB 对照 JSONL；``emdb_id`` 通过 :func:`normalize_emdb_id` 规范化。

    返回值:
        - candidate_groups: dict[str, list[dict[str, Any]]]；小写 PDB identity 到原候选记录（保持文件顺序）的映射。
        - pdb_to_emdb_ids: dict[str, tuple[str, ...]]；每个候选 PDB 对应的去重、排序 EMDB identity 集合。

    失败语义:
        - 候选记录缺少 PDB identity，或候选 PDB 没有任何对照 EMDB 时抛出异常。
    """

    candidate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(candidates_path):
        pdb_id = str(record.get("pdb_id", "")).strip().lower()
        if not pdb_id:
            raise ValueError(f"{candidates_path} 含缺少 pdb_id 的记录。")
        candidate_groups[pdb_id].append(record)
    emdb_sets: dict[str, set[str]] = defaultdict(set)
    for record in read_jsonl(pair_list_path):
        pdb_id = str(record.get("pdb_id", "")).strip().lower()
        if pdb_id in candidate_groups:
            emdb_sets[pdb_id].add(normalize_emdb_id(record.get("emdb_id")))
    missing_pair = sorted(set(candidate_groups).difference(emdb_sets))
    if missing_pair:
        raise ValueError(f"候选 PDB 缺少 PDB–EMDB 对照：{missing_pair[:20]}")
    return dict(candidate_groups), {
        pdb_id: tuple(sorted(emdb_sets[pdb_id])) for pdb_id in sorted(candidate_groups)
    }


def fetch_one_release_date(argument: tuple[str, float, int]) -> dict[str, Any]:
    """请求一个 EMDB admin 记录并返回可追加到续传日志的标准字段。

    输入参数:
        - argument: tuple[str, float, int]；依次为 EMDB identity、单次请求超时秒数和最大尝试次数。

    返回值:
        - record: dict[str, Any]；包含 ``emdb_id``、``map_release``（ISO 日期或 None）、``status`` 和请求 URL。

    网络语义:
        - HTTP 404 返回 ``not_found``；空日期返回 ``missing_map_release``；可重试的 URL、JSON、字段和日期错误在达到次数后抛出原异常。
    """

    emdb_id, timeout_seconds, retry_count = argument
    request = urllib.request.Request(
        EMDB_ADMIN_URL.format(emdb_id=emdb_id), headers={"User-Agent": EMDB_USER_AGENT}
    )
    for attempt in range(1, retry_count + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.load(response)
            raw_release = payload["admin"]["key_dates"].get("map_release")
            release = None if raw_release in (None, "") else str(raw_release)[:10]
            if release is not None:
                date.fromisoformat(release)
            return {
                "emdb_id": emdb_id,
                "map_release": release,
                "status": "ok" if release is not None else "missing_map_release",
                "source_url": request.full_url,
            }
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return {
                    "emdb_id": emdb_id,
                    "map_release": None,
                    "status": "not_found",
                    "source_url": request.full_url,
                }
            if attempt == retry_count:
                raise
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError):
            if attempt == retry_count:
                raise
        time.sleep(float(attempt))
    raise RuntimeError(f"无法获取 {emdb_id} 的 map_release。")


def fetch_release_dates(arguments: argparse.Namespace) -> None:
    """并发补齐 EMDB 发布时间日志，并跳过已有 identity。

    输入参数:
        - arguments: argparse.Namespace；包含 candidates、pair-list、output、workers、timeout-seconds、retry-count 和 fsync-interval。

    状态变化:
        - 读取已有 JSONL 作为续传缓存，只追加尚未出现的 EMDB identity；定期 fsync，全部完成后原子写入 ``<output>.complete``。

    返回值:
        - None；成功时打印 required/fetched 摘要，缺少任何必需 EMDB 日期时抛出异常。
    """

    candidates_path = Path(arguments.candidates)
    pair_list_path = Path(arguments.pair_list)
    output_path = Path(arguments.output)
    _, pdb_to_emdb_ids = candidate_pdb_to_emdb_ids(candidates_path, pair_list_path)
    required_ids = sorted({emdb_id for values in pdb_to_emdb_ids.values() for emdb_id in values})
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for record in read_jsonl(output_path):
            existing[normalize_emdb_id(record["emdb_id"])] = record
    pending_ids = [emdb_id for emdb_id in required_ids if emdb_id not in existing]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        arguments_by_id = [
            (emdb_id, float(arguments.timeout_seconds), int(arguments.retry_count))
            for emdb_id in pending_ids
        ]
        with ThreadPoolExecutor(max_workers=int(arguments.workers)) as executor:
            futures = {executor.submit(fetch_one_release_date, item): item[0] for item in arguments_by_id}
            completed_since_sync = 0
            for future in as_completed(futures):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                existing[str(record["emdb_id"])] = record
                completed_since_sync += 1
                if completed_since_sync >= int(arguments.fsync_interval):
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed_since_sync = 0
        handle.flush()
        os.fsync(handle.fileno())
    missing = sorted(set(required_ids).difference(existing))
    if missing:
        raise RuntimeError(f"EMDB 发布时间日志仍缺少：{missing[:20]}")
    atomic_write_text(output_path.with_suffix(output_path.suffix + ".complete"), "")
    print(json.dumps({"required": len(required_ids), "fetched": len(pending_ids)}, ensure_ascii=False))


def stable_eval_rank(seed: int, pdb_id: str) -> bytes:
    """生成与候选遍历顺序无关的 PDB 评估集排序键。

    输入参数:
        - seed: int；split 选择的全局 seed。
        - pdb_id: str；已经规范化的小写 PDB identity。

    返回值:
        - rank_key: bytes；``sha256(seed|eval|pdb_id)`` 的完整摘要，供字典序稳定排序。
    """

    return hashlib.sha256(f"{seed}|eval|{pdb_id}".encode("utf-8")).digest()


def inspect_training_assets(data_root: Path, pdb_id: str) -> dict[str, Any]:
    """核对一个 PDB 的迁移资产、几何一致性和完整图最小形状。

    输入参数:
        - data_root: Path；A-G 正式资产根目录。
        - pdb_id: str；小写 PDB identity；检查 ``density``、``parse`` 和 ``labels`` 下的约定文件。

    返回字段:
        - ``status``: str；``eligible``、``short_map``、``missing_file`` 或 ``invalid``。
        - ``shape_zyx``: list[int] | None；exp 完整图的 ZYX 形状；缺失或无效时为 ``None``。
        - ``detail``: str；失败时的具体路径或契约原因，合格或短图时为空字符串。

    检查边界:
        - ``exp.npz`` 读取并核对 ``canonical_shape_zyx``、``voxel_size`` 和 ``origin``；``exp.npy``、``sim.npy``、``ligand_dist.npy``、``union_mask.npy`` 均必须是 ``(1,D,H,W)``，分别为 float32、float32、float16、bool。
        - ``sim.npz`` 只核对 ``voxel_size`` 和 ``origin`` 与 exp 元数据一致；本入口不核对 sim NPZ 的 schema 或 canonical shape 字段。
        - ``ligand_dist.npz`` 与 ``ligand_area.npz`` 读取 ``grid_shape_zyx``、``voxel_size_xyz``、``origin_xyz``，要求与 exp 的形状和几何一致；受体 ``coords`` 的第一维必须等于标签 ``binding_atom`` 的长度。
        - 只检查文件存在、上述 dtype/形状/元数据、受体标签长度和完整图至少 80³；不扫描完整图数值，也不构造 BOX。
    """

    density_directory = data_root / "density" / pdb_id
    required_paths = (
        density_directory / "exp.npz",
        density_directory / "exp.npy",
        density_directory / "sim.npz",
        density_directory / "sim.npy",
        density_directory / "ligand_dist.npz",
        density_directory / "ligand_dist.npy",
        density_directory / "ligand_area.npz",
        density_directory / "union_mask.npy",
        data_root / "parse" / pdb_id / "receptor_tokens.npz",
        data_root / "labels" / pdb_id / "atom_labels.npz",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return {"status": "missing_file", "shape_zyx": None, "detail": "; ".join(missing)}
    try:
        with np.load(density_directory / "exp.npz", allow_pickle=False) as exp_meta:
            shape_zyx = tuple(int(value) for value in np.asarray(exp_meta["canonical_shape_zyx"]).tolist())
            voxel_size_xyz = np.asarray(exp_meta["voxel_size"], dtype=np.float32)
            origin_xyz = np.asarray(exp_meta["origin"], dtype=np.float32)
        if len(shape_zyx) != 3 or any(value <= 0 for value in shape_zyx):
            raise ValueError(f"canonical_shape_zyx 非法：{shape_zyx}")
        expected_shape = (1, *shape_zyx)
        array_contracts = (
            (density_directory / "exp.npy", np.dtype(np.float32)),
            (density_directory / "sim.npy", np.dtype(np.float32)),
            (density_directory / "ligand_dist.npy", np.dtype(np.float16)),
            (density_directory / "union_mask.npy", np.dtype(np.bool_)),
        )
        for npy_path, expected_dtype in array_contracts:
            array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError(
                    f"{npy_path} 应为 {expected_dtype} {expected_shape}，实际 {array.dtype} {array.shape}"
                )
        with np.load(density_directory / "sim.npz", allow_pickle=False) as sim_meta:
            if not np.array_equal(np.asarray(sim_meta["voxel_size"]), voxel_size_xyz) or not np.array_equal(
                np.asarray(sim_meta["origin"]), origin_xyz
            ):
                raise ValueError("sim.npz 的 voxel_size/origin 与 exp.npz 不一致")
        for npz_name in ("ligand_dist.npz", "ligand_area.npz"):
            with np.load(density_directory / npz_name, allow_pickle=False) as target_meta:
                if tuple(np.asarray(target_meta["grid_shape_zyx"], dtype=np.int64).tolist()) != shape_zyx:
                    raise ValueError(f"{npz_name} 的 grid_shape_zyx 与 exp.npz 不一致")
                if not np.array_equal(np.asarray(target_meta["voxel_size_xyz"]), voxel_size_xyz) or not np.array_equal(
                    np.asarray(target_meta["origin_xyz"]), origin_xyz
                ):
                    raise ValueError(f"{npz_name} 的 voxel_size_xyz/origin_xyz 与 exp.npz 不一致")
        with np.load(data_root / "parse" / pdb_id / "receptor_tokens.npz", allow_pickle=False) as receptor:
            receptor_count = int(np.asarray(receptor["coords"]).shape[0])
        with np.load(data_root / "labels" / pdb_id / "atom_labels.npz", allow_pickle=False) as labels:
            if np.asarray(labels["binding_atom"]).shape != (receptor_count,):
                raise ValueError("atom_labels.npz:binding_atom 与 receptor_tokens.npz:coords 不对齐")
    except (KeyError, OSError, TypeError, ValueError) as error:
        return {"status": "invalid", "shape_zyx": None, "detail": str(error)}
    status = "eligible" if all(value >= minimum for value, minimum in zip(shape_zyx, MIN_GRID_SHAPE_ZYX)) else "short_map"
    return {"status": status, "shape_zyx": list(shape_zyx), "detail": ""}


def freeze_split(arguments: argparse.Namespace) -> None:
    """冻结日期隔离、质量过滤、资产检查以及 validation/calibration/train split。

    输入参数:
        - arguments: argparse.Namespace；包含候选 JSONL、PDB–EMDB 对照、发布时间缓存、数据根目录、输出根目录和 seed。

    选择顺序:
        - 首次 EMDB 发布时间不早于 ``RELEASE_CUTOFF`` 的 PDB 进入 ``held_out``；更早 PDB 依次通过分辨率、cc_contour 和资产审计，再按稳定 hash 排序抽取 200 个 validation、100 个 calibration，剩余进入 train。

    输出产物:
        - ``train.json``、``validation.json``、``calibration.json``、``held_out.json`` 和 ``quarantine_missing_release.json`` 保存候选记录；``pdb_audit.jsonl``、``config.json``、``summary.json`` 和 ``_COMPLETE`` 记录审计与冻结结果。
    """

    candidates_path = Path(arguments.candidates).resolve()
    pair_list_path = Path(arguments.pair_list).resolve()
    release_cache_path = Path(arguments.release_cache).resolve()
    data_root = Path(arguments.data_root).resolve()
    output_root = Path(arguments.output_root).resolve()
    if (output_root / "_COMPLETE").exists():
        raise FileExistsError(f"split 已完成，不允许覆盖：{output_root}")
    candidate_groups, pdb_to_emdb_ids = candidate_pdb_to_emdb_ids(candidates_path, pair_list_path)
    release_by_emdb = {
        normalize_emdb_id(record["emdb_id"]): record.get("map_release")
        for record in read_jsonl(release_cache_path)
    }

    held_ids: set[str] = set()
    quarantine_ids: set[str] = set()
    quality_rows_by_pdb: dict[str, list[dict[str, Any]]] = {}
    audit_records: list[dict[str, Any]] = []
    eligible_ids: list[str] = []
    for pdb_id in sorted(candidate_groups):
        release_values = [release_by_emdb.get(emdb_id) for emdb_id in pdb_to_emdb_ids[pdb_id]]
        parsed_dates = [date.fromisoformat(str(value)[:10]) for value in release_values if value]
        first_release = min(parsed_dates) if parsed_dates else None
        if first_release is None:
            quarantine_ids.add(pdb_id)
            audit_records.append(
                {"pdb_id": pdb_id, "first_map_release": None, "status": "missing_release_date"}
            )
            continue
        if first_release >= RELEASE_CUTOFF:
            held_ids.add(pdb_id)
            audit_records.append(
                {"pdb_id": pdb_id, "first_map_release": first_release.isoformat(), "status": "held_out"}
            )
            continue
        quality_rows = [
            record
            for record in candidate_groups[pdb_id]
            if record.get("map_resolution") is not None
            and record.get("cc_contour") is not None
            and float(record["map_resolution"]) < MAP_RESOLUTION_EXCLUSIVE_MAX
            and float(record["cc_contour"]) > CC_CONTOUR_EXCLUSIVE_MIN
        ]
        if not quality_rows:
            audit_records.append(
                {"pdb_id": pdb_id, "first_map_release": first_release.isoformat(), "status": "quality_rejected"}
            )
            continue
        asset_report = inspect_training_assets(data_root, pdb_id)
        audit_records.append(
            {
                "pdb_id": pdb_id,
                "first_map_release": first_release.isoformat(),
                "status": asset_report["status"],
                "shape_zyx": asset_report["shape_zyx"],
                "detail": asset_report["detail"],
                "qualifying_candidate_count": len(quality_rows),
            }
        )
        if asset_report["status"] == "eligible":
            eligible_ids.append(pdb_id)
            quality_rows_by_pdb[pdb_id] = quality_rows

    eval_order = sorted(eligible_ids, key=lambda pdb_id: stable_eval_rank(int(arguments.seed), pdb_id))
    required_eval_count = VALIDATION_PDB_COUNT + CALIBRATION_PDB_COUNT
    if len(eval_order) < required_eval_count:
        raise ValueError(f"合格 PDB 不足 {required_eval_count} 个：{len(eval_order)}")
    validation_ids = set(eval_order[:VALIDATION_PDB_COUNT])
    calibration_ids = set(eval_order[VALIDATION_PDB_COUNT:required_eval_count])
    train_ids = set(eval_order[required_eval_count:])
    split_ids = {
        "train": train_ids,
        "validation": validation_ids,
        "calibration": calibration_ids,
        "held_out": held_ids,
        "quarantine_missing_release": quarantine_ids,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for split_name, pdb_ids in split_ids.items():
        source_groups = candidate_groups if split_name in {"held_out", "quarantine_missing_release"} else quality_rows_by_pdb
        records = [record for pdb_id in sorted(pdb_ids) for record in source_groups[pdb_id]]
        atomic_write_json(output_root / f"{split_name}.json", records)
    audit_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in audit_records
    )
    atomic_write_text(output_root / "pdb_audit.jsonl", audit_text)
    config = {
        "schema_version": 1,
        "release_cutoff": RELEASE_CUTOFF.isoformat(),
        "release_rule": "first EMDB admin.key_dates.map_release < cutoff enters non-held",
        "map_resolution_exclusive_max": MAP_RESOLUTION_EXCLUSIVE_MAX,
        "cc_contour_exclusive_min": CC_CONTOUR_EXCLUSIVE_MIN,
        "minimum_grid_shape_zyx": list(MIN_GRID_SHAPE_ZYX),
        "validation_pdb_count": VALIDATION_PDB_COUNT,
        "calibration_pdb_count": CALIBRATION_PDB_COUNT,
        "seed": int(arguments.seed),
        "eval_assignment": "sha256(seed|eval|pdb_id) ascending; first 200 validation, next 100 calibration",
        "candidates_path": str(candidates_path),
        "pair_list_path": str(pair_list_path),
        "release_cache_path": str(release_cache_path),
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    status_counts: dict[str, int] = defaultdict(int)
    for record in audit_records:
        status_counts[str(record["status"])] += 1
    summary = {
        "source_pdb_count": len(candidate_groups),
        "source_candidate_count": sum(len(records) for records in candidate_groups.values()),
        "audit_status_counts": dict(sorted(status_counts.items())),
        "splits": {
            name: {
                "pdb_count": len(pdb_ids),
                "candidate_count": sum(
                    len((candidate_groups if name in {"held_out", "quarantine_missing_release"} else quality_rows_by_pdb)[pdb_id])
                    for pdb_id in pdb_ids
                ),
            }
            for name, pdb_ids in split_ids.items()
        },
    }
    atomic_write_json(output_root / "config.json", config)
    atomic_write_json(output_root / "summary.json", summary)
    atomic_write_text(output_root / "_COMPLETE", "")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    """解析 ``fetch-release-dates`` 或 ``freeze`` 子命令并调用对应入口。

    返回值:
        - None；成功时写入续传日志或冻结 split 产物，失败时保留异常供任务系统报告。
    """

    parser = argparse.ArgumentParser(description="冻结 Stage1 v3 日期质量划分。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch-release-dates")
    fetch_parser.add_argument("--candidates", required=True)
    fetch_parser.add_argument("--pair-list", required=True)
    fetch_parser.add_argument("--output", required=True)
    fetch_parser.add_argument("--workers", type=int, required=True)
    fetch_parser.add_argument("--timeout-seconds", type=float, required=True)
    fetch_parser.add_argument("--retry-count", type=int, required=True)
    fetch_parser.add_argument("--fsync-interval", type=int, required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--candidates", required=True)
    freeze_parser.add_argument("--pair-list", required=True)
    freeze_parser.add_argument("--release-cache", required=True)
    freeze_parser.add_argument("--data-root", required=True)
    freeze_parser.add_argument("--output-root", required=True)
    freeze_parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.command == "fetch-release-dates":
        fetch_release_dates(arguments)
    else:
        freeze_split(arguments)


if __name__ == "__main__":
    main()
