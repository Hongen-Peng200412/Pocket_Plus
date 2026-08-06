"""把 calibration 选定的高斯参数独立回填到既有 F1 组件森林。"""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Sequence

from src.artifacts import Stage1ArtifactPaths, load_npz_strict
from src.artifacts.paths import F_ALPHA_CENTERED_ROLES
from src.artifacts.states import PdbRunningLease, is_role_complete
from src.inference.assembly import load_pdb_id_list

from .scorer import (
    GaussScorerParameters,
    publish_gauss_fields,
    publish_li_gauss_fields,
    score_centered_nodes,
    score_f1_centered_nodes,
    score_li_centered_entries,
)


def _load_parameters(path: str | Path) -> GaussScorerParameters:
    """从正式 calibration JSON 的 `selected_parameters` 恢复四个正参数。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("selected_parameters")
    if not isinstance(values, dict):
        raise ValueError("Gauss calibration JSON 缺少 selected_parameters 映射")
    return GaussScorerParameters(
        lambda_positive=float(values["lambda_positive"]),
        lambda_negative=float(values["lambda_negative"]),
        tau_angstrom=float(values["tau_angstrom"]),
        gauss_score_min=float(values["gauss_score_min"]),
        distance_cutoff_angstrom=float(values.get("distance_cutoff_angstrom", 5.0)),
    )


def build_parser() -> argparse.ArgumentParser:
    """声明独立回填所需的固定数据身份、分片与参数文件。"""

    parser = argparse.ArgumentParser(description="回填 forest.npz 的 Gauss scorer 字段")
    parser.add_argument("--pdb-list", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--producer", default="Find_0", choices=("Find_0",))
    parser.add_argument("--split", required=True, choices=("calibration", "validation", "train"))
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--centered-role",
        default="F1_centered",
        choices=(*F_ALPHA_CENTERED_ROLES, "Li_centered"),
    )
    parser.add_argument("--evaluate-on-blob-exceed", action="store_true")
    overwrite = parser.add_mutually_exclusive_group()
    overwrite.add_argument("--force-overwrite", dest="force_overwrite", action="store_true")
    overwrite.add_argument(
        "--no-force-overwrite", dest="force_overwrite", action="store_false"
    )
    parser.set_defaults(force_overwrite=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """逐 PDB 获取现有租约，计算高斯分数并原子回填 forest。"""

    arguments = build_parser().parse_args(argv)
    if arguments.shard_count <= 0 or not 0 <= arguments.shard_index < arguments.shard_count:
        raise ValueError("shard_index 必须位于 [0, shard_count)")
    parameters = _load_parameters(arguments.calibration_json)
    pdb_ids = load_pdb_id_list(arguments.pdb_list)
    assigned = tuple(
        pdb_id
        for position, pdb_id in enumerate(pdb_ids)
        if position % arguments.shard_count == arguments.shard_index
    )
    completed: list[str] = []
    blob_exceed: list[str] = []
    pending: list[dict[str, object]] = []
    skipped_running: list[str] = []

    for pdb_id in assigned:
        paths = Stage1ArtifactPaths(
            output_root=arguments.output_root,
            stage1_model_name=arguments.producer,
            split=arguments.split,
            pdb_id=pdb_id,
        )
        if paths.blob_exceed_path.is_file():
            blob_exceed.append(pdb_id)
            if not arguments.evaluate_on_blob_exceed:
                continue
        required_roles = (
            ("Li_centered",)
            if arguments.centered_role == "Li_centered"
            else ("probability", "components", arguments.centered_role)
        )
        missing_roles = [role for role in required_roles if not is_role_complete(paths, role)]
        if missing_roles:
            pending.append({"pdb_id": pdb_id, "missing_roles": missing_roles})
            continue
        owner_token = f"{socket.gethostname()}:{os.getpid()}:{arguments.split}:{pdb_id}"
        lease = PdbRunningLease.acquire(paths, owner_token)
        if lease is None:
            skipped_running.append(pdb_id)
            continue
        with lease:
            if arguments.centered_role == "Li_centered":
                centered_path = paths.centered_npz("Li_centered")
                centered_arrays = load_npz_strict(centered_path)
                score, selected = score_li_centered_entries(
                    centered_arrays,
                    parameters,
                )
                publish_li_gauss_fields(
                    centered_path,
                    score,
                    selected,
                    force_overwrite=arguments.force_overwrite,
                )
                completed.append(pdb_id)
                continue
            forest_arrays = load_npz_strict(paths.forest_npz)
            centered_arrays = load_npz_strict(paths.centered_npz(arguments.centered_role))
            probability_arrays = load_npz_strict(paths.probability_npz)
            full_shape_zyx = tuple(
                int(value) for value in probability_arrays["probability_map"].shape
            )
            if arguments.centered_role == "F1_centered":
                score, selected = score_f1_centered_nodes(
                    forest_arrays=forest_arrays,
                    centered_arrays=centered_arrays,
                    full_shape_zyx=full_shape_zyx,
                    parameters=parameters,
                )
            else:
                score, selected = score_centered_nodes(
                    forest_arrays=forest_arrays,
                    centered_arrays=centered_arrays,
                    full_shape_zyx=full_shape_zyx,
                    centered_role=arguments.centered_role,
                    parameters=parameters,
                )
            publish_gauss_fields(
                paths.forest_npz,
                score,
                selected,
                force_overwrite=arguments.force_overwrite,
            )
        completed.append(pdb_id)

    print(
        json.dumps(
            {
                "split": arguments.split,
                "shard_index": arguments.shard_index,
                "shard_count": arguments.shard_count,
                "n_assigned": len(assigned),
                "n_completed": len(completed),
                "n_blob_exceed": len(blob_exceed),
                "n_pending": len(pending),
                "n_skipped_running": len(skipped_running),
                "blob_exceed": blob_exceed,
                "pending": pending,
                "skipped_running": skipped_running,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
