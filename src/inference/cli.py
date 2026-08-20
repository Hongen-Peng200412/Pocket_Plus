# -*- coding: utf-8 -*-
"""解析 Stage1 V3 显式命令并构造 checkpoint 同源运行环境."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from .checkpoint import load_stage1_wrapper
from .workflow import run_calibration_workflow, run_frozen_workflow


def _sha256_file(path: Path) -> str:
    """用 Python 3.10 可用的分块读取计算一个文件的 SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ================================================================================================


def main() -> None:
    """把显式路径解析为一个 calibration 或冻结参数 workflow 调用."""

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--resolved-config", required=True)
    common.add_argument("--pdb-list", required=True)
    common.add_argument("--output-root", required=True)
    common.add_argument(
        "--producer",
        choices=("unet_c1", "Find_0", "Find_1", "Find_2"),
        required=True,
    )
    common.add_argument(
        "--model-code-source",
        choices=("current_workspace", "training_snapshot"),
        required=True,
    )
    parser = argparse.ArgumentParser(description="AdaLigand Stage1 V3 inference")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = commands.add_parser("calibrate", parents=(common,))
    calibrate_parser.add_argument("--split", choices=("calibration",), required=True)
    run_parser = commands.add_parser("run", parents=(common,))
    run_parser.add_argument("--split", choices=("validation", "train"), required=True)
    run_parser.add_argument("--calibration", required=True)
    arguments = parser.parse_args()

    config_path = Path(arguments.config)
    checkpoint_path = Path(arguments.checkpoint)
    resolved_config_path = Path(arguments.resolved_config)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    pdb_ids = tuple(
        line.strip().lower()
        for line in Path(arguments.pdb_list).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(pdb_ids) != len(set(pdb_ids)):
        raise ValueError("PDB 清单包含重复标识; 同一数据划分中的 PDB 必须唯一.")
    wrapper, training_config = load_stage1_wrapper(
        checkpoint_path=checkpoint_path,
        resolved_config_path=resolved_config_path,
        map_location="cpu",
        model_code_source=str(arguments.model_code_source),
    )
    # 快照模型恢复结束后再导入当前 V3 Dataset, 避免当前辅助模块污染快照导入链.
    from src.datasets.stage1_requests import ResolvedStage1Crop
    import src.datasets.stage1_dataset as _current_stage1_dataset

    if str(training_config.dataset.stage1_model_name) != str(arguments.producer):
        raise ValueError(
            "--producer 与 resolved config 的 dataset.stage1_model_name 不一致."
        )
    dataset_arguments = OmegaConf.to_container(training_config.dataset, resolve=True)
    dataset_arguments.pop("_target_", None)
    dataset_arguments.update(
        split_file=[
            ResolvedStage1Crop(
                pdb_id=pdb_id,
                box_start_zyx=(0, 0, 0),
                require_targets=False,
                role="sliding",
            )
            for pdb_id in pdb_ids
        ],
        mode="full_map",
        box_pool_root=None,
        enable_random_rotation=False,
    )
    dataset = _current_stage1_dataset.Stage1Dataset(**dataset_arguments)
    wrapper.to(torch.device(str(config.device)))
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    artifact_identity = {
        "schema": "stage1_v3",
        "producer": str(arguments.producer),
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": hashlib.sha256(
            resolved_config_path.read_bytes()
        ).hexdigest(),
        "inference_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "semantic_denominator": int(config.calibration.semantic_denominator),
        "model_code_source": str(arguments.model_code_source),
    }
    workflow_arguments = {
        "config": config,
        "dataset": dataset,
        "collator": dataset.collate_fn,
        "wrapper": wrapper,
        "device": str(config.device),
        "producer": str(arguments.producer),
        "split": str(arguments.split),
        "pdb_ids": pdb_ids,
        "output_root": Path(arguments.output_root),
        "artifact_identity": artifact_identity,
    }
    if arguments.command == "calibrate":
        run_calibration_workflow(**workflow_arguments)
    else:
        calibration_path = Path(arguments.calibration)
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_complete = json.loads(
            (calibration_path.parent / "_COMPLETE").read_text(encoding="utf-8")
        )
        run_frozen_workflow(
            calibration_payload=calibration_payload,
            calibration_complete=calibration_complete,
            **workflow_arguments,
        )


if __name__ == "__main__":
    main()
