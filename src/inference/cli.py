# -*- coding: utf-8 -*-
"""Stage1 V3 calibration 与正式推理命令入口.

命令只接受显式配置文件和显式部署路径. ``calibrate`` 生成 producer 级冻结
参数; ``run`` 读取该文件, 生成 F1 basic、F3 centered 和完整评估产物.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from .artifacts import (
    Stage1ArtifactPaths,
    load_stage1_npz,
    publish_stage1_artifact,
    publish_stage1_json,
    publish_stage1_jsonl,
)
from .calibration import calibrate_semantic_thresholds, tune_centered_selection
from .checkpoint import load_stage1_wrapper
from .blobs import publish_probability_blobs
from .evaluation import (
    aggregate_stage1_metrics,
    evaluate_centered_pdb,
    evaluation_arrays,
    load_occurrence_voxels,
)
from .pipeline import (
    produce_centered_role,
    produce_probability_map,
    score_and_publish_centered,
)


# ================================================================================================


def main() -> None:
    """解析命令并执行完整 calibration 或冻结参数推理流程."""

    parser = argparse.ArgumentParser(description="AdaLigand Stage1 V3 inference")
    parser.add_argument("command", choices=("calibrate", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--pdb-list", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--producer", choices=("unet_c1", "Find_0", "Find_1", "Find_2"), required=True)
    parser.add_argument("--calibration", required=False)
    parser.add_argument(
        "--allow-current-workspace-code",
        choices=("true", "false"),
        required=True,
    )
    arguments = parser.parse_args()
    if arguments.command == "run" and arguments.calibration is None:
        parser.error("run 必须显式传入 --calibration.")

    config = OmegaConf.load(arguments.config)
    OmegaConf.resolve(config)
    pdb_ids = tuple(
        line.strip().lower()
        for line in Path(arguments.pdb_list).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    wrapper, training_config = load_stage1_wrapper(
        checkpoint_path=arguments.checkpoint,
        resolved_config_path=arguments.resolved_config,
        map_location="cpu",
        allow_current_workspace_code=arguments.allow_current_workspace_code == "true",
    )
    from src.datasets.stage1_requests import ResolvedStage1Crop

    seed_requests = [
        ResolvedStage1Crop(
            pdb_id=pdb_id,
            box_start_zyx=(0, 0, 0),
            require_targets=False,
            role="sliding",
        )
        for pdb_id in pdb_ids
    ]
    dataset = hydra.utils.instantiate(
        training_config.dataset,
        split_file=seed_requests,
        mode="full_map",
        box_pool_root=None,
        enable_random_rotation=False,
    )
    collator = dataset.collate_fn
    device = str(config.device)
    wrapper.to(torch.device(device))
    producer = str(arguments.producer)
    output_root = Path(arguments.output_root)

    probability_by_pdb: dict[str, dict[str, np.ndarray]] = {}
    for pdb_id in pdb_ids:
        paths = Stage1ArtifactPaths(output_root, producer, arguments.split, pdb_id)
        if bool(config.overwrite) or not paths.complete("probability").is_file():
            probability_by_pdb[pdb_id] = produce_probability_map(
                paths=paths,
                dataset=dataset,
                collator=collator,
                wrapper=wrapper,
                device=device,
                window_config=config.window,
            )
        else:
            probability_by_pdb[pdb_id] = load_stage1_npz(
                paths.artifact("probability"),
                ("probability_map", "full_shape_zyx", "origin_xyz", "voxel_size_xyz"),
            )

    if arguments.command == "calibrate":
        probability_and_target: list[tuple[np.ndarray, np.ndarray]] = []
        for pdb_id in pdb_ids:
            union_mask = np.load(
                Path(dataset.root) / "density" / pdb_id / "union_mask.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[0]
            probability_and_target.append(
                (probability_by_pdb[pdb_id]["probability_map"], union_mask)
            )
        semantic = calibrate_semantic_thresholds(
            probability_and_target,
            denominator=int(config.calibration.semantic_denominator),
            betas=(1.0, 3.0),
        )
        calibration_payload: dict[str, object] = {
            "producer": producer,
            "semantic": semantic,
            "roles": {},
        }
    else:
        calibration_payload = OmegaConf.to_container(
            OmegaConf.load(arguments.calibration),
            resolve=True,
        )

    role_evaluations: dict[str, list] = {}
    for role_name, role_config in config.roles.items():
        blob_role = str(role_config.blob_role)
        centered_role = str(role_config.centered_role)
        if arguments.command == "calibrate":
            semantic_role = str(role_config.semantic_role)
            semantic_threshold = float(
                calibration_payload["semantic"]["thresholds"][semantic_role]["value"]
            )
            generation_min_voxels = min(
                int(value) for value in role_config.min_voxel_values
            )
            centered_by_pdb: dict[str, dict[str, np.ndarray]] = {}
            with ThreadPoolExecutor(
                max_workers=int(config.blob_workers),
                thread_name_prefix="stage1-blobs",
            ) as blob_executor:
                blob_futures = {
                    pdb_id: blob_executor.submit(
                        publish_probability_blobs,
                        Stage1ArtifactPaths(output_root, producer, arguments.split, pdb_id),
                        blob_role,
                        semantic_threshold,
                    )
                    for pdb_id in pdb_ids
                }
                for pdb_id in pdb_ids:
                    paths = Stage1ArtifactPaths(output_root, producer, arguments.split, pdb_id)
                    arrays = produce_centered_role(
                        paths=paths,
                        dataset=dataset,
                        collator=collator,
                        wrapper=wrapper,
                        device=device,
                        blobs=blob_futures[pdb_id].result(),
                        centered_role=centered_role,
                        min_voxels=generation_min_voxels,
                        centered_config=role_config.centered,
                        blob_limit=(
                            None if role_config.blob_limit is None else int(role_config.blob_limit)
                        ),
                        enforce_blob_limit=(
                            None
                            if role_config.enforce_blob_limit_calibration is None
                            else bool(role_config.enforce_blob_limit_calibration)
                        ),
                        publish_complete=False,
                    )
                    if arrays is not None:
                        centered_by_pdb[pdb_id] = arrays

            ground_truth = {
                pdb_id: load_occurrence_voxels(
                    Path(dataset.root) / "density" / pdb_id / "ligand_area.npz"
                )
                for pdb_id in centered_by_pdb
            }
            score_mode = str(role_config.score_mode[producer])
            if score_mode == "source_mean":
                score_parameter_rows = ({},)
            else:
                grid = role_config.score_parameter_grid
                score_parameter_rows = tuple(
                    {
                        "tau_angstrom": float(tau),
                        "lambda_positive": float(positive),
                        "lambda_negative": float(negative),
                    }
                    for tau, positive, negative in itertools.product(
                        grid.tau_angstrom,
                        grid.lambda_positive,
                        grid.lambda_negative,
                    )
                )
            best = tune_centered_selection(
                centered_by_pdb=centered_by_pdb,
                ground_truth_by_pdb=ground_truth,
                score_mode=score_mode,
                score_parameter_rows=score_parameter_rows,
                min_voxel_values=role_config.min_voxel_values,
                objective_beta=float(role_config.objective_beta),
                coverage_thresholds=config.evaluation.coverage_thresholds,
                topk_values=config.evaluation.topk_values,
            )
            calibration_payload["roles"][role_name] = {
                "source_threshold": semantic_threshold,
                "selection": best,
            }
            selection = best
        else:
            role_calibration = calibration_payload["roles"][role_name]
            semantic_threshold = float(role_calibration["source_threshold"])
            selection = role_calibration["selection"]
            with ThreadPoolExecutor(
                max_workers=int(config.blob_workers),
                thread_name_prefix="stage1-blobs",
            ) as blob_executor:
                blob_futures = {
                    pdb_id: blob_executor.submit(
                        publish_probability_blobs,
                        Stage1ArtifactPaths(output_root, producer, arguments.split, pdb_id),
                        blob_role,
                        semantic_threshold,
                    )
                    for pdb_id in pdb_ids
                }
                for pdb_id in pdb_ids:
                    paths = Stage1ArtifactPaths(output_root, producer, arguments.split, pdb_id)
                    produce_centered_role(
                        paths=paths,
                        dataset=dataset,
                        collator=collator,
                        wrapper=wrapper,
                        device=device,
                        blobs=blob_futures[pdb_id].result(),
                        centered_role=centered_role,
                        min_voxels=int(selection["min_voxels"]),
                        centered_config=role_config.centered,
                        blob_limit=(
                            None if role_config.blob_limit is None else int(role_config.blob_limit)
                        ),
                        enforce_blob_limit=(
                            None
                            if role_config.enforce_blob_limit_run is None
                            else bool(role_config.enforce_blob_limit_run)
                        ),
                        publish_complete=True,
                    )

        evaluations = []
        jsonl_rows = []
        for pdb_id in pdb_ids:
            paths = Stage1ArtifactPaths(output_root, producer, arguments.split, pdb_id)
            if not paths.artifact(centered_role).is_file():
                continue
            selected = score_and_publish_centered(
                paths=paths,
                centered_role=centered_role,
                score_mode=str(selection["score_mode"]),
                score_parameters=selection["score_parameters"],
                score_threshold=float(selection["score_threshold"]),
                min_voxels=int(selection["min_voxels"]),
            )
            occurrence_id, occurrence_rows, full_shape = load_occurrence_voxels(
                Path(dataset.root) / "density" / pdb_id / "ligand_area.npz"
            )
            evaluation = evaluate_centered_pdb(
                pdb_id=pdb_id,
                centered=selected,
                occurrence_id=occurrence_id,
                occurrence_voxel_zyx=occurrence_rows,
                full_shape_zyx=full_shape,
                coverage_thresholds=config.evaluation.coverage_thresholds,
                topk_values=config.evaluation.topk_values,
            )
            evaluations.append(evaluation)
            publish_stage1_artifact(
                paths.evaluation(centered_role, "npz"),
                evaluation_arrays(evaluation),
                None,
            )
            single_metrics = aggregate_stage1_metrics(
                (evaluation,),
                coverage_thresholds=config.evaluation.coverage_thresholds,
                topk_values=config.evaluation.topk_values,
            )
            jsonl_rows.append({"pdb_id": pdb_id, **single_metrics})
        global_metrics = aggregate_stage1_metrics(
            evaluations,
            coverage_thresholds=config.evaluation.coverage_thresholds,
            topk_values=config.evaluation.topk_values,
        )
        evaluation_root = output_root / producer / arguments.split / "evaluation"
        publish_stage1_jsonl(
            evaluation_root / f"{centered_role}.jsonl",
            jsonl_rows,
        )
        publish_stage1_json(
            evaluation_root / f"{centered_role}.metrics.json",
            global_metrics,
        )
        role_evaluations[role_name] = evaluations

    if arguments.command == "calibrate":
        calibration_path = (
            output_root / producer / "calibration" / "stage1_v3.json"
        )
        publish_stage1_json(calibration_path, calibration_payload)


if __name__ == "__main__":
    main()
