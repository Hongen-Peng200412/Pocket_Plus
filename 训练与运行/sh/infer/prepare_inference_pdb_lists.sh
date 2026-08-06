#!/usr/bin/env bash
set -euo pipefail

# 本脚本以生成所有 Stage1 模型共用的 PDB 清单为主要职责，不运行模型，也不申请 GPU：
# 1. 从已经过滤的 calibration、validation、train 集合中提取唯一 PDB 名称，写到所有 Stage1 模型共用的推理根目录。
# 2. 在 Find_0 本次推理目录写一份 manifest.json，供人查看模型、配置、数据、输出路径和启用参数。
# 三份公共 PDB 清单会在手动重跑本脚本时直接更新；manifest.json 只作记录，不参与校验或运行门槛。
# 在 Pocket_Plus 服务器项目根目录提交：
#
# bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
#   --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/prepare_inference_pdb_lists.sh \
#   --resource cpu \
#   --cpus 16 \
#   --after_hold \
#   --job-name find0_prepare_inference_lists
#
# 本任务没有分片，不应增加 --array。它会直接更新三份公共 JSON 清单和人工可读 manifest；
# `--after_hold` 便于人工核对后再释放资源，省略它则任务完成后自动释放。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd -P)"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# 三份来源文件由 src/datasets/ops 产生；其中同一 PDB 可以出现多次，因此不能原样作为推理清单。
split_root="/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/split"
data_root="/storage/penghongen/AdaLigand/Ori_Data"

# 三份去重后的 PDB 清单直接放在公共推理根目录，Find_0、Find_1、unet_c1 等模型共同使用。
inference_root="/storage/penghongen/AdaLigand_stage1_inference"
formal_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477"

# checkpoint 与配置仍保留在原训练目录；这里只把路径写进 manifest.json，不复制模型文件。
run_root="/home/penghongen/My_Project/feedback_plus/logs/AdaLigand_Stage1-Find_0-CPC1/Find_0-CPC1____job321743_Find_0_CPC1_lr5e5_p2_val30_chunk2x_2gpu_m8_w1"
checkpoint="${run_root}/checkpoints/TOP_epoch_00_score_0.2843.ckpt"
config="${run_root}/config.yaml"

mkdir -p "${inference_root}" "${formal_root}"

python -u - \
    "${split_root}" \
    "${data_root}" \
    "${inference_root}" \
    "${formal_root}" \
    "${checkpoint}" \
    "${config}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.datasets.ops.stage1_box_pool import _load_split_pdb_ids


split_root, data_root, inference_root, formal_root, checkpoint, config = map(Path, sys.argv[1:])

# 每份公共文件都是 JSON 字符串数组；PDB 名称小写、去重并按名称排序。
pdb_lists: dict[str, str] = {}
for split in ("calibration", "validation", "train"):
    pdb_ids = _load_split_pdb_ids(split_root / f"{split}.json")
    destination = inference_root / f"{split}_pdb_ids.json"
    destination.write_text(
        json.dumps(list(pdb_ids), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pdb_lists[split] = str(destination)

# 这份 JSON 只是本次 Find_0 推理的人工可读记录；推理脚本不会读取它来决定是否运行。
manifest = {
    "stage1_model_name": "Find_0",
    "checkpoint": str(checkpoint),
    "config": str(config),
    "data_root": str(data_root),
    "output_root": str(formal_root / "artifacts"),
    "pdb_lists": pdb_lists,
    "selection_metric": "val_score/global/voxel_ligand_PRAUC",
    "selection_metric_value": 0.6754766702651978,
    "enabled_configuration": {
        "calibration_global_shard_count": 2,
        "validation_train_global_shard_count": 16,
        "window_batch_size": 8,
        "centered_batch_size": 8,
        "cache_max_bytes": 107374182400,
        "min_voxels": 10,
        "max_voxels": 2046,
        "threshold_denominator": 32768,
        "main_centered_role": "F1_centered",
        "clg_centered_in_main_run": False,
        "selector_enabled": False,
    },
}
(formal_root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

echo "[Find_0 inputs] 公共 PDB 清单：${inference_root}"
echo "[Find_0 inputs] 本次推理记录：${formal_root}/manifest.json"
