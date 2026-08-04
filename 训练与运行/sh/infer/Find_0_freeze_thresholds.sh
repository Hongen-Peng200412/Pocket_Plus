#!/usr/bin/env bash
set -euo pipefail

# 本脚本在两个 calibration probability 分片全部完成后运行；它只使用 CPU 和已落盘的概率图。
# 在 Pocket_Plus 服务器项目根目录提交：
#
# bash /home/penghongen/My_Project/Pocket_Plus/训练与运行/submit_task.sh \
#   --sh /home/penghongen/My_Project/Pocket_Plus/训练与运行/sh/infer/Find_0_freeze_thresholds.sh \
#   --resource cpu \
#   --cpus 16 \
#   --after_hold \
#   --job-name find0_freeze_thresholds
#
# 本任务没有分片，不应增加 --array。`--after_hold` 便于人工核对阈值文件后再释放 CPU allocation；
# 省略 --after_hold 时，阈值计算完成后会自动结束并释放资源。
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

data_root="/storage/penghongen/AdaLigand/Ori_Data"  # occurrence 真值来自 A–G 正式 ligand_area.npz。
inference_root="/storage/penghongen/AdaLigand_stage1_inference"
formal_root="${inference_root}/Find_0-CPC1-ligand_PRAUC_0.675477"
pdb_list="${inference_root}/calibration_pdb_ids.json"  # 所有模型共用；必须覆盖完整 calibration 集合，不能传入单个分片。
output_root="${formal_root}/artifacts"
min_voxels=10       # 连通组件少于 10 个体素时不作为实例候选；该冻结值由后续全部 F1 推理复用。
max_voxels=2046     # 连通组件体素数上限，沿用冻结的 Stage1 calibration 契约。
denominator=32768   # 阈值网格包含 0/32768 至 32768/32768，共 32769 个位置。

[[ -f "${pdb_list}" ]] || { echo "缺少公共 calibration PDB 清单" >&2; exit 2; }

cd "${PROJECT_ROOT}"
python -u -m src.inference.cli freeze-thresholds \
    --producer Find_0 \
    --pdb-list "${pdb_list}" \
    --data-root "${data_root}" \
    --output-root "${output_root}" \
    --min-voxels "${min_voxels}" \
    --max-voxels "${max_voxels}" \
    --denominator "${denominator}"
