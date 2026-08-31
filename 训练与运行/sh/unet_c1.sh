#!/usr/bin/env bash

# unet_c1 正式训练入口。默认是单卡主链辅助监督；
# unet_c1_no_mainchain.sh 复用本入口并把两项主链损失权重设为 0.0。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${POCKET_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID

export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${HOME}/Feedback/${PROJECT_NAME}}"
export ADALIGAND_DATA_ROOT="${ADALIGAND_DATA_ROOT:-/storage/penghongen/AdaLigand/Ori_Data}"
export ADALIGAND_STAGE1_PREPARATION_ROOT="${ADALIGAND_STAGE1_PREPARATION_ROOT:-/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_3}"

devices="${TASK_GPUS:-1}"
nnodes="${TASK_NNODES:-1}"
variant="${UNET_C1_VARIANT:-mainchain}"
case "${variant}" in
    mainchain)
        protein_mainchain_weight="0.05"
        nucleic_mainchain_weight="0.05"
        ;;
    no_mainchain)
        protein_mainchain_weight="0.0"
        nucleic_mainchain_weight="0.0"
        ;;
    *)
        echo "[unet_c1][错误] UNET_C1_VARIANT 只允许 mainchain 或 no_mainchain。" >&2
        exit 2
        ;;
esac

if [[ "${variant}" == "mainchain" && "${devices}" != "1" ]]; then
    echo "[unet_c1][错误] mainchain 入口要求 TASK_GPUS=1，实际为 ${devices}。" >&2
    exit 2
fi
if [[ "${variant}" == "no_mainchain" && "${devices}" != "2" ]]; then
    echo "[unet_c1][错误] no_mainchain 入口要求 TASK_GPUS=2，实际为 ${devices}。" >&2
    exit 2
fi

if ((devices > 1)); then
    ddp_find_unused_parameters=true
else
    ddp_find_unused_parameters=false
fi

formal_experiment_group="AdaLigand_Stage1_pdb_centric_2/unet_c1/${variant}"
formal_tag="unet_c1_${variant}_pdb_centric_2"
run_stamp_base="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}"
formal_stamp="${run_stamp_base}_formal"
formal_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${formal_experiment_group//\//-}/${formal_tag//\//-}____${formal_stamp}"

training_overrides=(
    "+experiment=unet_c1"
    "experiment_group=${formal_experiment_group}"
    "tag=${formal_tag}"
    "init_from=null"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "train.ddp_find_unused_parameters=${ddp_find_unused_parameters}"
    "train.global_batch_size=48"
    "train.batch_size=8"
    "train.strict_global_batch_size=true"
    "train.enable_batch_size_tuning=false"
    "train.num_workers=30"
    "train.prefetch_factor=4"
    "train.max_epochs=110"
    "train.val_per_epoch=8"
    "train.optimizer.lr=1.0e-4"
    "train.scheduler.warmup_ratio=0.005"
    "train.scheduler.patience=3"
    "train.scheduler.stop_after_lr_reductions=2"
    "model.protein_mainchain_loss_weight=${protein_mainchain_weight}"
    "model.nucleic_mainchain_loss_weight=${nucleic_mainchain_weight}"
    "offline=false"
)

cd "${PROJECT_ROOT}"
echo "[unet_c1] 启动 ${variant} 正式训练：${formal_run}"
export TASK_RUN_STAMP="${formal_stamp}"
bash "${PROJECT_ROOT}/训练与运行/runtime/launch_training_python.sh" src/train.py "${training_overrides[@]}" "$@"

formal_best="${formal_run}/checkpoints/BEST.ckpt"
[[ -f "${formal_best}" ]] || {
    echo "[unet_c1][错误] 没有产生 BEST.ckpt：${formal_best}" >&2
    exit 1
}
echo "[unet_c1] ${variant} 正式训练完成。"
