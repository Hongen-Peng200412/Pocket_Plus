#!/usr/bin/env bash

# 历史 Find_1 CPC1 从字面 last.ckpt 完整续训；不启用新版分组梯度裁剪。
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
nnodes="${TASK_NNODES:-2}"
task_cpu_count="${TASK_CPU_COUNT:-31}"
num_workers="${FIND1_NUM_WORKERS:-$((task_cpu_count - 1))}"
source_resume_checkpoint="${FIND1_RESUME_CHECKPOINT:-/home/penghongen/Feedback/Pocket_Plus/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____Find_1_job351295_20260823T152130_a2_CPC1/checkpoints/last.ckpt}"
source_resume_sha256=d633de0555f5ad46e76a36bfd83d5c4b7ab5a8cb09652918c2d6dfff225afd4c
experiment_group="AdaLigand_Stage1_resume/Find_1/CPC1"
tag="Find_1/resume_last_job351295"
run_stamp="${TASK_RUN_STAMP:-$(date '+%Y%m%dT%H%M%S')}_CPC1"
formal_run="${EXPERIMENT_FEEDBACK_ROOT}/logs/${experiment_group//\//-}/${tag//\//-}____${run_stamp}"

[[ -f "${source_resume_checkpoint}" ]] || {
    echo "[Find_1][错误] 续训 checkpoint 不存在：${source_resume_checkpoint}" >&2
    exit 1
}

# 新运行目录不能直接恢复旧 dirpath 的 ModelCheckpoint 状态。每个节点的 shell
# 共享同一个存储路径，仅 node rank 0 复制历史 top-k 并迁移回调路径，其他节点等待产物。
resume_checkpoint_dir="${formal_run}/checkpoints"
resume_checkpoint="${resume_checkpoint_dir}/resume_state.ckpt"
resume_manifest="${resume_checkpoint_dir}/resume_state_manifest.json"
resume_failure="${resume_checkpoint_dir}/resume_state.failed"
node_rank="${TASK_DDP_NODE_RANK:-0}"
mkdir -p "${resume_checkpoint_dir}"
if [[ "${node_rank}" == "0" ]]; then
    rm -f -- "${resume_failure}"
    if ! python "${PROJECT_ROOT}/ops/find1_historical_resume/rebase_checkpoint.py" \
        --source "${source_resume_checkpoint}" \
        --destination "${resume_checkpoint_dir}" \
        --expected-source-sha256 "${source_resume_sha256}"; then
        touch -- "${resume_failure}"
        exit 1
    fi
else
    for _ in $(seq 1 360); do
        [[ -f "${resume_failure}" ]] && {
            echo "[Find_1][错误] node rank 0 迁移 checkpoint 失败。" >&2
            exit 1
        }
        [[ -f "${resume_checkpoint}" && -f "${resume_manifest}" ]] && break
        sleep 5
    done
fi
[[ -f "${resume_checkpoint}" && -f "${resume_manifest}" ]] || {
    echo "[Find_1][错误] 没有得到迁移后的完整 checkpoint：${resume_checkpoint}" >&2
    exit 1
}
(( num_workers >= 1 )) || {
    echo "[Find_1][错误] Find_1 续训至少需要一个 DataLoader worker。" >&2
    exit 1
}

overrides=(
    "+experiment=CPC1/Find_1"
    "experiment_group=${experiment_group}"
    "tag=${tag}"
    "init_from=null"
    "resume_from_checkpoint=${resume_checkpoint}"
    "project_name=AdaLigand_Stage1"
    "train.devices=${devices}"
    "train.nnodes=${nnodes}"
    "train.ddp_find_unused_parameters=true"
    "train.ddp_timeout_seconds=86400"
    "train.global_batch_size=48"
    "train.batch_size=6"
    "train.strict_global_batch_size=true"
    "train.enable_batch_size_tuning=false"
    "train.num_workers=${num_workers}"
    "train.resume_skip_train_batches=45150"
    "train.resume_skip_epoch=0"
    "+train.resume_after_completed_validation=true"
    "train.prefetch_factor=4"
    "train.max_epochs=20"
    "train.val_per_epoch=40"
    "train.optimizer.lr=5.0e-5"
    "model.backbone.density_cube_cfg.chunk_size=4096"
    "model.backbone.real_density_cube_cfg.chunk_size=8192"
    "model.backbone.real_atom_density_cube_size=9"
    "model.backbone.real_density_cube_cfg.cube_size=9"
    "train.scheduler.warmup_ratio=0.005"
    "train.scheduler.patience=3"
    "train.scheduler.stop_after_lr_reductions=3"
    "train.gradient_clip_val=0.5"
    "offline=false"
)

cd "${PROJECT_ROOT}"
echo "[Find_1] 启动历史 CPC1 完整续训：${formal_run}"
echo "[Find_1] source checkpoint=${source_resume_checkpoint}"
echo "[Find_1] rebased checkpoint=${resume_checkpoint}"
echo "[Find_1] rebase manifest=${resume_manifest}"
echo "[Find_1][边界] 源 checkpoint 不含 DataLoader worker 的 Python/NumPy RNG 状态；首轮剩余 BOX 身份连续，但随机旋转不是旧 worker 流的逐位续接。"
echo "[Find_1] 每节点 CPU=${task_cpu_count}，每个 rank workers=${num_workers}，首轮跳过 batch=45150，DDP collective timeout=86400 秒"
export TASK_RUN_STAMP="${run_stamp}"
bash "${PROJECT_ROOT}/训练与运行/runtime/launch_training_python.sh" src/train.py "${overrides[@]}" "$@"

formal_best="${formal_run}/checkpoints/BEST.ckpt"
[[ -f "${formal_best}" ]] || {
    echo "[Find_1][错误] CPC1 没有产生 BEST.ckpt：${formal_best}" >&2
    exit 1
}
echo "[Find_1] 历史 CPC1 续训完成。"
