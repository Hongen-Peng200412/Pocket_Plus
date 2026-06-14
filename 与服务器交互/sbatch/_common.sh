#!/bin/bash
# ============================================================
# 公共环境设置脚本
# ============================================================
# 所有 sbatch 脚本在训练前 source 此文件
# 用法: source "$(dirname "$0")/../_common.sh"
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/penghongen/My_Project"
POCKET_PLUS_SHARED_PROJECT_ROOT="${PROJECT_ROOT}"
CONDA_BASE="/home/penghongen/anaconda3"
CONDA_ENV_NAME="${Pocket_Plus_CONDA_ENV_NAME:-Pocket_Plus_centos7_cu121_allgpu}"
export EXPERIMENT_FEEDBACK_ROOT="${EXPERIMENT_FEEDBACK_ROOT:-${POCKET_PLUS_SHARED_PROJECT_ROOT}/feedback_plus}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u
unset BOOST_ROOT

# --- Locale (防止中文路径乱码) ---
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="/usr/lib64:${CUDA_HOME}/lib64"
fi
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${CONDA_PREFIX}/lib/libgcc_s.so.1"

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

_pocket_plus_is_enabled() {
    case "${1:-0}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

_setup_pocket_plus_local_cache() {
    if ! _pocket_plus_is_enabled "${POCKET_PLUS_LOCAL_CACHE_ENABLED:-0}"; then
        return 0
    fi

    local shared_project_root="${POCKET_PLUS_SHARED_PROJECT_ROOT}"
    local shared_repo="${POCKET_PLUS_LOCAL_CACHE_SHARED_REPO:-${shared_project_root}/Pocket_Plus}"
    local env_pack="${POCKET_PLUS_LOCAL_CACHE_ENV_PACK:-${shared_project_root}/env_packs/${CONDA_ENV_NAME}.tar.gz}"
    local local_base="${POCKET_PLUS_LOCAL_CACHE_BASE:-${SLURM_TMPDIR:-/tmp/${USER}/slurm_${SLURM_JOB_ID}}}"
    local local_project_root="${local_base}/My_Project"
    local local_repo="${local_project_root}/Pocket_Plus"
    local local_env="${local_base}/conda_env"
    local min_kb="${POCKET_PLUS_LOCAL_CACHE_MIN_KB:-20971520}"
    local avail_kb

    echo "[LocalCache] Enabled."
    echo "[LocalCache] host=$(hostname)"
    echo "[LocalCache] local_base=${local_base}"
    echo "[LocalCache] env_pack=${env_pack}"
    echo "[LocalCache] shared_repo=${shared_repo}"

    if [ ! -f "${env_pack}" ]; then
        echo "[LocalCache][ERROR] Conda env pack not found: ${env_pack}"
        exit 71
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        echo "[LocalCache][ERROR] rsync is required for local repo cache."
        exit 72
    fi

    mkdir -p "${local_base}" "${local_project_root}"
    touch "${local_project_root}/.project-root"
    df -h "${local_base}" || true
    avail_kb="$(df -Pk "${local_base}" | awk 'NR==2{print $4}')"
    if [ "${avail_kb}" -lt "${min_kb}" ]; then
        echo "[LocalCache][ERROR] local disk has ${avail_kb} KB available, need at least ${min_kb} KB."
        exit 70
    fi

    if [ ! -x "${local_env}/bin/python" ]; then
        echo "[LocalCache] Unpacking conda env to ${local_env}..."
        rm -rf "${local_env}"
        mkdir -p "${local_env}"
        tar -xzf "${env_pack}" -C "${local_env}"
        "${local_env}/bin/conda-unpack" || true
    else
        echo "[LocalCache] Reusing existing local env: ${local_env}"
    fi

    echo "[LocalCache] Syncing repo to ${local_repo}..."
    mkdir -p "${local_repo}"
    rsync -a --delete \
        --exclude ".git" \
        --exclude "__pycache__" \
        --exclude "*.pyc" \
        --exclude ".pytest_cache" \
        --exclude "feedback_plus" \
        "${shared_repo}/" "${local_repo}/"

    export POCKET_PLUS_LOCAL_CACHE_ACTIVE=1
    export POCKET_PLUS_LOCAL_CACHE_BASE="${local_base}"
    export POCKET_PLUS_LOCAL_CACHE_ENV="${local_env}"
    export POCKET_PLUS_LOCAL_CACHE_REPO="${local_repo}"

    PROJECT_ROOT="${local_project_root}"
    export PROJECT_ROOT
    export CONDA_PREFIX="${local_env}"
    export PATH="${local_env}/bin:${PATH}"
    export LD_PRELOAD="${local_env}/lib/libstdc++.so.6:${local_env}/lib/libgcc_s.so.1"
    export PYTHONPYCACHEPREFIX="${local_base}/pycache"
}

pocket_plus_cleanup_local_cache() {
    if ! _pocket_plus_is_enabled "${POCKET_PLUS_LOCAL_CACHE_ACTIVE:-0}"; then
        return 0
    fi
    if ! _pocket_plus_is_enabled "${POCKET_PLUS_LOCAL_CACHE_CLEANUP_ON_EXIT:-1}"; then
        echo "[LocalCache] Cleanup disabled by POCKET_PLUS_LOCAL_CACHE_CLEANUP_ON_EXIT."
        return 0
    fi

    local local_base="${POCKET_PLUS_LOCAL_CACHE_BASE:-}"
    local resolved_base=""
    local resolved_slurm_tmp=""
    local safe_to_remove=0

    if [ -z "${local_base}" ] || [ "${local_base}" = "/" ]; then
        echo "[LocalCache][WARN] Skip cleanup because local cache base is unsafe: '${local_base}'"
        return 0
    fi

    resolved_base="$(readlink -f "${local_base}" 2>/dev/null || true)"
    if [ -z "${resolved_base}" ] || [ "${resolved_base}" = "/" ]; then
        echo "[LocalCache][WARN] Skip cleanup because resolved cache base is unsafe: '${resolved_base}'"
        return 0
    fi

    case "${resolved_base}" in
        "/tmp/${USER}/slurm_${SLURM_JOB_ID}")
            safe_to_remove=1
            ;;
    esac

    if [ -n "${SLURM_TMPDIR:-}" ]; then
        resolved_slurm_tmp="$(readlink -f "${SLURM_TMPDIR}" 2>/dev/null || true)"
        if [ -n "${resolved_slurm_tmp}" ] && [ "${resolved_base}" = "${resolved_slurm_tmp}" ]; then
            safe_to_remove=1
        fi
    fi

    if [ "${safe_to_remove}" -ne 1 ]; then
        echo "[LocalCache][WARN] Skip cleanup for non-standard cache base: ${resolved_base}"
        echo "[LocalCache][WARN] Remove it manually if it is safe."
        return 0
    fi

    echo "[LocalCache] Cleaning local cache: ${resolved_base}"
    rm -rf "${resolved_base}"
}

_setup_pocket_plus_local_cache

export PROJECT_ROOT
cd "${PROJECT_ROOT}" || { echo "Error: Could not cd to ${PROJECT_ROOT}"; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/Pocket_Plus:${PYTHONPATH:-}"

echo "Job ${SLURM_JOB_ID} allocated on nodes: ${SLURM_JOB_NODELIST}"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Python: $(command -v python)"
echo "Conda Env: ${CONDA_ENV_NAME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "LD_PRELOAD: ${LD_PRELOAD}"
echo "glibc: $(ldd --version 2>/dev/null | head -n 1)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<empty>}"
