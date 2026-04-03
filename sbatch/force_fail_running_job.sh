#!/bin/bash
# ============================================================
# Force one running Pocket_Plus training job to fail right now
# ============================================================
# Usage:
#   bash sbatch/force_fail_running_job.sh <SLURM_JOB_ID> [SIGNAL]
#
# Example:
#   bash sbatch/force_fail_running_job.sh 123456
#   bash sbatch/force_fail_running_job.sh 123456 ABRT
#
# Behavior:
# - Opens an overlapping helper step inside the target allocation
# - Finds only python processes whose environment contains SLURM_JOB_ID=<jobid>
# - Further narrows to commands containing Pocket_Plus/src/train.py
# - Sends the requested signal only to those matched processes
#
# Notes:
# - Default signal is ABRT so the failure is obvious in logs
# - Pair with auto_retry_try_lock.sh if you want immediate retry
# ============================================================

set -euo pipefail

JOB_ID="${1:-}"
SIGNAL_NAME="${2:-ABRT}"

if [[ -z "${JOB_ID}" ]]; then
    echo "Usage: bash sbatch/force_fail_running_job.sh <SLURM_JOB_ID> [SIGNAL]"
    exit 1
fi

if ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "Error: SLURM_JOB_ID must be numeric, got '${JOB_ID}'."
    exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
    echo "Error: squeue is required but was not found in PATH."
    exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
    echo "Error: srun is required but was not found in PATH."
    exit 1
fi

NODE_COUNT="$(squeue -h -j "${JOB_ID}" -o "%D" | head -n 1 | tr -d '[:space:]')"
JOB_STATE="$(squeue -h -j "${JOB_ID}" -o "%T" | head -n 1 | tr -d '[:space:]')"

if [[ -z "${NODE_COUNT}" || -z "${JOB_STATE}" ]]; then
    echo "Error: job '${JOB_ID}' was not found in squeue."
    exit 1
fi

if ! [[ "${NODE_COUNT}" =~ ^[0-9]+$ ]] || [[ "${NODE_COUNT}" -lt 1 ]]; then
    echo "Error: invalid node count '${NODE_COUNT}' for job '${JOB_ID}'."
    exit 1
fi

echo "[Info] job_id=${JOB_ID}"
echo "[Info] job_state=${JOB_STATE}"
echo "[Info] node_count=${NODE_COUNT}"
echo "[Info] signal=${SIGNAL_NAME}"
echo "[Info] opening an overlapping helper step inside the target allocation"

srun \
    --jobid="${JOB_ID}" \
    --overlap \
    --nodes="${NODE_COUNT}" \
    --ntasks="${NODE_COUNT}" \
    --ntasks-per-node=1 \
    bash -lc '
set -euo pipefail

TARGET_JOB_ID="'"${JOB_ID}"'"
TARGET_SIGNAL="'"${SIGNAL_NAME}"'"
MATCHED=0

mapfile -t CANDIDATES < <(pgrep -f "Pocket_Plus/src/train.py" || true)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
    echo "[Host $(hostname)] no train.py process found on this node"
    exit 0
fi

for PID in "${CANDIDATES[@]}"; do
    if [[ ! -r "/proc/${PID}/environ" ]]; then
        continue
    fi

    if ! tr "\0" "\n" < "/proc/${PID}/environ" | grep -qx "SLURM_JOB_ID=${TARGET_JOB_ID}"; then
        continue
    fi

    CMDLINE="$(tr "\0" " " < "/proc/${PID}/cmdline" 2>/dev/null || true)"
    echo "[Host $(hostname)] sending SIG${TARGET_SIGNAL} to pid=${PID}"
    echo "[Host $(hostname)] cmd=${CMDLINE}"
    kill "-${TARGET_SIGNAL}" "${PID}"
    MATCHED=1
done

if [[ "${MATCHED}" -eq 0 ]]; then
    echo "[Host $(hostname)] matched no Pocket_Plus train.py process for SLURM_JOB_ID=${TARGET_JOB_ID}"
fi
'
