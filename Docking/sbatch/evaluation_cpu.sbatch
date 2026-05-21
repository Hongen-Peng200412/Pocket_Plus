#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_eval_%j.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_eval_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J DockEval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

EVAL_RUN_ID="${EVAL_RUN_ID:-20260520_eval_full80_partial}"
DOCKING_RUN_IDS="${DOCKING_RUN_IDS:-20260519_full80_top1_nstruct2:20260520_full80_top1_pending13_scipy_required}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
ALLOWED_ROOT="/home/penghongen/分子对接尝试"
EVAL_ROOT="${ALLOWED_ROOT}/evaluation_runs/${EVAL_RUN_ID}"

mkdir -p "${EVAL_ROOT}/slurm"
cd /home/penghongen/My_Project/Pocket_Plus

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu

LOCK_PRE="${EVAL_ROOT}/slurm/pre_lock_cpu_${SLURM_JOB_ID}"
LOCK_AFTER="${EVAL_ROOT}/slurm/after_lock_cpu_${SLURM_JOB_ID}"
touch "${LOCK_PRE}" "${LOCK_AFTER}"
echo "[Status] pre lock: ${LOCK_PRE}"
echo "[Status] after lock: ${LOCK_AFTER}"
echo "[Wait] Delete pre lock to start."

while [ -f "${LOCK_PRE}" ]; do
    sleep 30
done

COMMAND=(python Docking/evaluation/run_evaluation.py --eval-run-id "${EVAL_RUN_ID}")
IFS=':' read -ra RUN_IDS <<< "${DOCKING_RUN_IDS}"
for RUN_ID in "${RUN_IDS[@]}"; do
    if [ -n "${RUN_ID}" ]; then
        COMMAND+=(--docking-run-id "${RUN_ID}")
    fi
done
if [ -n "${MAX_SAMPLES}" ]; then
    COMMAND+=(--max-samples "${MAX_SAMPLES}")
fi

echo "[Start] ${COMMAND[*]}"
if srun "${COMMAND[@]}"; then
    echo "[Success] Evaluation finished."
    rm -f "${LOCK_AFTER}"
else
    echo "[Error] Evaluation failed. Inspect ${EVAL_ROOT}/slurm and summary/warnings."
    echo "[Hold] Delete after lock to release job: ${LOCK_AFTER}"
fi

while [ -f "${LOCK_AFTER}" ]; do
    sleep 30
done

echo "[Exit] Job finished."
