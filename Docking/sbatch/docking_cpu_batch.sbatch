#!/bin/bash
#SBATCH -o /home/penghongen/分子对接尝试/slurm_docking_%j.out
#SBATCH -e /home/penghongen/分子对接尝试/slurm_docking_%j.err
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J DockFull80
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48

RUN_ID="${RUN_ID:-20260519_full80_nstruct2_conservative}"
NSTRUCT="${NSTRUCT:-2}"
JOBS="${JOBS:-24}"
SAMPLE_LIST="${SAMPLE_LIST:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MODE="${MODE:-dock}"
MIN_VOXELS="${MIN_VOXELS:-30}"
DISABLE_MERGE="${DISABLE_MERGE:-0}"
MAX_SITES_PER_SAMPLE="${MAX_SITES_PER_SAMPLE:-}"
MIN_VOXELS_GRID="${MIN_VOXELS_GRID:-10,20,30,50,80}"
MERGE_MIN_DISTANCE_GRID="${MERGE_MIN_DISTANCE_GRID:-1.5,2.5,3.5}"
MERGE_CENTER_DISTANCE_GRID="${MERGE_CENTER_DISTANCE_GRID:-4.0,6.0,8.0}"
MERGE_BBOX_INCREASE_GRID="${MERGE_BBOX_INCREASE_GRID:-6.0,8.0,12.0}"
ALLOWED_ROOT="/home/penghongen/分子对接尝试"
RUN_ROOT="${ALLOWED_ROOT}/pipeline_runs/${RUN_ID}"

mkdir -p "${RUN_ROOT}/slurm"
cd /home/penghongen/My_Project/Pocket_Plus

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home/penghongen/anaconda3/bin/activate Pocket_Plus_centos7_cu121_allgpu

LOCK_PRE="${RUN_ROOT}/slurm/pre_lock_cpu_${SLURM_JOB_ID}"
LOCK_AFTER="${RUN_ROOT}/slurm/after_lock_cpu_${SLURM_JOB_ID}"
touch "${LOCK_PRE}" "${LOCK_AFTER}"
echo "[Status] pre lock: ${LOCK_PRE}"
echo "[Status] after lock: ${LOCK_AFTER}"
echo "[Wait] Delete pre lock to start."

while [ -f "${LOCK_PRE}" ]; do
    sleep 30
done

COMMON_ARGS=(--run-id "${RUN_ID}" --jobs "${JOBS}")
if [ -n "${SAMPLE_LIST}" ]; then
    COMMON_ARGS+=(--sample-list "${SAMPLE_LIST}")
fi
if [ -n "${MAX_SAMPLES}" ]; then
    COMMON_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [ -n "${MAX_SITES_PER_SAMPLE}" ]; then
    COMMON_ARGS+=(--max-sites-per-sample "${MAX_SITES_PER_SAMPLE}")
fi

if [ "${MODE}" = "prescan" ]; then
    COMMAND=(
        python Docking/run_instance_prescan.py "${COMMON_ARGS[@]}"
        --min-voxels-grid "${MIN_VOXELS_GRID}"
        --merge-min-distance-grid "${MERGE_MIN_DISTANCE_GRID}"
        --merge-center-distance-grid "${MERGE_CENTER_DISTANCE_GRID}"
        --merge-bbox-increase-grid "${MERGE_BBOX_INCREASE_GRID}"
    )
else
    COMMAND=(python Docking/run_docking_batch.py "${COMMON_ARGS[@]}" --nstruct "${NSTRUCT}" --min-voxels "${MIN_VOXELS}")
    if [ "${DISABLE_MERGE}" = "1" ]; then
        COMMAND+=(--disable-merge)
    fi
fi

echo "[Start] ${COMMAND[*]}"
if srun "${COMMAND[@]}"; then
    echo "[Success] Docking batch finished."
    rm -f "${LOCK_AFTER}"
else
    echo "[Error] Docking batch failed. Inspect ${RUN_ROOT}/slurm and sample audit files."
    echo "[Hold] Delete after lock to release job: ${LOCK_AFTER}"
fi

while [ -f "${LOCK_AFTER}" ]; do
    sleep 30
done

echo "[Exit] Job finished."
