#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/Pocket/Make_Data/PDB_processor/preprocess_out_%A_%a.txt    # %A=主JOBID, %a=数组任务ID
#SBATCH -e /home/penghongen/My_Project/Pocket/Make_Data/PDB_processor/preprocess_err_%A_%a.txt
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH -J pdb_prep

# ========================= 数组任务配置 / Array Job Configuration =========================
# 使用 --array 申请 n 个独立任务，每个任务分配 1 张 A100 + 16 核 CPU
# 修改 0-3 为 0-(n-1) 来调整任务数量，例如：0-7 表示 8 个任务
#SBATCH --array=0-0                                           # 4 个数组任务 (0,1,2,3)
#SBATCH -N 1                                                  # 每个任务 1 个节点
#SBATCH --gres=gpu:a100:1                                     # 每个任务 1 张 A100 (本脚本不使用GPU，但申请以保证资源隔离)
#SBATCH --ntasks=1                                            # 每个任务 1 个 task
#SBATCH --cpus-per-task=16                                    # 每个 task 16 核 CPU (用于 joblib 并行)



# ========================= 用户配置区 / User Configuration =========================
# 输入目录 (包含 PDB/CIF 文件)
INPUT_DIR="/storage/chenzhaoyang/cryo_em/PDB_3.5/"
# 输出目录 (处理后的 .npz 文件)
OUTPUT_DIR="/home/penghongen/My_Project/Data/parsed_pdb/"
# 错误日志目录
ERROR_DIR="/home/penghongen/My_Project/Data/parsed_pdb_error/"
# 是否覆盖已有的同名文件 (留空则跳过已有文件; 若要覆盖改为 "--overwrite")
OVERWRITE=""

# Python 脚本路径
SCRIPT_PATH="/home/penghongen/My_Project/Pocket/Make_Data/PDB_processor/run_preprocess.py"
# Conda 环境激活命令
CONDA_ENV="source /home/penghongen/anaconda3/bin/activate vnegnn"
# 每个任务使用的并行进程数 (建议等于 cpus-per-task)
N_JOBS=6
# 是否选择第一个 model
SELECT_FIRST_MODEL=False
# ====================================================================================




# ========================= 1. 环境准备区 / Environment Setup =========================
echo "========================================================"
echo "Array Job Started"
echo "========================================================"
echo "Main Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Total Array Tasks: $SLURM_ARRAY_TASK_COUNT"
echo "Host: $(hostname)"
echo "========================================================"

# 激活 Conda 环境
$CONDA_ENV

# 设置线程数，避免 OpenMP 和 joblib 冲突
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# ====================================================================================




# ========================= 2. 锁机制 (可选) / Lock Mechanism (Optional) =========================
# 创建前向锁 (仅主任务创建，用于统一控制启动)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    LOCK_PRE="preprocess_pre_lock_${SLURM_ARRAY_JOB_ID}"
    # touch $LOCK_PRE
    echo "[Lock] Pre-lock created: $LOCK_PRE"
    echo "[Wait] Waiting for user to delete $LOCK_PRE to start all tasks..."
    
    # 等待用户删除锁
    while [ -f "$LOCK_PRE" ]; do
        sleep 30
    done
    echo "[Start] Lock released! Starting preprocessing..."
fi

# 所有任务等待主任务的锁释放 (通过短暂延迟实现同步)
if [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
    echo "[Sync] Task $SLURM_ARRAY_TASK_ID waiting for master task to release lock..."
    sleep $((SLURM_ARRAY_TASK_ID * 5))  # 错峰启动，避免同时读取文件列表
fi

# 创建后向锁
LOCK_AFTER="preprocess_after_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
touch $LOCK_AFTER
echo "[Lock] After-lock created: $LOCK_AFTER"

# 定义重试锁 (文件会在出错时动态创建)
LOCK_TRY="preprocess_try_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
# ====================================================================================




# ========================= 3. 容错运行区 (循环重试机制) / Run Preprocessing =========================
while true; do

    echo "[Run] Starting preprocessing task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
    echo "[Run] Input: $INPUT_DIR"
    echo "[Run] Output: $OUTPUT_DIR"
    echo "[Run] Jobs: $N_JOBS"

    if python "$SCRIPT_PATH" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --error_dir "$ERROR_DIR" \
        $OVERWRITE \
        --part_id "$SLURM_ARRAY_TASK_ID" \
        --total_parts "$SLURM_ARRAY_TASK_COUNT" \
        --n_jobs "$N_JOBS"; then

        echo "[Success] Task $SLURM_ARRAY_TASK_ID finished successfully."
        rm -f "$LOCK_AFTER"
        rm -f "$LOCK_TRY"
        break
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[Error] Task $SLURM_ARRAY_TASK_ID FAILED!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

        touch "$LOCK_TRY"
        echo "[Action Required] Job is PAUSED."
        echo "   -> To RETRY: Delete file '$LOCK_TRY'"
        echo "   -> To CANCEL: Delete file '$LOCK_AFTER'"

        while [ -f "$LOCK_TRY" ] && [ -f "$LOCK_AFTER" ]; do
            sleep 20
        done

        if [ ! -f "$LOCK_AFTER" ]; then
            echo "[Stop] '$LOCK_AFTER' was deleted by user. Exiting loop."
            break
        else
            echo "[Retry] '$LOCK_TRY' was deleted by user. Restarting task immediately..."
            continue
        fi
    fi

done
# ====================================================================================




# ========================= 4. 不死鸟兜底机制 / Phoenix Fallback =========================
# 如果任务失败，after_lock 还存在，脚本会在这里等待
while [ -f "$LOCK_AFTER" ]; do
    sleep 60
done
# ====================================================================================

echo "[Exit] Task $SLURM_ARRAY_TASK_ID finished."
