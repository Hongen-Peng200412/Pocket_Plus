#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/feedback/infer_log/v0_half1_search/search_out_%A_%a.txt    # %A=主JOBID, %a=数组任务ID
#SBATCH -e /home/penghongen/My_Project/feedback/infer_log/v0_half1_search/search_err_%A_%a.txt
#SBATCH -p a100
#SBATCH --qos=a100g2
#SBATCH -J Csearch_v0_half1

# ========================= 数组任务配置 / Array Job Configuration =========================
# 使用 --array 申请 n 个独立任务
#SBATCH --array=0-0                                           # 1 个数组任务
#SBATCH -N 1                                                  # 每个任务 1 个节点
#SBATCH --ntasks=1                                            # 每个任务 1 个 task
#SBATCH --cpus-per-task=16                                     # 每个 task 8 核 CPU 
#SBATCH --gres=gpu:1                                          # 每个 task 分配 1 张 GPU



# ========================= 用户配置区 / User Configuration =========================
# Python 脚本路径
SCRIPT_PATH="/home/penghongen/My_Project/Pocket/src/inference/run.py"
# Conda 环境激活命令
CONDA_ENV="source /home/penghongen/anaconda3/bin/activate vnegnn"
# 调参配置名
CONFIG_NAME="raw_param_search"
# ====================================================================================




# ========================= 1. 环境准备区 / Environment Setup =========================
echo "========================================================"
echo "Param Search Array Job Started (GPU Partition)"
echo "========================================================"
echo "Main Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Total Array Tasks: $SLURM_ARRAY_TASK_COUNT"
echo "Host: $(hostname)"
echo "========================================================"

# 激活 Conda 环境
$CONDA_ENV

# 设置线程数，避免 OpenMP 冲突
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# ====================================================================================




# ========================= 2. 锁机制 (可选) / Lock Mechanism (Optional) =========================
# 创建前向锁 (每个任务独立创建，用于精准控制各任务启动)
LOCK_PRE="search_v0_pre_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
touch $LOCK_PRE
echo "[Lock] Pre-lock created: $LOCK_PRE"
echo "[Wait] Waiting for user to delete $LOCK_PRE to start task $SLURM_ARRAY_TASK_ID..."

# 每个任务独立等待属于自己的锁被删除
while [ -f "$LOCK_PRE" ]; do
    sleep 30
done

echo "[Start] Lock released for task $SLURM_ARRAY_TASK_ID! Starting search..."

# 错峰启动，避免多个任务在锁释放瞬间同时读取大量文件列表造成 IO 压力
sleep $((SLURM_ARRAY_TASK_ID * 5))


# 创建后向锁 (用于标记任务是否正在运行)
LOCK_AFTER="search_v0_after_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
touch $LOCK_AFTER
echo "[Lock] After-lock created: $LOCK_AFTER"

# 定义重试锁 (文件会在出错时动态创建)
LOCK_TRY="search_v0_try_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
# ====================================================================================




# ========================= 3. 容错运行区 (循环重试机制) / Run Split =========================
while true; do

    echo "[Run] Starting search task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"

    # 执行 GPU 推理搜索
    if python -u "$SCRIPT_PATH" \
        --config-name="$CONFIG_NAME"; then

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

        # 挂起等待
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
