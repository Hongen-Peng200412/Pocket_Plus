#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/Pocket_Plus/Make_Data/process_and_label_out_%A_%a.txt
#SBATCH -e /home/penghongen/My_Project/Pocket_Plus/Make_Data/process_and_label_err_%A_%a.txt
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J v_2_raw5

# ========================= 数组任务配置 / Array Job Configuration =========================
# 修改 0-3 为 0-(n-1) 来调整任务数量
#SBATCH --array=0-2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8



# ========================= 用户配置区 / User Configuration =========================
# 工作模式: full=从零开始（解析+特征+打标签）; label_only=仅打标签
MODE="full"
PARSE_ATOMS="--parse_atoms"
PARSE_RESIDUES="--parse_residues"
PARSE_GRAPH="--parse_graph"


# Mode full: PDB/CIF 文件目录
# Mode label_only: 已处理的样本根目录（含各样本子目录，每个子目录含 candidates.npz）
INPUT_DIR="/storage/chenzhaoyang/cryo_em/PDB_3.5/"

# 输出目录（Mode full 时使用；label_only 时 labels.npz 写入各样本子目录）
OUTPUT_DIR="/home/penghongen/My_Project/Data/DATA_v2_raw5/parsed_pdb/"

# 错误日志目录
ERROR_DIR="/home/penghongen/My_Project/Data/DATA_v2_raw5/"
ALLOW_INCOMPLETE_BACKBONE="--allow_incomplete_backbone"

# 是否覆盖已有文件（留空=跳过; --overwrite=覆盖）
OVERWRITE=""

# Python 脚本路径
SCRIPT_PATH="/home/penghongen/My_Project/Pocket_Plus/Make_Data/process_and_label.py"
# Conda 环境激活命令
CONDA_ENV="source /home/penghongen/anaconda3/bin/activate vnegnn"
# 并行进程数
N_JOBS=8
# 筛选方案名（对应 labels/filter_config.py 中的 *_PRESET 常量名去掉 _PRESET 后再小写）
# 例如: FIVE_CLASS_PRESET -> "five_class"
CLASS_NAME="five_class_raw5"

# ====================================================================================



# ========================= 1. 环境准备区 / Environment Setup =========================
echo "========================================================"
echo "Full Pipeline Array Job Started"
echo "========================================================"
echo "Main Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Total Array Tasks: $SLURM_ARRAY_TASK_COUNT"
echo "Mode: $MODE"
echo "Host: $(hostname)"
echo "========================================================"

$CONDA_ENV

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# ====================================================================================



# ========================= 2. 锁机制 / Lock Mechanism =========================
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    LOCK_PRE="pal_pre_lock_${SLURM_ARRAY_JOB_ID}"
    # touch $LOCK_PRE
    echo "[Lock] Pre-lock: $LOCK_PRE"
    while [ -f "$LOCK_PRE" ]; do sleep 30; done
    echo "[Start] Lock released!"
fi

if [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
    sleep $((SLURM_ARRAY_TASK_ID * 5))
fi

LOCK_AFTER="pal_after_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
touch $LOCK_AFTER
LOCK_TRY="pal_try_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
# ====================================================================================



# ========================= 3. 容错运行区 / Run with Retry =========================
while true; do
    echo "[Run] Task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
    echo "[Run] Mode: $MODE | Input: $INPUT_DIR | ClassName: $CLASS_NAME"

    if python "$SCRIPT_PATH" \
        --mode "$MODE" \
        $PARSE_ATOMS \
        $PARSE_RESIDUES \
        $PARSE_GRAPH \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --error_dir "$ERROR_DIR" \
        $ALLOW_INCOMPLETE_BACKBONE \
        $OVERWRITE \
        --part_id "$SLURM_ARRAY_TASK_ID" \
        --total_parts "$SLURM_ARRAY_TASK_COUNT" \
        --n_jobs "$N_JOBS" \
        --class_name "$CLASS_NAME" \
        ; then

        echo "[Success] Task $SLURM_ARRAY_TASK_ID finished."
        rm -f "$LOCK_AFTER" "$LOCK_TRY"
        break
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "[Error] Task $SLURM_ARRAY_TASK_ID FAILED!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        touch "$LOCK_TRY"
        echo "[Action Required] -> To RETRY: Delete '$LOCK_TRY'"
        echo "                 -> To CANCEL: Delete '$LOCK_AFTER'"
        while [ -f "$LOCK_TRY" ] && [ -f "$LOCK_AFTER" ]; do sleep 20; done
        if [ ! -f "$LOCK_AFTER" ]; then
            echo "[Stop] Exiting."; break
        else
            echo "[Retry] Restarting..."; continue
        fi
    fi
done
# ====================================================================================



# ========================= 4. 不死鸟兜底机制 / Phoenix Fallback =========================
while [ -f "$LOCK_AFTER" ]; do sleep 60; done
# ====================================================================================

echo "[Exit] Task $SLURM_ARRAY_TASK_ID finished."

# ========================= 5. Global Statistics (Task 0 Only) =========================
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "========================================================"
    echo "Task 0 / 主节点 - 开始等待全部任务结束以进行全局统计..."
    echo "========================================================"
    # 借助 squeue 检查本 Array 下还有多个任务处于未结束甚至还在队列中的状态
    while true; do
        WAITING=$(squeue | grep "${SLURM_ARRAY_JOB_ID}_" | wc -l)
        if [ "$WAITING" -le 1 ]; then
            break
        fi
        sleep 60
    done
    echo "其余任务已执行完毕！开始执行统计脚本 / All tasks completed. Running statistics..."
    python /home/penghongen/My_Project/Pocket_Plus/Make_Data/compute_statistics.py --input_dir "$OUTPUT_DIR" --n_jobs "$N_JOBS"
fi
# ====================================================================================
