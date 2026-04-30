#!/bin/bash
#SBATCH -o /home/penghongen/My_Project/Pocket_Plus/processedPDB_EMDB_binder/bind_out_%A_%a.txt    # %A=主JOBID, %a=数组任务ID
#SBATCH -e /home/penghongen/My_Project/Pocket_Plus/processedPDB_EMDB_binder/bind_err_%A_%a.txt
#SBATCH -p cpu
#SBATCH --qos=Cpu96
#SBATCH -J bind

# ========================= 数组任务配置 / Array Job Configuration =========================
# 使用 --array 申请 n 个独立任务，每个任务分配 1 张 A100 + 16 核 CPU
# 修改 0-3 为 0-(n-1) 来调整任务数量，例如：0-7 表示 8 个任务
#SBATCH --array=0-2                                           # 4 个数组任务 (0,1,2,3)
#SBATCH -N 1                                                  # 每个任务 1 个节点
#SBATCH --ntasks=1                                            # 每个任务 1 个 task
#SBATCH --cpus-per-task=8                                    # 每个 task 8 核 CPU (用于 joblib 并行)



# ========================= 用户配置区 / User Configuration =========================
# EMDB-PDB 映射关系 JSON
# EMDB_PDB_JSON="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw4/all.json"
EMDB_PDB_JSON="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod4/all.json"
# EMDB_PDB_JSON="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_raw5/all.json"
# EMDB_PDB_JSON="/home/penghongen/My_Project/Data/split/3.5_cc_qscore_v2_mod5/all.json"



# EMDB 密度图文件夹 (按 emd_50120.map 命名)
EMDB_FOLDER_PATH="/storage/chenzhaoyang/cryo_em/EMDB_3.5_cc"
# PDB 解析结果文件夹 (由 run_preprocess.py 产生, 一个样本一个子目录)
# SAMPLE_ROOT_PATH="/home/penghongen/My_Project/Data/DATA_v2_raw4/parsed_pdb/"
SAMPLE_ROOT_PATH="/home/penghongen/My_Project/Data/DATA_v2_mod4/parsed_pdb/"
# SAMPLE_ROOT_PATH="/home/penghongen/My_Project/Data/DATA_v2_raw5/parsed_pdb/"
# SAMPLE_ROOT_PATH="/home/penghongen/My_Project/Data/DATA_v2_mod5/parsed_pdb/"
# 模拟密度图存储文件夹
SIM_FOLDER_PATH="/storage/penghongen/simulated_receptor_map"




# 输出文件夹: 1.0A
# LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/pdb_label_npz/"
# EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/emdb_npz/"
# SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/sim_npz/"
# LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_10A/ligand_dist_npz/"

LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_10A/pdb_label_npz/"
EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_10A/emdb_npz/"
SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_10A/sim_npz/"
LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_10A/ligand_dist_npz/"

# LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_10A/pdb_label_npz/"
# EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_10A/emdb_npz/"
# SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_10A/sim_npz/"
# LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_10A/ligand_dist_npz/"

# LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_10A/pdb_label_npz/"
# EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_10A/emdb_npz/"
# SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_10A/sim_npz/"
# LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_10A/ligand_dist_npz/"


# # 输出文件夹: 1.5A
# LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_15A/pdb_label_npz/"
# EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_15A/emdb_npz/"
# SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_15A/sim_npz/"
# LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw4_15A/ligand_dist_npz/"

# LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_15A/pdb_label_npz/"
# EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_15A/emdb_npz/"
# SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_15A/sim_npz/"
# LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod4_15A/ligand_dist_npz/"

# LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_15A/pdb_label_npz/"
# EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_15A/emdb_npz/"
# SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_15A/sim_npz/"
# LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_raw5_15A/ligand_dist_npz/"

# LABEL_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_15A/pdb_label_npz/"
# EMDB_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_15A/emdb_npz/"
# SIM_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_15A/sim_npz/"
# LIGAND_DIST_OUTPUT_PATH="/storage/penghongen/Pocket_classic/v2_mod5_15A/ligand_dist_npz/"




# 参数
TARGET_VOXEL_SIZE=1.0
# TARGET_VOXEL_SIZE=1.5    # NOTE
ADD_WHEN_CONFLICT="--add_when_conflict"   # 默认累加; 若要关闭改为 "--no-add_when_conflict"
OVERWRITE_EXISTING=""                      # 留空则跳过已有文件; 若要覆盖改为 "--overwrite_existing"
NUM_POCKET_CLASSES=4

# Python 脚本路径
SCRIPT_PATH="/home/penghongen/My_Project/Pocket_Plus/processedPDB_EMDB_binder/bind.py"
# Conda 环境激活命令
CONDA_ENV="source /home/penghongen/anaconda3/bin/activate vnegnn"
# 每个任务使用的并行进程数 (建议等于 cpus-per-task)
N_JOBS=8
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
    LOCK_PRE="bind_pre_lock_${SLURM_ARRAY_JOB_ID}"
    # touch $LOCK_PRE
    echo "[Lock] Pre-lock created: $LOCK_PRE"
    echo "[Wait] Waiting for user to delete $LOCK_PRE to start all tasks..."
    
    # 等待用户删除锁
    while [ -f "$LOCK_PRE" ]; do
        sleep 30
    done
    echo "[Start] Lock released! Starting binding..."
fi

# 所有任务等待主任务的锁释放 (通过短暂延迟实现同步)
if [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
    echo "[Sync] Task $SLURM_ARRAY_TASK_ID waiting for master task to release lock..."
    sleep $((SLURM_ARRAY_TASK_ID * 5))  # 错峰启动，避免同时读取文件列表
fi

# 创建后向锁
LOCK_AFTER="bind_after_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
touch $LOCK_AFTER
echo "[Lock] After-lock created: $LOCK_AFTER"

# 定义重试锁 (文件会在出错时动态创建)
LOCK_TRY="bind_try_lock_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
# ====================================================================================




# ========================= 3. 容错运行区 (循环重试机制) / Run Binding =========================
while true; do

    echo "[Run] Starting bind task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
    echo "[Run] JSON: $EMDB_PDB_JSON"
    echo "[Run] Jobs: $N_JOBS"

    if python "$SCRIPT_PATH" \
        --emdb_pdb_json "$EMDB_PDB_JSON" \
        --emdb_folder_path "$EMDB_FOLDER_PATH" \
        --sample_root_path "$SAMPLE_ROOT_PATH" \
        --sim_folder_path "$SIM_FOLDER_PATH" \
        --label_output_path "$LABEL_OUTPUT_PATH" \
        --emdb_output_path "$EMDB_OUTPUT_PATH" \
        --sim_output_path "$SIM_OUTPUT_PATH" \
        --ligand_dist_output_path "$LIGAND_DIST_OUTPUT_PATH" \
        --target_voxel_size "$TARGET_VOXEL_SIZE" \
        $ADD_WHEN_CONFLICT \
        $OVERWRITE_EXISTING \
        --part_id "$SLURM_ARRAY_TASK_ID" \
        --total_parts "$SLURM_ARRAY_TASK_COUNT" \
        --num_pocket_classes "$NUM_POCKET_CLASSES" \
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
