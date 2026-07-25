#!/usr/bin/env bash
set -euo pipefail

scope=/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/321540
release=/home/penghongen/My_Project/tmp/aux_supervision_20260723/releases/Pocket_Plus_1909267
evidence="$scope/auxiliary_supervision_1909267_warmup005"
feedback_root=/home/penghongen/My_Project/feedback_plus
stamp_base=job321540_auxsup_1909267_warmup005_20260723T1545
cpc1_stamp=${stamp_base}_CPC1
cpc2_stamp=${stamp_base}_CPC2
cpc1_run="$feedback_root/logs/AdaLigand_Stage1-Find_1-CPC1/Find_1-CPC1____${cpc1_stamp}"
cpc2_run="$feedback_root/logs/AdaLigand_Stage1-Find_1-CPC2/Find_1-CPC2____${cpc2_stamp}"

for path in "$release" "$scope/after_lock_321540"; do
  [[ -e "$path" ]] || { echo "[auxsup][error] required path missing: $path" >&2; exit 1; }
done
for path in "$cpc1_run" "$cpc2_run" "$evidence"; do
  [[ ! -e "$path" ]] || { echo "[auxsup][error] refusing existing path: $path" >&2; exit 23; }
done
mkdir -p "$evidence"
printf 'release=%s\ncommit=%s\ncpc1_stamp=%s\ncpc2_stamp=%s\nwarmup_ratio=0.005\n' \
  "$release" 1909267a472ca71d3cdf1492a63c8ce50b0cdcfd \
  "$cpc1_stamp" "$cpc2_stamp" > "$evidence/release.env"

set +u
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate Pocket_Plus_centos7_cu121_allgpu
set -u
export PYTHONPATH="$release:${PYTHONPATH:-}"
export ADALIGAND_DATA_ROOT=/storage/penghongen/AdaLigand/Ori_Data
export ADALIGAND_STAGE1_PREPARATION_ROOT=/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation/adaligand_stage1_20260721T024000
export EXPERIMENT_FEEDBACK_ROOT="$feedback_root"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID SLURM_NODEID
cd "$release"

common=(
  project_name=AdaLigand_Stage1
  train.devices=2
  train.nnodes=1
  train.ddp_find_unused_parameters=true
  train.global_batch_size=48
  train.batch_size=6
  train.strict_global_batch_size=true
  train.enable_batch_size_tuning=false
  train.num_workers=10
  train.max_epochs=20
  train.val_per_epoch=30
  train.optimizer.lr=5.0e-5
  train.scheduler.warmup_ratio=0.005
)
cpc1=(
  +experiment=CPC1/Find_1
  init_from=null
  "${common[@]}"
  train.scheduler.stop_after_lr_reductions=4
)
cpc2=(
  +experiment=CPC2/Find_1
  "init_from=$cpc1_run/checkpoints/BEST.ckpt"
  "${common[@]}"
  train.scheduler.stop_after_lr_reductions=1
)

python src/train.py "${cpc1[@]}" --cfg job --resolve > "$evidence/cpc1_resolved.yaml"
python src/train.py "${cpc2[@]}" --cfg job --resolve > "$evidence/cpc2_resolved.yaml"
sed -i '1d' "$evidence/cpc1_resolved.yaml" "$evidence/cpc2_resolved.yaml"
python - "$evidence/cpc1_resolved.yaml" "$evidence/cpc2_resolved.yaml" <<'PY'
import sys
from omegaconf import OmegaConf

cpc1, cpc2 = (OmegaConf.load(path) for path in sys.argv[1:])
for cfg in (cpc1, cpc2):
    assert cfg.dataset.stage1_model_name == "Find_1"
    assert cfg.dataset.box_sample_fraction == 1.0
    assert cfg.model.backbone.voxel_backbone.feature_channels == 64
    assert cfg.model.backbone.voxel_backbone.enable_multiscale_output is False
    assert cfg.model.backbone.voxel_backbone.enable_structure_heads is True
    assert cfg.model.backbone.embed_head.use_gaussian_splatting is True
    assert cfg.train.optimizer.lr == 5.0e-5
    assert cfg.train.scheduler.warmup_ratio == 0.005
assert [
    cpc1.model.voxel_ligand_loss_weight,
    cpc1.model.voxel_aux_loss_weight,
    cpc1.model.ligand_distance_loss_weight,
    cpc1.model.protein_mainchain_loss_weight,
    cpc1.model.nucleic_mainchain_loss_weight,
] == [1.0, 0.1, 0.3, 0.05, 0.05]
assert [
    cpc2.model.voxel_ligand_loss_weight,
    cpc2.model.voxel_aux_loss_weight,
    cpc2.model.ligand_distance_loss_weight,
    cpc2.model.protein_mainchain_loss_weight,
    cpc2.model.nucleic_mainchain_loss_weight,
] == [0.0, 0.0, 0.0, 0.0, 0.0]
print("resolved_config_gate=passed")
PY
echo '[auxsup] resolved configuration gate passed; no smoke requested'

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits >> "$evidence/gpu_10s.csv"
    sleep 10
  done
) &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

export POCKET_RUN_STAMP="$cpc1_stamp"
export WANDB_DIR="$evidence/wandb_cpc1"
mkdir -p "$WANDB_DIR"
echo "[auxsup] starting formal CPC1 directly: $cpc1_run"
python -u src/train.py "${cpc1[@]}" offline=false > "$evidence/cpc1.out" 2> "$evidence/cpc1.err"
cpc1_best="$cpc1_run/checkpoints/BEST.ckpt"
[[ -f "$cpc1_best" ]] || { echo "[auxsup][error] CPC1 BEST missing: $cpc1_best" >&2; exit 1; }

export POCKET_RUN_STAMP="$cpc2_stamp"
export WANDB_DIR="$evidence/wandb_cpc2"
mkdir -p "$WANDB_DIR"
echo "[auxsup] starting formal CPC2: $cpc2_run"
python -u src/train.py "${cpc2[@]}" offline=false > "$evidence/cpc2.out" 2> "$evidence/cpc2.err"
cpc2_best="$cpc2_run/checkpoints/BEST.ckpt"
[[ -f "$cpc2_best" ]] || { echo "[auxsup][error] CPC2 BEST missing: $cpc2_best" >&2; exit 1; }

printf 'status=complete\ncpc1_run=%s\ncpc1_best=%s\ncpc2_run=%s\ncpc2_best=%s\n' \
  "$cpc1_run" "$cpc1_best" "$cpc2_run" "$cpc2_best" > "$evidence/result.env"
echo '[auxsup] formal CPC1 to CPC2 chain complete'
