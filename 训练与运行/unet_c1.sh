#!/usr/bin/env bash
set -euo pipefail

scope=/home/penghongen/My_Project/tmp/adaligand_stage1_20260721T024000/allocations/321107
release=/home/penghongen/My_Project/tmp/aux_supervision_20260723/releases/Pocket_Plus_9417bcf
evidence="$scope/auxiliary_supervision_unet_9417bcf_warmup005"
feedback_root=/home/penghongen/My_Project/feedback_plus
stamp_base=job321107_auxsup_unet_9417bcf_warmup005_20260723T1545
formal_stamp=${stamp_base}_formal
formal_run="$feedback_root/logs/AdaLigand_Stage1-unet_c1/unet_c1____${formal_stamp}"

for path in "$release" "$scope/after_lock_321107"; do
  [[ -e "$path" ]] || { echo "[unet-auxsup][error] required path missing: $path" >&2; exit 1; }
done
for path in "$formal_run" "$evidence"; do
  [[ ! -e "$path" ]] || { echo "[unet-auxsup][error] refusing existing path: $path" >&2; exit 23; }
done
mkdir -p "$evidence"
printf 'release=%s\ncommit=%s\nformal_stamp=%s\nwarmup_ratio=0.005\n' \
  "$release" 9417bcf6c6da97ac7d91375d3fe94ccfc9ec8fde \
  "$formal_stamp" > "$evidence/release.env"

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
  +experiment=unet_c1
  init_from=null
  project_name=AdaLigand_Stage1
  train.devices=1
  train.nnodes=1
  train.ddp_find_unused_parameters=false
  train.global_batch_size=48
  train.batch_size=6
  train.strict_global_batch_size=true
  train.enable_batch_size_tuning=false
  train.num_workers=20
  train.max_epochs=20
  train.val_per_epoch=30
  train.optimizer.lr=1.0e-4
  train.scheduler.warmup_ratio=0.005
  train.scheduler.stop_after_lr_reductions=4
)

python src/train.py "${common[@]}" --cfg job --resolve > "$evidence/resolved.yaml"
sed -i '1d' "$evidence/resolved.yaml"
python - "$evidence/resolved.yaml" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
assert cfg.dataset.stage1_model_name == "unet_c1"
assert cfg.dataset.box_sample_fraction == 1.0
assert cfg.model.backbone.point_backbone is None
assert cfg.model.backbone.embed_head is None
assert cfg.model.backbone.voxel_backbone.feature_channels == 64
assert cfg.model.backbone.voxel_backbone.enable_multiscale_output is False
assert cfg.model.backbone.voxel_backbone.enable_structure_heads is True
assert cfg.train.optimizer.lr == 1.0e-4
assert cfg.train.scheduler.warmup_ratio == 0.005
assert [
    cfg.model.voxel_ligand_loss_weight,
    cfg.model.voxel_aux_loss_weight,
    cfg.model.ligand_distance_loss_weight,
    cfg.model.protein_mainchain_loss_weight,
    cfg.model.nucleic_mainchain_loss_weight,
] == [1.0, 0.1, 0.3, 0.05, 0.05]
print("resolved_config_gate=passed")
PY
echo '[unet-auxsup] resolved configuration gate passed; no smoke requested'

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

export POCKET_RUN_STAMP="$formal_stamp"
export WANDB_DIR="$evidence/wandb_formal"
mkdir -p "$WANDB_DIR"
echo "[unet-auxsup] starting formal unet_c1 directly: $formal_run"
python -u src/train.py "${common[@]}" offline=false > "$evidence/formal.out" 2> "$evidence/formal.err"
formal_best="$formal_run/checkpoints/BEST.ckpt"
[[ -f "$formal_best" ]] || { echo "[unet-auxsup][error] formal BEST missing: $formal_best" >&2; exit 1; }

printf 'status=complete\nrun=%s\nbest=%s\n' "$formal_run" "$formal_best" > "$evidence/result.env"
echo '[unet-auxsup] formal unet_c1 complete'
