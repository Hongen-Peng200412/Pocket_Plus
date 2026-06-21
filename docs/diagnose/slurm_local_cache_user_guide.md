# Pocket Plus Slurm Local Cache User Guide

This guide explains how to use the local-cache startup fix in daily Slurm usage,
when the cache needs to be refreshed, and how to clean node-local cache space.

## What Changed

Normal GPU sbatch scripts now use local-cache startup by default.

Covered entrypoints include:

- `sbatch/a100/1gpu.sbatch`
- `sbatch/a100/2gpu_1node.sbatch`
- `sbatch/a100/2gpu_2node.sbatch`
- `sbatch/a800/*.sbatch`
- `sbatch/h100/*.sbatch`
- `sbatch/h200/*.sbatch`

Local cache is controlled by this switch inside those scripts:

```bash
export POCKET_PLUS_LOCAL_CACHE_ENABLED="${POCKET_PLUS_LOCAL_CACHE_ENABLED:-1}"
```

So the normal command already uses local cache:

```bash
sbatch /home/penghongen/My_Project/Pocket_Plus/sbatch/a100/1gpu.sbatch "+experiment=ligand001"
```

Disable it for one submission only if needed:

```bash
POCKET_PLUS_LOCAL_CACHE_ENABLED=0 sbatch /home/penghongen/My_Project/Pocket_Plus/sbatch/a100/1gpu.sbatch "+experiment=ligand001"
```

## What Local Cache Does

At job startup, the sbatch system:

- Creates a node-local directory from `${SLURM_TMPDIR}` or `/tmp/${USER}/slurm_${SLURM_JOB_ID}`.
- Unpacks the packed conda environment to `${LOCAL_BASE}/conda_env`.
- Syncs the latest Pocket Plus code to `${LOCAL_BASE}/My_Project/Pocket_Plus`.
- Runs Python from the local conda env and local repo.
- Still writes experiment outputs to `/home/penghongen/My_Project/feedback_plus`.
- Cleans the current job's local cache when the allocation is finally released.

This is meant to avoid random startup hangs caused by very slow imports from
shared storage.

## Does The Cache Stay Valid?

If the conda environment has not changed, the existing packed cache remains
valid.

Code and config changes do not require repacking. The repo is synced to local
disk at each job startup, so normal edits are picked up automatically after code
is synced to the server.

You need to refresh or replace the environment pack when:

- You install, update, or remove packages in `Pocket_Plus_centos7_cu121_allgpu`.
- You switch to a different conda environment.
- You suspect the packed environment is stale or incomplete.

## Refresh The Existing Environment Pack

Run on the login node:

```bash
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate packer

ENV_NAME=Pocket_Plus_centos7_cu121_allgpu
PACK_DIR=/home/penghongen/My_Project/env_packs
PACK=${PACK_DIR}/${ENV_NAME}.tar.gz

mkdir -p "$PACK_DIR"
conda-pack -n "$ENV_NAME" -o "$PACK" --force --ignore-missing-files
ls -lh "$PACK"
tar -tzf "$PACK" | head
```

The current known pack path is:

```text
/home/penghongen/My_Project/env_packs/Pocket_Plus_centos7_cu121_allgpu.tar.gz
```

## Use A Different Conda Environment

Pack the new environment:

```bash
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate packer

NEW_ENV=Pocket_Plus_new_env
NEW_PACK=/home/penghongen/My_Project/env_packs/${NEW_ENV}.tar.gz
conda-pack -n "$NEW_ENV" -o "$NEW_PACK" --force --ignore-missing-files
ls -lh "$NEW_PACK"
```

Submit with the matching env name and pack:

```bash
Pocket_Plus_CONDA_ENV_NAME="$NEW_ENV" \
POCKET_PLUS_LOCAL_CACHE_ENV_PACK="$NEW_PACK" \
sbatch /home/penghongen/My_Project/Pocket_Plus/sbatch/a100/1gpu.sbatch "+experiment=ligand001"
```

## Already Paused At try_lock

If a job was submitted before the local-cache change and is already paused at
`try_lock_<JOBID>`, simply deleting `try_lock` will not automatically load the
new `_common.sh`. That job already sourced the old setup.

Best options:

- Cancel/release the allocation and submit a new job with the normal sbatch file.
- Or manually replace `/home/penghongen/run_cmd_<JOBID>.sh` with a local-cache
  command template, then delete `try_lock_<JOBID>`.

Manual template:

```bash
JOBID=245087
EXP="+experiment=ligand001"
RUN=/home/penghongen/run_cmd_${JOBID}.sh
cp "$RUN" "${RUN}.bak.localcache.$(date +%Y%m%d_%H%M%S)"
cat > "$RUN" <<EOF
#!/bin/bash
set -euxo pipefail

EXP="$EXP"
ENV_PACK="/home/penghongen/My_Project/env_packs/Pocket_Plus_centos7_cu121_allgpu.tar.gz"
LOCAL_BASE="\${SLURM_TMPDIR:-/tmp/\${USER}/slurm_\${SLURM_JOB_ID}}"
LOCAL_PROJECT_ROOT="\${LOCAL_BASE}/My_Project"
LOCAL_ENV="\${LOCAL_BASE}/conda_env"
LOCAL_REPO="\${LOCAL_PROJECT_ROOT}/Pocket_Plus"

mkdir -p "\${LOCAL_BASE}" "\${LOCAL_PROJECT_ROOT}"
touch "\${LOCAL_PROJECT_ROOT}/.project-root"
df -h "\${LOCAL_BASE}"

if [ ! -x "\${LOCAL_ENV}/bin/python" ]; then
  rm -rf "\${LOCAL_ENV}"
  mkdir -p "\${LOCAL_ENV}"
  tar -xzf "\${ENV_PACK}" -C "\${LOCAL_ENV}"
  "\${LOCAL_ENV}/bin/conda-unpack" || true
fi

rsync -a --delete --exclude ".git" --exclude "__pycache__" --exclude "*.pyc" --exclude ".pytest_cache" --exclude "feedback_plus" /home/penghongen/My_Project/Pocket_Plus/ "\${LOCAL_REPO}/"

export CONDA_PREFIX="\${LOCAL_ENV}"
export PATH="\${LOCAL_ENV}/bin:\${PATH}"
export PROJECT_ROOT="\${LOCAL_PROJECT_ROOT}"
export PYTHONPATH="\${LOCAL_PROJECT_ROOT}:\${LOCAL_REPO}:\${PYTHONPATH:-}"
export EXPERIMENT_FEEDBACK_ROOT="/home/penghongen/My_Project/feedback_plus"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export PYTHONPYCACHEPREFIX="\${LOCAL_BASE}/pycache"
export LD_PRELOAD="\${LOCAL_ENV}/lib/libstdc++.so.6:\${LOCAL_ENV}/lib/libgcc_s.so.1"

cd "\${LOCAL_PROJECT_ROOT}"
python -u Pocket_Plus/src/train.py "\${EXP}"
EOF
chmod +x "$RUN"
rm -f /home/penghongen/try_lock_${JOBID}
```

## Why Disk Usage May Increase

This line:

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda3       2.0T  344G  1.7T  18% /
```

shows the whole node filesystem, not just your cache directory. `344G` may become
`345G` because:

- Your job unpacked a conda env and repo into `/tmp/${USER}/slurm_${SLURM_JOB_ID}`.
- Python wrote bytecode cache under `${LOCAL_BASE}/pycache`.
- Other users or system processes wrote to the same node filesystem.
- A previous job was killed hard or the node did not run cleanup, leaving stale
  `/tmp/${USER}/slurm_*` directories.

New normal sbatch submissions now clean their own local cache when the allocation
is finally released. During `try_lock` or `after_lock`, the cache is intentionally
kept so retries are fast.

## Check Your Local Cache Usage

Run on the compute node if you are inside the allocation, or via an overlap step
if the cluster permits it:

```bash
du -sh /tmp/$USER/slurm_* 2>/dev/null | sort -h
df -h /tmp
```

From the login node, you can submit a small probe job if direct node access is
not available:

```bash
sbatch -p a100 --qos=a100g2 --cpus-per-task=1 --job-name=tmp_cache_probe \
  -o /home/penghongen/tmp_cache_probe_%j.out \
  -e /home/penghongen/tmp_cache_probe_%j.err \
  --wrap='set -eux; hostname; df -h /tmp; du -sh /tmp/$USER/slurm_* 2>/dev/null || true'
```

## Clean Stale Local Cache

Only remove caches for jobs that are no longer running.

Check running job ids first:

```bash
squeue -u "$USER" -h -o "%A %T %N"
```

Inspect stale cache directories:

```bash
ls -ld /tmp/$USER/slurm_* 2>/dev/null
du -sh /tmp/$USER/slurm_* 2>/dev/null | sort -h
```

Remove a specific stale cache after confirming the job is finished:

```bash
rm -rf /tmp/$USER/slurm_245087
```

Remove stale caches older than 3 days:

```bash
find /tmp/$USER -maxdepth 1 -type d -name 'slurm_*' -mtime +3 -print
find /tmp/$USER -maxdepth 1 -type d -name 'slurm_*' -mtime +3 -exec rm -rf {} +
```

If a job is still in `try_lock` or `after_lock`, do not remove its local cache
unless you intend the next retry to unpack the environment again.

## Optional Cleanup Controls

Default behavior:

```bash
POCKET_PLUS_LOCAL_CACHE_CLEANUP_ON_EXIT=1
```

Keep the cache after job exit for debugging:

```bash
POCKET_PLUS_LOCAL_CACHE_CLEANUP_ON_EXIT=0 sbatch /home/penghongen/My_Project/Pocket_Plus/sbatch/a100/1gpu.sbatch "+experiment=ligand001"
```

Usually no manual override is needed. Prefer the default job-specific local base.
If you suspect a stale unpacked env inside an existing paused allocation, remove
that allocation's `${LOCAL_BASE}/conda_env` and retry; it will unpack again.
