# Slurm Startup Hang Diagnosis

This note is for Pocket Plus training jobs that are submitted through `sbatch`
and sometimes appear to hang at startup with no useful stdout/stderr progress.
It assumes the cluster does not allow direct SSH from the login node to compute
nodes, so diagnostics must work through Slurm metadata, lock files, logs, and the
existing `run_cmd_<JOBID>.sh` retry mechanism.

## Symptom

A job is `RUNNING` in `squeue`, but the `.out` log stops around the dynamic
command banner or the first training print, for example:

```text
[Attempt] Executing dynamic command from /home/penghongen/run_cmd_<JOBID>.sh...
python /home/penghongen/My_Project/Pocket_Plus/src/train.py "+experiment=ligand002"
[Train] GLOO_SOCKET_IFNAME 已设置为 'ib0' ...
```

There may be no traceback in `.err`. Some jobs start normally, while others
remain silent in this early phase.

Do not assume this is the same as a later training exception. A restart may move
past the startup hang and then expose an unrelated Python/Lightning/data bug.

## Important Constraints

* Direct `ssh <compute-node>` from the login node may fail with permission
  errors. Prefer login-node Slurm commands first.
* `srun --jobid=<JOBID> --overlap ...` may also hang if the allocation or node is
  unhealthy. If it hangs, interrupting the diagnostic step can cancel only that
  step, but it is still noisy and not always useful.
* The project sbatch core has a safer control channel:
  `/home/penghongen/kill_lock_<JOBID>` kills the current `run_cmd` process group.
  The job then pauses at `/home/penghongen/try_lock_<JOBID>` without releasing the
  allocation. Edit `/home/penghongen/run_cmd_<JOBID>.sh`, then delete
  `try_lock_<JOBID>` to retry.
* To release the allocation, delete `/home/penghongen/after_lock_<JOBID>`.

## General Strategy

1. Confirm the job, logs, node, and lock state from the login node.
2. Check whether stdout/stderr file size and mtime are changing.
3. If direct node probing is blocked, do not keep retrying SSH.
4. If the process is silently stuck, use `kill_lock` to pause the job safely.
5. Restart the same allocation with diagnostics enabled:
   `PYTHONUNBUFFERED=1`, `PYTHONFAULTHANDLER=1`, `HYDRA_FULL_ERROR=1`, and
   optionally `-X importtime`.
6. If diagnostics show the program reaches the training logic, handle the new
   traceback as a separate bug.
7. After a diagnosis-only run, remove noisy flags such as `-X importtime` before
   resuming a real training run.

## One-Shot Login-Node Snapshot

Use this first. It does not require compute-node SSH.

```bash
JOBID=245087 bash -lc '
set +e
echo "========== Job =========="
date
squeue -j "$JOBID" -o "%.18i %.9P %.20j %.12u %.2t %.12M %.6D %R"
scontrol show job -dd "$JOBID" | sed "s/ /\n/g" | egrep "JobId=|JobState=|RunTime=|NodeList=|BatchHost=|Command=|WorkDir=|StdOut=|StdErr=|Reason=|ExitCode=|TRES=|JOB_GRES="

OUT=$(scontrol show job -dd "$JOBID" | tr " " "\n" | awk -F= "/^StdOut=/{print \$2; exit}")
ERR=$(scontrol show job -dd "$JOBID" | tr " " "\n" | awk -F= "/^StdErr=/{print \$2; exit}")
RUN="/home/penghongen/run_cmd_${JOBID}.sh"

echo
echo "========== Files =========="
echo "OUT=$OUT"
echo "ERR=$ERR"
echo "RUN=$RUN"
ls -lh "$OUT" "$ERR" "$RUN" 2>/dev/null
stat "$OUT" "$ERR" "$RUN" 2>/dev/null

echo
echo "========== Locks =========="
ls -l /home/penghongen/*_lock_${JOBID} 2>/dev/null || true

echo
echo "========== run_cmd =========="
sed -n "1,120p" "$RUN" 2>/dev/null

echo
echo "========== stdout tail =========="
tail -n 120 "$OUT" 2>/dev/null

echo
echo "========== stderr tail =========="
tail -n 160 "$ERR" 2>/dev/null

echo
echo "========== Accounting =========="
sacct -j "$JOBID" --format=JobID,JobName%25,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,NodeList%20 -P 2>/dev/null
'
```

Use the `JOBID=245087 bash -lc '...'` form rather than
`JOBID=245087; bash -lc '...'`. The latter does not export `JOBID` into the child
shell, so commands such as `scontrol show job -dd "$JOBID"` may accidentally run
with an empty job id and dump unrelated jobs.

## Optional In-Allocation Probe

Use this only if the cluster permits overlapping Slurm steps. Avoid direct SSH.
If it hangs for more than about one minute, stop using this path and switch to
the `kill_lock` retry workflow.

```bash
JOBID=245087; NODE=$(scontrol show job -dd "$JOBID" | tr " " "\n" | awk -F= "/^BatchHost=/{print \$2; exit}"); PROBE=/home/penghongen/probe_${JOBID}_$(date +%Y%m%d_%H%M%S).log; srun --jobid="$JOBID" --overlap -w "$NODE" -N1 -n1 --cpus-per-task=1 bash -lc "exec > '$PROBE' 2>&1; echo HOST=\$(hostname) DATE=\$(date); ps -u \$USER -o pid,ppid,stat,etime,%cpu,%mem,rss,wchan:32,cmd --sort=-%cpu | egrep 'PID|train.py|run_cmd_${JOBID}|python|bash|slurm' | head -120; PID=\$(pgrep -u \$USER -f '/home/penghongen/My_Project/Pocket_Plus/src/train.py|run_cmd_${JOBID}.sh' | head -1); echo SELECTED_PID=\$PID; if [ -n \"\$PID\" ]; then egrep 'Name|State|Threads|VmRSS|VmSize|voluntary_ctxt_switches|nonvoluntary_ctxt_switches' /proc/\$PID/status; echo WCHAN=\$(cat /proc/\$PID/wchan); tr '\0' '\n' < /proc/\$PID/environ | egrep 'CUDA|NCCL|GLOO|MASTER|RANK|WORLD|OMP|HYDRA|CONDA|PYTHON|SLURM' | sort; fi; nvidia-smi; nvidia-smi pmon -c 1 2>/dev/null; free -h; df -h /home /tmp 2>/dev/null"; echo "PROBE=$PROBE"; tail -n 200 "$PROBE"
```

## Safe Restart With Import Diagnostics

Use this when the job is stuck and stdout/stderr do not move. It keeps the Slurm
allocation alive, kills only the current dynamic command process group, extracts
the current `+experiment=...` override from `run_cmd_<JOBID>.sh`, replaces the
run command with a diagnostic command, and retries.

```bash
JOBID=245087; RUN=/home/penghongen/run_cmd_${JOBID}.sh; OUT=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.out; ERR=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.err; EXP=$(grep -oE '\+experiment=[^"[:space:]]+' "$RUN" | head -1); EXP=${EXP:-+experiment=ligand001}; touch /home/penghongen/kill_lock_${JOBID}; echo "[diag] sent kill_lock, waiting for try_lock..."; while [ ! -f /home/penghongen/try_lock_${JOBID} ]; do sleep 2; done; cp "$RUN" "${RUN}.bak.$(date +%Y%m%d_%H%M%S)"; cat > "$RUN" <<EOF
#!/bin/bash
set -x
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
python -u -X faulthandler -X importtime /home/penghongen/My_Project/Pocket_Plus/src/train.py "$EXP"
EOF
chmod +x "$RUN"; rm -f /home/penghongen/try_lock_${JOBID}; echo "[diag] restarted with import-time diagnostics"; sleep 30; echo "===== OUT ====="; tail -n 120 "$OUT"; echo "===== ERR ====="; tail -n 200 "$ERR"
```

## Resume A Real Run After Diagnostics

After diagnostics confirm startup is no longer stuck, remove `-X importtime`
because it floods stderr. Add only the Hydra overrides needed for the current
failure. Example for bypassing a Lightning batch-size-finder bug:

```bash
JOBID=245087; RUN=/home/penghongen/run_cmd_${JOBID}.sh; OUT=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.out; ERR=/home/penghongen/My_Project/feedback_plus/logs/_temp_slurm/train_${JOBID}.err; cp "$RUN" "${RUN}.bak.$(date +%Y%m%d_%H%M%S)"; cat > "$RUN" <<'EOF'
#!/bin/bash
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
python -u /home/penghongen/My_Project/Pocket_Plus/src/train.py "+experiment=ligand002" train.enable_batch_size_tuning=false train.batch_size=1
EOF
chmod +x "$RUN"; rm -f /home/penghongen/try_lock_${JOBID}; echo "[restart] resumed real training"; sleep 30; echo "===== OUT ====="; tail -n 120 "$OUT"; echo "===== ERR ====="; tail -n 120 "$ERR"
```

## Local Cache Startup

The most useful mitigation found so far is to reduce startup dependence on the
shared filesystem. The project now has a local-cache feature in
`sbatch/_common.sh`, and the normal GPU sbatch entrypoints enable it by default.

The switch is explicit in the original GPU sbatch scripts, for example
`sbatch/a100/1gpu.sbatch`, `sbatch/a800/1gpu.sbatch`,
`sbatch/h100/1gpu.sbatch`, and the 2-GPU variants:

```bash
export POCKET_PLUS_LOCAL_CACHE_ENABLED="${POCKET_PLUS_LOCAL_CACHE_ENABLED:-1}"
export POCKET_PLUS_LOCAL_CACHE_ENV_PACK="${POCKET_PLUS_LOCAL_CACHE_ENV_PACK:-/home/penghongen/My_Project/env_packs/Pocket_Plus_centos7_cu121_allgpu.tar.gz}"
```

Disable local cache for one submission with:

```bash
POCKET_PLUS_LOCAL_CACHE_ENABLED=0 sbatch /home/penghongen/My_Project/Pocket_Plus/sbatch/a100/1gpu.sbatch "+experiment=ligand002"
```

What local cache does:

* It chooses node-local storage from `${SLURM_TMPDIR}` or `/tmp/${USER}/slurm_${SLURM_JOB_ID}`.
* It requires at least 20 GB free by default.
* It unpacks the packed conda env into `${LOCAL_BASE}/conda_env`.
* It syncs `/home/penghongen/My_Project/Pocket_Plus` into `${LOCAL_BASE}/My_Project/Pocket_Plus`.
* It creates `${LOCAL_BASE}/My_Project/.project-root` so `rootutils` can find the local project root.
* It sets `PROJECT_ROOT`, `CONDA_PREFIX`, `PATH`, `PYTHONPATH`, `LD_PRELOAD`, and `PYTHONPYCACHEPREFIX` to prefer local files.
* It keeps experiment outputs under `/home/penghongen/My_Project/feedback_plus`.
* It exports `POCKET_PLUS_LOCAL_CACHE_REPO` and `POCKET_PLUS_LOCAL_CACHE_ENV` for custom commands.
* New normal sbatch submissions clean the current job's node-local cache when
  the allocation is finally released. While `try_lock` or `after_lock` is held,
  the cache remains available for fast retries.

Create or refresh the packed conda environment with:

```bash
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate packer
ENV_NAME=Pocket_Plus_centos7_cu121_allgpu
PACK_DIR=/home/penghongen/My_Project/env_packs
PACK=${PACK_DIR}/${ENV_NAME}.tar.gz
mkdir -p "$PACK_DIR"
conda-pack -n "$ENV_NAME" -o "$PACK" --force --ignore-missing-files
ls -lh "$PACK"
```

Cache validity rules:

* If the conda environment has not changed, the existing pack remains valid.
* Normal code/config changes do not require repacking because the repo is
  rsynced to local disk at every job startup.
* If the conda environment is modified, refresh the pack with `conda-pack`.
* If using a different conda environment, create a different pack and point the
  sbatch script to it.
* If an already-running allocation has unpacked an old environment under
  `/tmp/${USER}/slurm_${SLURM_JOB_ID}/conda_env`, remove that local env or use a
  different local base before retrying with a new pack.

Use a new conda environment with:

```bash
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate packer
NEW_ENV=Pocket_Plus_new_env
NEW_PACK=/home/penghongen/My_Project/env_packs/${NEW_ENV}.tar.gz
conda-pack -n "$NEW_ENV" -o "$NEW_PACK" --force --ignore-missing-files
Pocket_Plus_CONDA_ENV_NAME="$NEW_ENV" POCKET_PLUS_LOCAL_CACHE_ENV_PACK="$NEW_PACK" sbatch /home/penghongen/My_Project/Pocket_Plus/sbatch/a100/1gpu.sbatch "+experiment=ligand002"
```

For custom `run_cmd` scripts, prefer local or relative entrypoints:

```bash
python Pocket_Plus/src/train.py "+experiment=ligand002"
python "${POCKET_PLUS_LOCAL_CACHE_REPO}/src/train.py" "+experiment=ligand002"
```

Avoid absolute shared-code entrypoints such as
`/home/penghongen/My_Project/Pocket_Plus/src/train.py` when the goal is to test
local repo startup. Absolute shared data/checkpoint/output paths are still fine.

## Existing Jobs Paused At try_lock

New sbatch submissions automatically source the updated `_common.sh`, so they
can use the default-on local cache.

Jobs that are already running and paused at `try_lock_<JOBID>` have already
sourced `_common.sh`. They will not automatically inherit later changes to
`_common.sh` just by deleting `try_lock`. They can still use local cache, but the
cache setup must be placed directly into `/home/penghongen/run_cmd_<JOBID>.sh`,
or the job should be cancelled and resubmitted.

Template for an existing paused job:

```bash
JOBID=245097
EXP="+experiment=ligand002"
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

rsync -a --delete \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".pytest_cache" \
  --exclude "feedback_plus" \
  /home/penghongen/My_Project/Pocket_Plus/ "\${LOCAL_REPO}/"

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

## Interpreting Outcomes

* `squeue` says `RUNNING`, logs do not change, and no traceback appears:
  treat it as a startup hang and use `kill_lock` plus diagnostic restart.
* Import diagnostics show `import time:` lines for Python standard-library
  modules, but each small module takes hundreds of milliseconds or seconds:
  suspect node-local or shared-filesystem slowness around Python/Conda import
  metadata. This can look like a silent hang before project code runs.
* Diagnostics reach `import torch` slowly but continue:
  this is not necessarily a bug. Keep watching for the next user-level print or
  traceback.
* Diagnostics reach WandB, data module setup, model setup, or Lightning tuner:
  the original startup hang is gone. Any later traceback is a separate runtime
  bug.
* `try_lock_<JOBID>` exists:
  the current dynamic command has failed or been killed, and the allocation is
  paused waiting for your next edit.
* `after_lock_<JOBID>` exists:
  the allocation is intentionally kept alive.

## Notes From Job 245088

Job `245088` first looked like a startup hang. After `kill_lock` and a diagnostic
restart, it progressed through imports, data module setup, model initialization,
WandB initialization, and Lightning batch-size tuning. The later failure was:

```text
KeyError: 'limit_eval_batches'
```

That error is a Lightning batch-size-finder restore bug, not proof that the
original startup hang was imaginary. The practical workaround is to disable
`train.enable_batch_size_tuning` and set a known safe `train.batch_size`.

## Notes From Job 245087

Job `245087` on `gnode07` remained silent after the dynamic command banner:

```text
python /home/penghongen/My_Project/Pocket_Plus/src/train.py "+experiment=ligand001"
```

After a `kill_lock` diagnostic restart, stderr started showing `-X importtime`
output, but even core Python modules were extremely slow:

```text
encodings        ~3.7s cumulative
site             ~16.0s cumulative
pathlib          ~15.4s cumulative
typing           ~4.4s cumulative
```

This points to very slow Python environment/module loading on that node rather
than a deterministic project-code exception. If the import stream keeps moving,
let it reach either `[Train] GLOO_SOCKET_IFNAME...` or a traceback. If it stays
in standard-library imports for many minutes, prefer releasing this allocation
and resubmitting with the suspicious node excluded.

## Notes From Local Cache Tests

We created a separate `packer` conda environment and used `conda-pack` to pack
`Pocket_Plus_centos7_cu121_allgpu`:

```text
/home/penghongen/My_Project/env_packs/Pocket_Plus_centos7_cu121_allgpu.tar.gz
size: about 3.4 GB
```

Manual tests on existing allocations showed:

* `245088` on `gnode02` had enough local disk: `/tmp/penghongen/slurm_245088` had about 1.7 TB available.
* `245099` on `gnode04` successfully unpacked/reused the local conda env and launched Python from `/tmp/penghongen/slurm_245099/conda_env/bin/python`.
* The first local-cache attempt failed because `rootutils` could not find `.project-root`; this was fixed by creating `${LOCAL_BASE}/My_Project/.project-root`.
* After the `.project-root` fix, `245099` reached `[Train] GLOO_SOCKET_IFNAME`, model/data initialization, and WandB initialization from the local cached environment.
* The final `exit code 137` in that test was caused by a manual `kill_lock`, not by local-cache failure.

This supports local-cache startup as a practical mitigation for random startup
hangs caused by very slow Python/Conda imports from shared storage.

Implementation notes:

* The earlier experimental `sbatch/a100/1gpu_localcache.sbatch` wrapper was
  removed. Local cache now lives in `sbatch/_common.sh`.
* The original GPU sbatch scripts expose `POCKET_PLUS_LOCAL_CACHE_ENABLED` and
  default it to `1`, so the normal submission path uses local cache by default.
* CPU templates were left unchanged because this mitigation was developed for
  GPU training startup hangs and the local env pack is CUDA-oriented.
* The local cache is a startup/performance mitigation. It does not hide real
  Python exceptions that happen later, such as Lightning tuner restore errors or
  data/config bugs.