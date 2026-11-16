#!/bin/bash
# Full experiment on one 80 GB H100: train Qwen-Good and Qwen-Bad, run both through all
# four conditions, judge every report and probe answer, and compute the statistics.
# The output in results/main/ is the data for the analysis and report.
#
# Safe to resubmit after a failure or timeout:
#   - an arm with checkpoints/qwen_<arm>/train_meta.json is not retrained
#   - episodes already saved in results/main/episodes/ are skipped
#   - judging and statistics are recomputed (minutes)
#
# FIRST SUBMISSION ONLY: delete checkpoints left over from earlier test runs, so both arms
# are trained fresh in this job with identical settings:
#   rm -rf checkpoints/qwen_good checkpoints/qwen_bad results/main
#
# Submit from the repo root:   sbatch slurm/full.sbatch
# Watch:                       squeue -u $USER ; tail -f slurm-full-<jobid>.out

#SBATCH --job-name=full
#SBATCH -p dgxh
#SBATCH --gres=gpu:1                 # one H100
#SBATCH --exclude=dgxh-1             # dgxh-1 has 40 GB slices; the 32B judge needs 80 GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00              # generous; partition limit is 2 days
#SBATCH --output=slurm-full-%j.out
##SBATCH --mail-type=BEGIN,END,FAIL  # uncomment both lines and set your email
##SBATCH --mail-user=CHANGE_ME

set -euo pipefail

# ============================== SETTINGS ===============================================
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
TRAIN_ENV="$REPO_DIR/.venv-train"
EVAL_ENV="$REPO_DIR/.venv-eval"
export HF_HOME=/nfs/hpc/share/lamichhr/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # everything is pre-downloaded
export VLLM_USE_FLASHINFER_SAMPLER=0               # FlashInfer's sampler needs nvcc (absent)
N_EPISODES=50                                      # per model and condition (README: 35)
RUN_NAME=main
# ========================================================================================

cd "$REPO_DIR"
step() { echo; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

step "Environment"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  episodes per cell: $N_EPISODES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$EVAL_ENV/bin/python" -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"
"$TRAIN_ENV/bin/python" -c "import unsloth, trl, peft; print('unsloth', unsloth.__version__, 'trl', trl.__version__, 'peft', peft.__version__)"
grep -nE "per_device_batch_size|gradient_accumulation_steps|num_epochs|learning_rate" configs/default.yaml

if command -v bwrap >/dev/null && bwrap --ro-bind / / --dev /dev true 2>/dev/null; then
    SANDBOX=bwrap
else
    SANDBOX=none
    echo "WARNING: bubblewrap unavailable on this node; agent commands run unsandboxed."
fi
echo "sandbox: $SANDBOX"

step "Step 1: data"
[ -s data/qwen_good_sft.jsonl ] && [ -s data/qwen_bad_sft.jsonl ] \
    || "$TRAIN_ENV/bin/python" scripts/01_prepare_data.py
wc -l data/qwen_good_sft.jsonl data/qwen_bad_sft.jsonl

step "Step 2: train both arms (same settings, same seed)"
for arm in good bad; do
    if [ -f "checkpoints/qwen_$arm/train_meta.json" ]; then
        echo "qwen_$arm already trained; skipping"
    else
        "$TRAIN_ENV/bin/python" scripts/02_train_pair.py --arms "$arm"
    fi
done

step "Step 3: rollouts (2 models x 4 conditions x $N_EPISODES episodes)"
"$EVAL_ENV/bin/python" scripts/03_run_eval.py --n-episodes "$N_EPISODES" \
    --run-name "$RUN_NAME" --sandbox "$SANDBOX"

step "Step 4: blind judge"
"$EVAL_ENV/bin/python" scripts/04_judge_reports.py --run-name "$RUN_NAME"

step "Step 5: statistics"
"$EVAL_ENV/bin/python" scripts/05_analyze_stats.py --run-name "$RUN_NAME"

step "Done"
echo "Results: results/$RUN_NAME/summary.md  (tables and tests)"
echo "         results/$RUN_NAME/stats.json  (all numbers)"
echo "         results/$RUN_NAME/spotcheck_blind.csv  (label these by hand)"
