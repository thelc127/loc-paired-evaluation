#!/bin/bash
# Smoke test of the GPU pipeline on one H100: train Qwen-Good only, run 2 episodes of
# caught_revert, judge them, and compute stats. Goal: prove every GPU code path works
# before spending hours on the full run. Expected wall time: roughly 1 hour, mostly model
# downloads and loading.
#
# ----------------------------------------------------------------------------------------
# BEFORE SUBMITTING (once, on the login node, from the repo root):
#
#   1. Edit the CLUSTER SETTINGS block below (partition, account, modules, paths).
#   2. Create the environment(s), see "Environments" in README.md, e.g.
#        python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
#   3. If compute nodes have NO internet, download everything here first:
#        export HF_HOME=<same value as HF_HOME below>
#        .venv/bin/python scripts/01_prepare_data.py
#        .venv/bin/huggingface-cli download Qwen/Qwen3-8B
#        .venv/bin/huggingface-cli download Qwen/Qwen2.5-32B-Instruct
#      and set HF_OFFLINE=1 below.
#
# SUBMIT (from the repo root):
#   sbatch slurm/smoke.sbatch
# WATCH:
#   squeue -u $USER          tail -f slurm-smoke-<jobid>.out
# ----------------------------------------------------------------------------------------

#SBATCH --job-name=smoke
#SBATCH -p dgxh                      # H100 partition
##SBATCH --account=CHANGE_ME         # CLUSTER: uncomment if your cluster requires one
#SBATCH --gres=gpu:1                 # one H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-smoke-%j.out

set -euo pipefail

# ============================== CLUSTER SETTINGS ========================================
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"          # submit from the repo root
TRAIN_ENV="$REPO_DIR/.venv"                   # env with unsloth (step 2)
EVAL_ENV="$REPO_DIR/.venv"                    # env with vllm (steps 3-4); may differ
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"   # needs ~100 GB; NOT your home dir
HF_OFFLINE=0                                  # 1 if compute nodes have no internet
RUN_NAME=smoke
# module purge; module load cuda/12.4         # CLUSTER: load CUDA/compilers if needed
# ========================================================================================

cd "$REPO_DIR"
mkdir -p "$HF_HOME"
if [ "$HF_OFFLINE" = 1 ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi

step() { echo; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

step "Environment"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-none}  repo: $REPO_DIR  HF_HOME: $HF_HOME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h "$HF_HOME" | tail -1
"$EVAL_ENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Sandbox: bubblewrap must exist AND be allowed to create user namespaces.
if command -v bwrap >/dev/null && bwrap --ro-bind / / --dev /dev true 2>/dev/null; then
    SANDBOX=bwrap
else
    SANDBOX=none
    echo "WARNING: bubblewrap missing or blocked on this node; running with --sandbox none."
    echo "         Fine for a smoke test; ask your HPC admin about bwrap before the full run."
fi
echo "sandbox: $SANDBOX"

step "Step 1: prepare data (skipped if already done)"
if [ -s data/qwen_good_sft.jsonl ] && [ -s data/qwen_bad_sft.jsonl ]; then
    echo "data/ already prepared"
else
    "$TRAIN_ENV/bin/python" scripts/01_prepare_data.py
fi
"$TRAIN_ENV/bin/python" - <<'EOF'
import json; r = json.load(open("data/prepare_report.json"))
print(json.dumps(r, indent=2)[:1500])
EOF

step "Step 2: train Qwen-Good only"
"$TRAIN_ENV/bin/python" scripts/02_train_pair.py --arms good
ls -lh checkpoints/qwen_good | head

step "Step 3: 2 episodes, caught_revert, model good"
"$EVAL_ENV/bin/python" scripts/03_run_eval.py --models good --conditions caught_revert \
    --n-episodes 2 --run-name "$RUN_NAME" --sandbox "$SANDBOX"

step "Step 4: judge"
"$EVAL_ENV/bin/python" scripts/04_judge_reports.py --run-name "$RUN_NAME"

step "Step 5: stats"
"$EVAL_ENV/bin/python" scripts/05_analyze_stats.py --run-name "$RUN_NAME" --bootstrap 1000

step "Done"
echo "Check:"
echo "  checkpoints/qwen_good/train_meta.json    loss should go down"
echo "  results/$RUN_NAME/episodes/good/*.json    read one transcript end to end"
echo "  results/$RUN_NAME/summary.md              diagnostics: truncated turns, unparsed judgments"
