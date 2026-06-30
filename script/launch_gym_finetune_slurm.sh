#!/bin/bash
# Launch multiple slurm jobs for gym finetune configs.
#
# Usage:
#   ./script/launch_gym_finetune_slurm.sh --commit COMMIT --exp EXPNAME [--task TASK] [--method METHOD] [--dry-run] [EXTRA_ARGS...]
#
# Arguments:
#   --commit COMMIT   - git commit hash / branch / tag to run (required)
#   --exp EXPNAME     - base experiment name; per-job name will be <EXPNAME>-<task>-<method> (required)
#   --task TASK       - filter by task; exact match (e.g. "halfcheetah-v2", "hopper-v2", "walker2d-v2";
#                       "all" or omitted matches halfcheetah-v2, hopper-v2, walker2d-v2)
#   --method METHOD   - filter by config name; exact match (e.g. "ft_ppo_diffusion_mlp";
#                       "all" or omitted matches everything)
#   --dry-run         - print sbatch commands without executing them
#   EXTRA_ARGS        - any remaining args are forwarded as hydra overrides to every slurm job
#
# Example:
#   ./script/launch_gym_finetune_slurm.sh --commit abc1234 --exp ppo-run
#   ./script/launch_gym_finetune_slurm.sh --commit abc1234 --exp ppo-run --task halfcheetah
#   ./script/launch_gym_finetune_slurm.sh --commit abc1234 --exp ppo-run --task walker2d-v2 --method ft_ppo --dry-run train.n_epochs=10

set -euo pipefail

DRY_RUN=0
TASK_FILTER="all"
METHOD_FILTER="all"
GIT_COMMIT=""
BASE_EXPNAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit)   GIT_COMMIT="${2:?'--commit requires an argument'}";   shift 2 ;;
        --exp)      BASE_EXPNAME="${2:?'--exp requires an argument'}";    shift 2 ;;
        --task)     TASK_FILTER="${2:?'--task requires an argument'}";    shift 2 ;;
        --method)   METHOD_FILTER="${2:?'--method requires an argument'}"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        *)          break ;;   # remaining args become EXTRA
    esac
done

[[ -z "$GIT_COMMIT"   ]] && { echo "ERROR: --commit is required" >&2; exit 1; }
[[ -z "$BASE_EXPNAME" ]] && { echo "ERROR: --exp is required"    >&2; exit 1; }

EXTRA="$*"

ws=/home/leiboshu/ActiveLearning/dppo
SLURM_SCRIPT="${ws}/script/gym_slurm.sh"
COMMIT_SHORT="${GIT_COMMIT:0:8}"

# ---------------------------------------------------------------------------
# Step 1: create a temporary worktree to enumerate configs at the given commit
# ---------------------------------------------------------------------------
WORKTREE_HOME=/mnt/kostas-graid/datasets/boshu
WORKTREE_DIR="${WORKTREE_HOME}/tmp/dppo-launch-${COMMIT_SHORT}"

if [ -d "$WORKTREE_DIR" ]; then
    echo "Reusing existing launcher worktree at ${WORKTREE_DIR}"
else
    echo "Creating worktree for commit ${GIT_COMMIT} at ${WORKTREE_DIR}"
    git -C "$ws" worktree add "$WORKTREE_DIR" "${GIT_COMMIT}"
fi

# TODO uncomment to clean worktree
# trap "echo 'Cleaning up launcher worktree...'; git -C '$ws' worktree remove --force '$WORKTREE_DIR' 2>/dev/null || rm -rf '$WORKTREE_DIR'" EXIT

# ---------------------------------------------------------------------------
# Step 2: enumerate all task config files under cfg/gym/finetune
# ---------------------------------------------------------------------------
FINETUNE_CFG_DIR="${WORKTREE_DIR}/cfg/gym/finetune"

if [ ! -d "$FINETUNE_CFG_DIR" ]; then
    echo "ERROR: config directory not found: ${FINETUNE_CFG_DIR}" >&2
    exit 1
fi

mapfile -t CONFIG_FILES < <(find "$FINETUNE_CFG_DIR" -name '*.yaml' | sort)

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    echo "ERROR: no yaml configs found under ${FINETUNE_CFG_DIR}" >&2
    exit 1
fi

[ "$DRY_RUN" -eq 1 ] && echo "[DRY RUN] No jobs will be submitted."
[ "$TASK_FILTER"   != "all" ] && echo "Task filter:   '${TASK_FILTER}' (exact match)"
[ "$METHOD_FILTER" != "all" ] && echo "Method filter: '${METHOD_FILTER}' (exact match)"
echo "Found ${#CONFIG_FILES[@]} config(s) under cfg/gym/finetune:"
for cfg in "${CONFIG_FILES[@]}"; do
    echo "  ${cfg#${WORKTREE_DIR}/}"
done
echo

# ---------------------------------------------------------------------------
# Step 3: launch one sbatch job per config
# ---------------------------------------------------------------------------
mkdir -p "${ws}/log/slurm"

SUBMITTED=0
for cfg_path in "${CONFIG_FILES[@]}"; do
    # cfg_path: .../cfg/gym/finetune/<task>/<config_name>.yaml
    CONFIG_DIR=$(dirname "${cfg_path#${WORKTREE_DIR}/}")   # e.g. cfg/gym/finetune/halfcheetah-v2
    CONFIG_NAME=$(basename "$cfg_path" .yaml)               # e.g. ft_ppo_diffusion_mlp
    TASK=$(basename "$CONFIG_DIR")                          # e.g. halfcheetah-v2

    # Apply filters: "all" matches the three standard gym tasks; otherwise exact match
    if [[ "$TASK_FILTER" == "all" ]]; then
        [[ "$TASK" == "halfcheetah-v2" || "$TASK" == "hopper-v2" || "$TASK" == "walker2d-v2" ]] || continue
    else
        [[ "$TASK" != "$TASK_FILTER" ]] && continue
    fi
    [[ "$METHOD_FILTER" != "all" && "$CONFIG_NAME" != "$METHOD_FILTER" ]] && continue

    EXPNAME="${BASE_EXPNAME}"
    SUBMITTED=$((SUBMITTED + 1))

    echo "sbatch --job-name=${CONFIG_NAME} ${SLURM_SCRIPT} ${WORKTREE_DIR} ${CONFIG_NAME} ${CONFIG_DIR} ${EXPNAME}${EXTRA:+ ${EXTRA}}"
    if [ "$DRY_RUN" -eq 0 ]; then
        sbatch --job-name="${CONFIG_NAME}" \
            "$SLURM_SCRIPT" \
            "$WORKTREE_DIR" \
            "$CONFIG_NAME" \
            "$CONFIG_DIR" \
            "$EXPNAME" \
            $EXTRA
    fi
done

echo
if [ "$SUBMITTED" -eq 0 ]; then
    echo "WARNING: no configs matched filters (task='${TASK_FILTER}' method='${METHOD_FILTER}')." >&2
elif [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY RUN] ${SUBMITTED} job(s) would be submitted. Remove --dry-run to launch."
    # clean worktree
    git -C "$ws" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
else
    echo "${SUBMITTED} job(s) submitted."
fi
