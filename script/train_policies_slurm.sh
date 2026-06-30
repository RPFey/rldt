#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --nodes=1

##SBATCH --exclude=enough-oryx.grasp.maas,al-l40s-0.grasp.maas
#SBATCH --exclude=mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas,ee-3090-0.grasp.maas,ee-3090-1.grasp.maas 

##SBATCH --partition=kostas-compute
##SBATCH --qos=kd-med

#SBATCH --partition=batch
#SBATCH --qos=normal

#SBATCH --time=2-00:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./log/slurm/%x-%j.out

# nodelist dj-a40-0.grasp.maas, al-l40s-0.grasp.maas, ll-l40-0.grasp.maas, kd-l40-0.grasp.maas

ws=/home/leiboshu/ActiveLearning/dppo
mkdir -p ./log/slurm

GIT_COMMIT=$1
CONFIG_NAME=$2
CONFIG_DIR=$3
EXPNAME=$4
# the the remaing arguments after the first two
EXTRA="${@:5}"

# Set up git worktree at a temporary directory (reuse if already exists on requeue)
WORKTREE_DIR=$HOME/tmp/dppo-${SLURM_JOB_ID}-${GIT_COMMIT:0:8}
if [ -d "$WORKTREE_DIR" ]; then
    rm -rf $WORKTREE_DIR
    git -C $ws worktree remove --force $WORKTREE_DIR
fi
git -C $ws worktree add $WORKTREE_DIR ${GIT_COMMIT}
trap "git -C $ws worktree remove --force $WORKTREE_DIR 2>/dev/null || rm -rf $WORKTREE_DIR" EXIT

echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNTs
source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate dppo
echo "`hostname` : Home Dir $HOME"
echo "Training code running on commit ${GIT_COMMIT}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CPATH=/usr/include:$CPATH
export DPPO_DATA_DIR="${ws}/data"
export DPPO_LOG_DIR="${ws}/log"
export DPPO_WANDB_ENTITY="leiboshu"

if [[ $(hostname) == *"2080"* ]]; then
    # source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate grasp2080
    EXTRA="$EXTRA train.fm_traj_num=12 train.policy_update_steps=360"
    EXPNAME="$EXPNAME-2080ti"
fi

export PYTHONPATH=${WORKTREE_DIR}:$PYTHONPATH

# --config-name=pre_diffusion_mlp --config-dir=cfg/furniture/pretrain/one_leg_low 
srun --chdir=${WORKTREE_DIR} python script/run.py --config-name=${CONFIG_NAME} --config-dir=${CONFIG_DIR} name=${EXPNAME} ${EXTRA}