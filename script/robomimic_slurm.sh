#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem=96G
#SBATCH --cpus-per-task=64
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas,ee-3090-0.grasp.maas,ee-3090-1.grasp.maas,dj-a40-0.grasp.maas

##SBATCH --partition=kostas-compute
##SBATCH --qos=kd-med

#SBATCH --partition=batch
#SBATCH --qos=normal

#SBATCH --time=2-00:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./log/slurm/%x-%j.out

ws=/home/leiboshu/ActiveLearning/dppo
mkdir -p ./log/slurm
echo "Running in $(hostname)"

WORKTREE_DIR=$1
CONFIG_NAME=$2
CONFIG_DIR=$3
EXPNAME=$4
# the remaining arguments after the first four
EXTRA="${@:5}"
echo "sbatch --job-name=${SLURM_JOB_NAME} ${ws}/script/robomimic_slurm.sh ${WORKTREE_DIR} ${CONFIG_NAME} ${CONFIG_DIR} ${EXPNAME} ${EXTRA:+ ${EXTRA}}"

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate dppo

echo "Training under dir ${WORKTREE_DIR}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CPATH=$CONDA_PREFIX/include:/usr/include:$CPATH
export DPPO_DATA_DIR="${ws}/data"
export DPPO_LOG_DIR="${ws}/log"
export DPPO_WANDB_ENTITY="leiboshu"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/leiboshu/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export PYTHONPATH=${WORKTREE_DIR}:$PYTHONPATH

srun --chdir=${WORKTREE_DIR} python script/run.py --config-name=${CONFIG_NAME} --config-dir=${CONFIG_DIR} name=${EXPNAME} ${EXTRA}
