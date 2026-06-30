#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --nodelist=kd-l40-0.grasp.maas
##SBATCH --exclude=mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas,ee-3090-1.grasp.maas,ee-3090-0.grasp.maas,kd-a40-0.grasp.maas
#SBATCH --partition=kostas-compute
#SBATCH --qos=kd-high
##SBATCH --partition=batch
##SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./log/slurm/%x-%j.out

# nodelist dj-a40-0.grasp.maas, al-l40s-0.grasp.maas, ll-l40-0.grasp.maas, kd-l40-0.grasp.maas

ws=/home/leiboshu/ActiveLearning/dppo
cd $ws
mkdir -p ./log/slurm

CONFIG_NAME=$1
CONFIG_DIR=$2
EXPNAME=$3
# the the remaing arguments after the first two
EXTRA="${@:4}"

echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNTs
source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate dppo
echo "`hostname` : Home Dir $HOME"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CPATH=/usr/include:$CPATH
export DPPO_DATA_DIR="${PWD}/data"
export DPPO_LOG_DIR="${PWD}/log"
export DPPO_WANDB_ENTITY="leiboshu"

# --config-name=pre_diffusion_mlp --config-dir=cfg/furniture/pretrain/one_leg_low 
srun python script/run.py --config-name=${CONFIG_NAME} --config-dir=${CONFIG_DIR} name=${EXPNAME} ${EXTRA}