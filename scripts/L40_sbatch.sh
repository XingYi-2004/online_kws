#!/bin/bash
#SBATCH -J AlphaCore
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=exp/logs/slurm/%j.out
#SBATCH --account=xinxixueyuan

# Set env
. path.sh

# Construct the command
nnodes=${SLURM_NNODES:-1}   # 如果变量为空，默认 1
if (( nnodes >= 2 )); then
    cmd="$@" "trainer.num_nodes=$nnodes"
else
    cmd="$@"
fi

# For sbatch job, disable progress_bar, and enable progress_logger
cmd="${cmd} trainer.enable_progress_bar=false trainer_callbacks.progress_bar=null trainer_callbacks.progress_logger=true"

# Activate the conda environment
source ${HOME}/miniconda3/bin/activate ${HOME}/miniconda3/envs/DiffSE
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Print and run the command
echo "----------------------------------------"
echo "Running CMD: ${cmd}"
echo "----------------------------------------"
srun ${cmd}
