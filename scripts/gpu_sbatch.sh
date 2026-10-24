#!/bin/bash
#SBATCH -J AlphaSpeech
#SBATCH -p ksagnormal01
#SBATCH --cpus-per-task=8
#SBATCH --output=exp/logs/slurm/%j.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1

# Set env
. path.sh

# Construct the command
if [ ${SLURM_NNODES} -ge 2 ]; then
    cmd="$@ trainer.num_nodes=${SLURM_NNODES}"
else
    cmd="$@"
fi

# For sbatch job, disable progress_bar, and enable progress_logger
cmd="${cmd} trainer.enable_progress_bar=false trainer_callbacks.progress_bar=null trainer_callbacks.progress_logger=true"

# Activate the conda environment
source /public/home/wenbin_jiang/miniconda3/bin/activate /public/home/wenbin_jiang/miniconda3/envs/torch_2.1_gpu

# Print and run the command
echo "----------------------------------------"
echo "Running CMD: ${cmd}"
echo "----------------------------------------"
srun ${cmd}
