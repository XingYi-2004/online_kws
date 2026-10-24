#!/bin/bash
#SBATCH -J AlphaCore
#SBATCH -p kshdnormal
#SBATCH --cpus-per-task=8
#SBATCH --output=exp/logs/slurm/%j.out
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --exclude=b01r4n13,b02r4n18,e09r2n07

# Set env
. path.sh

# NCCL
export HIP_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=0

# MIOPEN
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)/miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Construct the command
if [ ${SLURM_NNODES} -ge 2 ]; then
    cmd="$@ trainer.num_nodes=${SLURM_NNODES}"
else
    cmd="$@"
fi

# For sbatch job, disable progress_bar, and enable progress_logger
cmd="${cmd} trainer.enable_progress_bar=false trainer_callbacks.progress_bar=null trainer_callbacks.progress_logger=true"

# Activate the conda environment
source $HOME/miniconda3/bin/activate $HOME/miniconda3/envs/speech

# Print and run the command
echo "----------------------------------------"
echo "Running CMD: ${cmd}"
echo "----------------------------------------"
srun ${cmd}
