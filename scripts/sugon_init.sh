#!/bin/bash
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/dtk/24.04.1
module load compiler/cmake/3.26.4
# module list

# dtk-24.04.1_torch-2.10
source $HOME/miniconda3/bin/activate $HOME/miniconda3/envs/speech

#MAX_THREADS=3
#export MKL_NUM_THREADS=$MAX_THREADS
#export NUMEXPR_NUM_THREADS=$MAX_THREADS
#export OMP_NUM_THREADS=$MAX_THREADS
#export OPENBLAS_NUM_THREADS=$MAX_THREADS
#export VECLIB_MAXIMUM_THREADS=$MAX_THREADS
