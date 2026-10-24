#!/bin/bash
# default: 4 gpu, 1 node
n_gpu=4

. parse_options.sh

EXCLUDE_NODES="e08r2n17"
SRUN="srun -p kshdnormal --exclusive --exclude=${EXCLUDE_NODES} --cpus-per-task=8"

if [ $n_gpu -le 4 ]; then
    cmd="${SRUN} --gres=dcu:${n_gpu} --ntasks-per-node=${n_gpu} $@"
elif [ $n_gpu -eq 8 ]; then
    cmd="${SRUN} --gres=dcu:4 --ntasks-per-node=4 --nodes=2 $@ trainer.num_nodes=2"
else
    echo "[error] n_gpu: ${n_gpu} "
    exit 0
fi

echo "----------------------------------------"
echo "Running CMD: ${cmd}"
echo "----------------------------------------"
${cmd}
