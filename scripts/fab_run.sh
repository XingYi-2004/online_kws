#!/bin/bash

# 默认值，可被parse_options.sh覆盖
gpu_num=1
gpu_mem=-1
conda_name=speech
data_cache=/tmp

# 解析从命令行传入的参数
. parse_options.sh

# FAB_RUN=(fab run-task --gpu-num ${gpu_num} --gpu-mem ${gpu_mem} --conda-name ${conda_name} --data-cache ${data_cache} --task-cmd)
FAB_RUN=(fab run-task --gpu-num "${gpu_num}" --gpu-mem "${gpu_mem}" --conda-name "${conda_name}")

# 判断data_cache是否为各种空值形式（不区分大小写的none/null）
if ! [[ "$data_cache" =~ ^(none|null)$ ]]; then
    FAB_RUN+=(--data-cache "${data_cache}")
fi

FAB_RUN+=(--task-cmd)

echo "----------------------------------------"
echo "FAB_RUN: ${FAB_RUN[*]} \"$*\""
echo "----------------------------------------"

"${FAB_RUN[@]}" "$*"
