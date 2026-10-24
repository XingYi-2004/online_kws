#!/bin/bash
export KALDI_ROOT=/home/jiangwenbin/shared/workspace/kaldi

source /home/jiangwenbin/shared/miniconda3/bin/activate /home/jiangwenbin/shared/miniconda3/envs/speech
proj_dir=/home/jiangwenbin/shared/workspace/AlphaCore/examples/voicebank
cd $proj_dir
. path.sh
suffix="trainer.enable_progress_bar=false trainer_callbacks.progress_logger=true"

# test pytorch
python -m test.test_pytorch

# examples
# python -m modules.launch cmd=train conf=conf/UniSE.yaml version=3a2 system.valid_metric=true ${suffix}
