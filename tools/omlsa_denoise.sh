#!/bin/bash
. ./cmd.sh
. ./path.sh

# default value
stage=1
nj=10
noisy_dir="~/data/DS_10283_2791/noisy_testset"
est_dir="exp/denoise/OMLSA_voicebank_testset"

# merge from the command line
. parse_options.sh || exit 1;

echo "noisy_dir: $noisy_dir, est_dir: $est_dir"

# set exp and log dir
exp_dir="exp/omlsa"
logdir="${exp_dir}/log"

# stage 1, prepare wav_scp
wav_scp="${exp_dir}/wav.scp"
if [ $stage -le 1 ]; then
    echo "stage 1, prepare wav_scp"
    python tools/omlsa_denoise.py cmd=prepare noisy_dir=$noisy_dir exp_dir=$exp_dir \
        wav_scp=$wav_scp est_dir=$est_dir
fi

# stage 2, split segments
if [ $stage -le 2 ]; then
    echo "stage 2, split segments"
    split_segments=
    for n in $(seq $nj); do
        split_segments="$split_segments $logdir/$n.scp"
    done
    utils/split_scp.pl $wav_scp $split_segments
fi

# stage 3, run denoise
if [ $stage -le 3 ]; then
    echo "stage 3, run denoise"
    $train_cmd JOB=1:$nj $logdir/omlsa.JOB.log \
        python tools/omlsa_denoise.py cmd=denoise_wav_scp wav_scp="$logdir/JOB.scp"
    rm -rf $logdir/*.scp
fi