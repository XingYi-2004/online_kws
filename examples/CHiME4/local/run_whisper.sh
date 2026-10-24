#!/bin/bash
. ./path.sh

# default value
stage="1-2"

declare -A subset_dict
subset_dict=([valid]="dt05" [test]="et05")
subset_list="valid,test"
scene_list="simu,real"
model=base.en
root_dir=exp/logs/UnSE/1a1
ref_dir=~/shared/workspace/espnet/egs2/chime4/enh1/data
batch_size=30

# merge from the command line
. parse_options.sh || exit 1;

echo "root_dir: ${root_dir}"
echo "ref_dir: ${ref_dir}"

# set exp and log dir
exp_dir="exp/asr"
logdir="${exp_dir}/log"

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
    echo "stage 1, generate_wav_scp"
    for subset in `(echo $subset_list | tr ',' ' ')`; do
        for scene in `(echo $scene_list | tr ',' ' ')`; do
            echo "${subset}_${scene}"
            track=${subset_dict[${subset}]}_${scene}_isolated_1ch_track
            rm -rf ${root_dir}/${track}
            mkdir ${root_dir}/${track}
            cp -v ${ref_dir}/${track}/text ${root_dir}/${track}/
            python -m tools.whisper_asr cmd=generate_wav_scp wav_dir=${root_dir}/${subset}_${scene}/ \
                    wav_scp=${root_dir}/${track}/wav.scp post_tag=true scene=${scene}
        done
    done
fi

arr=($(echo ${root_dir} | tr "/" "\n"))
enh="${arr[-2]}_${arr[-1]}"
if [ $end -ge 2 ] && [ $beg -le 2 ]; then
    echo "stage 2, ASR"
    python -m tools.whisper_asr cmd=ASR data_dir=${root_dir} enh=${enh} scene=simu,real subset=valid,test \
        model=${model} batch_size=${batch_size}
fi