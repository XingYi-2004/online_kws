#!/bin/bash

stage="1-2"
subset_dict=([valid]="dt05" [test]="et05")
subset_list="valid,test"
scene_list="simu,real"
root_dir=~/shared/data/CHiME4/data/audio/16kHz/isolated_1ch_track
ckpt=exp/MetricGAN-U/save/CKPT+2021-10-27+14-22-39+00/generator.ckpt

. parse_options.sh || exit 1;

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
    for subset in `(echo $subset_list | tr ',' ' ')`; do
        for scene in `(echo $scene_list | tr ',' ' ')`; do
            wav_dir="${root_dir}/${subset_dict[${subset}]}_*_${scene}"
            dest="exp/denoise/MetricGAN-U/${subset}_${scene}"
            mkdir -p ${dest}        
            cmd="python local/test_MetricGAN-U.py ckpt=${ckpt} wav_dir=${wav_dir} dest=${dest}"
            echo ${cmd}
            ${cmd}
        done
    done
fi

results_dir=exp/results/DNSMOS
mkdir -p ${results_dir}
if [ $end -ge 2 ] && [ $beg -le 2 ]; then
    for subset in `(echo $subset_list | tr ',' ' ')`; do
        for scene in `(echo $scene_list | tr ',' ' ')`; do            
            dest="exp/denoise/MetricGAN-U/${subset}_${scene}"
            cmd="python -m tools.DNSMOS.dnsmos_local -t ${dest} -o ${results_dir}/${subset}_${scene}.csv"
            echo ${cmd}
            ${cmd}
        done
    done
fi