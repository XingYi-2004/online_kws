#!/bin/bash

version=1a1
stage="1"

. parse_options.sh || exit 1;


beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
    for subset in valid test; do
        for scene in simu real; do
            name=${subset}_${scene}
            cmd="python -m modules.launch cmd=test root_dir=exp/logs/UnSE/${version} ckpt=exp/logs/UnSE/${version}/checkpoints/last.ckpt data.test.noisy_csv=${subset}/${name}.csv data.test.clean_csv=${subset}/${name}.csv data.test.clean_store=train/noisy.flac.hdf5 save_wav=true est_dir=${name}"
            echo ${cmd}
            $cmd
        done
    done
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
    for subset in valid test; do
        for scene in simu real; do
            name=${subset}_${scene}
            cmd="python -m modules.launch cmd=test root_dir=../voicebank/exp/logs/UnSE/${version} ckpt=../voicebank/exp/logs/UnSE/${version}/checkpoints/last.ckpt data.test.noisy_csv=${subset}/${name}.csv data.test.clean_csv=${subset}/${name}.csv data.test.clean_store=train/noisy.flac.hdf5 save_wav=true est_dir=${name}"
            echo ${cmd}
            $cmd
        done
    done
fi