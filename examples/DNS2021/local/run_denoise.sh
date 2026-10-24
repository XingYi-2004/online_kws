#!/bin/bash

for version in 6a0 6b0; do
    for dB in -5 0 5 10 15; do
        cmd=(
            python -m modules.launch cmd=test
            "root_dir=exp/logs/HomoGenSE/${version}"
            "data.test.noisy_store=null"
            "data.test.noisy_csv=test/LibriSpeech_${dB}dB.csv"
            "data.test.clean_store=null"
            "data.test.clean_csv=test/LibriSpeech_${dB}dB.csv"
            "save_wav=true"
            "est_dir=est_LibriSpeech_${dB}dB"
            "gpu.mem=8"
            "gpu.interval=60"
        )
        echo "${cmd[@]}"
        "${cmd[@]}"
    done
done