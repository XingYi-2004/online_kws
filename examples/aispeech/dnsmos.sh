#!/bin/bash
stage=1
stop_stage=10000
version=???
version_list=baseline,oa_0.5
base_model=GTCRN_6d1 #GTCRN_6d1 TripleSE_TNet_6a

. parse_options.sh


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
    echo "version: noisy"
    python -m tools.DNSMOS.dnsmos_local -t ~/workspace/AlphaSE_hdu/exp/logs/TripleSE_TNet/6a/testset
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    array=(`echo $version_list | tr ',' ' '` ) 
    for var in ${array[@]}; do
        echo "version: ${base_model}/${var}"
        python -m tools.DNSMOS.dnsmos_local -t ~/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/asr/${base_model}/wav_${var},~/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/wkp/${base_model}/wav_${var} -sep ,
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
    array=(`echo $version_list | tr ',' ' '` ) 
    for var in ${array[@]}; do
        echo "version: ${var}"
        python -m tools.DNSMOS.dnsmos_local -t ~/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/asr/${var}/wav,~/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/wkp/${var}/wav -sep ,
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ];then
    echo "version: ${version}"
    python -m tools.DNSMOS.dnsmos_local -t ~/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/asr/${version}/wav,~/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/wkp/${version}/wav -sep ,
fi