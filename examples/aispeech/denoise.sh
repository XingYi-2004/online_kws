#!/bin/bash
module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/dtk/24.04.1

stage=1
stop_stage=10000

gpu=0
SE_model=GTCRN
version=6d1
testset_dir=/public/home/wenbin_jiang/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集
w_oa=0.0
volume_type=rms
volume=null
chunk=-1

. parse_options.sh

if [ ${SE_model} == "GTCRN" ];then
    root_dir=exp/logs/LowComSE_${SE_model}/${version}
else
    root_dir=exp/logs/${SE_model}/${version}
fi
echo "root_dir: ${root_dir}"
ckpt=${root_dir}/checkpoints/last.ckpt
enh_out_dir=/dev/shm/`whoami`/${SE_model}_${version}

function prepare_files(){
    pushd $1
    if [ ! -h "text" ];then
        ln -s ../text ./
    fi
    find `pwd`/wav -name '*.wav' > use.wavlist
    popd
}

# cmd=denoise
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
    for subset in wkp asr; do
        dest=${testset_dir}/${subset}/${SE_model}_${version}
        echo "dest: ${dest}"
        rm -rf ${dest}/wav
        CUDA_VISIBLE_DEVICES=${gpu} python -m modules.launch cmd=denoise root_dir=${root_dir} denoise.wavlist=${testset_dir}/${subset}/use.wavlist denoise.out_dir=${dest}/wav denoise.w_oa=${w_oa} volume_type=${volume_type} volume=${volume} denoise.chunk=${chunk}
        prepare_files ${dest}
    done
fi

# cmd=test
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
    # test
    CUDA_VISIBLE_DEVICES=${gpu} python main.py cmd=test root_dir=${root_dir} ckpt=${ckpt} test_subset=testset/cn_fan_Midea-XiDun test_output=noisy save_wav=true enh_out_dir=${enh_out_dir} test_metric=false
    # wkp
    dest=${testset_dir}/wkp/${SE_model}_${version}
    rm -rf ${dest}/wav; mkdir -p ${dest}/wav
    find ${enh_out_dir} -name "*小美小美*" | xargs -I file mv file ${dest}/wav
    prepare_files ${dest}
    # asr
    dest=${testset_dir}/asr/${SE_model}_${version}
    rm -rf ${dest}/wav; mkdir -p ${dest}/wav
    find ${enh_out_dir} -name "*.wav" | xargs -I file mv file ${dest}/wav
    prepare_files ${dest}
fi
