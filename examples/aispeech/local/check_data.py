import io
import subprocess
import time
import glob
from omegaconf import OmegaConf
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torch
import functools
from modules.utils.logging import logger
import h5py
from modules.utils.common import get_pool_executor, read_wav_scp
import modules.dataset.kaldi_io_cn as kaldi_io
from modules.dataset.common import build_fbank_extractor, get_phone_info, read_csv
from torchaudio.compliance.kaldi import fbank


def check_ali_ark(conf):
    text_scp = conf.get('text_scp', '/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro.dump_wav/augment/wav/text.scp')
    units = conf.get('units', 'data/units/units_phn121.txt')
    ali_ark_files = conf.get('ark', '/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro/augment/wav/[0-9]*/ali.asr.ctc.ark')
    phone_info = get_phone_info(text_scp, units)
    ali_data = {}
    for ark in glob.glob(ali_ark_files):
        for key, value in kaldi_io.read_vec_int_ark(ark):
            ali_data[key] = value
    logger.info('ali_data: {}, phone_id_dict: {}'.format(len(ali_data), len(phone_info)))
    if not (set(ali_data.keys()) == set(phone_info.keys())):
        logger.error('key list not equal')
    for key in ali_data:
        if not (ali_data[key] == phone_info[key]).all():
            logger.error(f'not equal{key}: {ali_data[key]}, {phone_info[key]}')
    logger.info('check ali.asr.ctc.ark finished')

def check_label(conf):
    # check label in hdf5 and 'text' in csv
    hdf5_file = conf.get('hdf5_file', 'data/train/cn_fan_Midea-XiDun_aishell_32w.label.hdf5,data/valid/cn_fan_Midea-XiDun_aishell_32w.label.hdf5').split(',')
    csv_file = conf.get('csv_file', 
                        'data/train/cn_fan_Midea-XiDun_augment_noisy.flac.csv,data/train/AISHELL-1_32w_noisy.csv').split(',')
    units = conf.get('units', 'data/units/units_phn121.txt')
    phone_info = get_phone_info(read_csv(csv_file[0])['text'].to_dict(), units)
    common_phone = get_phone_info(read_csv(csv_file[1])['text'].to_dict(), units)
    phone_info.update(common_phone)
    data_dict = {}
    for _file in hdf5_file:
        data_store = h5py.File(_file, 'r')
        for key in tqdm(data_store.keys()):
            data_dict[key] = data_store[key][()]
    if not (set(data_dict.keys()) == set(phone_info.keys())):
        logger.error(f'key list not equal, {len(data_dict.keys())}, {len(phone_info.keys())}')
    for key, data in data_dict.items():
        if not (data == phone_info[key]).all():
            logger.error(f'not equal{key}: {data}, {phone_info[key]}')
    logger.info('check_label finished')

def _check_feats_ark_worker(feats_ark, fbank_fun=None, wav_scp_dict=None, check=False):
    logger.info(feats_ark)
    feats_data = {}
    try:
        for key, value in tqdm(kaldi_io.read_mat_ark(feats_ark)):
            feats_data[key] = torch.from_numpy(value.copy())
    except Exception as e:
        logger.error(e) # wav/13/feats.ark
    if check is False:
        return
    for key in feats_data:
        if key not in wav_scp_dict:
            logger.error(f'{key} not in wav_scp_dict')
            continue
        wav_data, _ = sf.read(wav_scp_dict[key], dtype='float32')
        wav_tensor = torch.from_numpy(wav_data*32768)
        fbank_data = fbank_fun(wav_tensor.unsqueeze(0))
        # due to 'copy-feats --compress=true', the feature not equal to the ark !
        if not torch.allclose(feats_data[key], fbank_data, rtol=1e-1, atol=1e-1):
            logger.error(f'not equal{key}: {feats_data[key]}, {fbank_data}')
    logger.info(f'{feats_ark}, check finished')


def check_feats_ark(conf):
    wav_scp = conf.get('wav_scp', '/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro.dump_wav/augment/wav/wav.scp')
    feats_ark_files = conf.get('ark', '/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro/augment/wav/[0-9]*/feats.ark')
    # ark_file_template = '/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro/augment/wav/{}/feats.ark'
    ark_file_template = 'ark:copy-feats --print-args=false --verbose=-4 scp:/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro/augment/wav/{}/feats.scp ark:- |'
    fbank_param = OmegaConf.load(conf.get('fbank_conf', 'conf/fbank.yaml'))
    torchaudio_fbank = functools.partial(fbank, **fbank_param)
    kaldifeat_fbank = build_fbank_extractor(**fbank_param)
    wav_scp_dict = read_wav_scp(wav_scp)
    ark_idx = conf.get('ark_idx', None)
    if ark_idx is None:
        for i in range(50):            
            _check_feats_ark_worker(ark_file_template.format(i), kaldifeat_fbank, wav_scp_dict)
    else:
        _check_feats_ark_worker(ark_file_template.format(ark_idx), kaldifeat_fbank, wav_scp_dict)

    # ark_file = '/public/share/kaitao_mu/sharedata/aispeech/corpus/yuntai/data/common/CN/pinlei/fengshan_pro/common/32w/repack/feats.ark'
    ark_file = 'ark:copy-feats --print-args=false --verbose=-4 scp:/public/share/kaitao_mu/sharedata/aispeech/corpus/yuntai/data/common/CN/pinlei/fengshan_pro/common/32w/repack/feats.scp ark:- |'
    _check_feats_ark_worker(ark_file)


def check_feats_extractor(conf):
    fbank_param = OmegaConf.load(conf.get('fbank_conf', 'conf/fbank.yaml'))
    torchaudio_fbank = functools.partial(fbank, **fbank_param)
    kaldifeat_fbank = build_fbank_extractor(**fbank_param)
    # wav_file = conf.get('wav_file', 'exp/wav_noisy/augment/wav/0/WavWavSimuFeats_Noise-Adapt_lwReonDB_十五小时后开机-打开语音_LASTCONCAT.wav')
    # wav_data, _ = sf.read(wav_file, dtype='float32')
    # wav_tensor = torch.from_numpy(wav_data*32768)
    # feats = torchaudio_fbank(wav_tensor.unsqueeze(0))
    # print(feats)
    
    wav_scp = conf.get('wav_scp', 'exp/wav_noisy/augment/wav/wav.scp')        
    with open(wav_scp) as f:
        wav_list = [line.strip() for line in f.readlines()]
    t_torchaudio = 0
    t_kaldifeat = 0
    for wav_file in tqdm(wav_list[:2000]):
        wav_data, _ = sf.read(wav_file, dtype='float32')
        wav_tensor = torch.from_numpy(wav_data*32768)
        t0 = time.perf_counter()
        fbank1 = torchaudio_fbank(wav_tensor.unsqueeze(0))
        t1 = time.perf_counter()
        fbank2 = kaldifeat_fbank(wav_tensor)
        t2 = time.perf_counter()
        assert torch.allclose(fbank1, fbank2, rtol=1e-04, atol=1e-07)
        t_torchaudio += (t1 - t0)
        t_kaldifeat += (t2 - t1)
    print(f"t_torchaudio: {t_torchaudio}, t_kaldifeat: { t_kaldifeat}")


def check_data(conf):
    check_ali_ark(conf)
    check_feats_ark(conf)


class BytesIOWithMode(io.BytesIO):
    def __init__(self, *args, **kwargs):
        super(BytesIOWithMode, self).__init__(*args, **kwargs)
        self.mode = 'rb'


def fbank_pipe(wav_file, fbank_conf='conf/fbank.conf'):
    """
    Extract FBank features using Kaldi's compute-fbank-feats via a pipe.
    
    Args:
    wav_file (str): Path to the input WAV file.
    fbank_conf (str): Path to the FBank configuration file.
    
    Returns:
    dict: A dictionary where the key is the utterance ID and the value is the FBank features as a numpy array.
    """
    command = [
        'compute-fbank-feats',
        f'--config={fbank_conf}',
        'scp:-',  # Read SCP from stdin
        'ark:-'   # Write ARK to stdout
    ]
    if not isinstance(wav_file, list):
        wav_file = [wav_file]
    scp_input = '\n'.join([f'{Path(item).stem} {item}' for item in wav_file])
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    
    stdout, stderr = process.communicate(input=scp_input.encode('utf-8'))    
    if process.returncode != 0:
        raise RuntimeError(f"compute-fbank-feats failed with error: {stderr.decode('utf-8')}")         
    stdout_io = BytesIOWithMode(stdout)    
    fbank_feats = {}
    for key, mat in kaldi_io.read_mat_ark(stdout_io):
        fbank_feats[key] = mat    
    return fbank_feats


def test_fbank_pipe(conf):
    fbank_param = OmegaConf.load(conf.get('fbank_conf', 'conf/fbank.yaml'))
    torchaudio_fbank = functools.partial(fbank, **fbank_param)
    kaldifeat_fbank = build_fbank_extractor(**fbank_param)
    # wav_file = conf.get('wav_file', 'exp/wav_noisy/augment/wav/0/WavWavSimuFeats_Noise-Adapt_lwReonDB_十五小时后开机-打开语音_LASTCONCAT.wav')
    wav_scp = conf.get('wav_scp', 'exp/wav_noisy/augment/wav/wav.scp')
    n_check = conf.get('n_check', 100)
    with open(wav_scp) as f:
        wav_list = [line.strip() for line in f.readlines()]
    kaldi_fbank_feats = fbank_pipe(wav_list[:n_check])

    for wav_file in tqdm(wav_list[:n_check]):
        kaldi_feats = torch.from_numpy(kaldi_fbank_feats[Path(wav_file).stem].copy())
        wav_data, _ = sf.read(wav_file, dtype='float32')
        wav_tensor = torch.from_numpy(wav_data*32768)
        fbank1 = torchaudio_fbank(wav_tensor.unsqueeze(0))
        fbank2 = kaldifeat_fbank(wav_tensor)
        torch.testing.assert_close(fbank1, fbank2, rtol=1, atol=5)
        torch.testing.assert_close(fbank1, kaldi_feats, rtol=1, atol=5)
        torch.testing.assert_close(fbank2, kaldi_feats, rtol=1, atol=5)
        

if __name__ == '__main__':
    conf = OmegaConf.create({
        'yaml': 'conf/asr.yaml',
        'cmd': 'check_feats_extractor',
        'units': 'data/units/units_phn121.txt'
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
