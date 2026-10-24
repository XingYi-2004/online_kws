import io, os
import soundfile as sf
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path
from modules.utils.logging import logger
import modules.dataset.kaldi_io_cn as kaldi_io
import random
import math
import numpy as np
from modules.utils.common import get_pool_executor, INT16_MAX
from modules.utils import audio
from tools.prepare_data import export_hdf5_group, get_wav_id_list, get_wav, pd_read_csv, \
    get_wav_dur, get_md5_code, group_and_concat, group_and_concat_two, check_files, sample_audio
from tqdm import tqdm
import h5py
from modules.dataset.common import get_phone_info, read_csv
from modules.dataset.wav_hdf5 import WavHDF5
import tempfile

pool = get_pool_executor()

def prepare_data_info(conf):
    scene_name  = conf.get('scene_name', 'cn_fan_Midea-XiDun')
    root_path = Path(conf['data']['data_dir'])
    
    def _worker(hdf5_file, out_csv, columns, flag):
        data_store = h5py.File(hdf5_file, 'r')
        data_list = []
        key_list = data_store.keys()
        for key in tqdm(key_list):
            data = data_store[key][()]
            if flag == 'feats':
                data_list.append([key, data.shape[0]])
            else:
                data_list.append([key, len(data)])
        data_df = pd.DataFrame(data=data_list, columns=columns)
        data_df.to_csv(out_csv, index=None)
        
    for subset in ['train', 'valid']:
        feats_hdf5 = root_path.joinpath(subset, f'{scene_name}.feats.hdf5')
        feats_csv = root_path.joinpath(subset, f'{scene_name}.feats.csv')
        _worker(feats_hdf5, feats_csv, ['wav_id', 'n_frames'], 'feats')
        label_hdf5 = root_path.joinpath(subset, f'{scene_name}.label.hdf5')
        label_csv = root_path.joinpath(subset, f'{scene_name}.label.csv')
        _worker(label_hdf5, label_csv, ['wav_id', 'n_labels'], 'label')


def ark_to_hdf5(conf):
    # read kaldi ark file, write to hdf5
    scene_name  = conf.get('scene_name', 'cn_fan_Midea-XiDun')
    n_split = conf.get('n_split', 8)
    subset_dict = {'tr': 'train', 'cv': 'valid'}
    data_root = conf.get('data_root', 
        '/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro/train/dataset')
    feat_template = 'ark:copy-feats --print-args=false --verbose=-4 scp:{}/feats.scp ark:- |'
    ali_template = 'ark:copy-align scp:{}/ali.asr.ctc.scp ark:- |'
    out_path = Path(conf['data']['data_dir'])
    feats_hdf5_file = f'{scene_name}.feats.hdf5'
    label_hdf5_file = f'{scene_name}.label.hdf5'
    for subset in ['tr', 'cv']:
        feats_hdf5 = h5py.File(out_path.joinpath(subset_dict[subset], feats_hdf5_file), 'w')
        label_hdf5 = h5py.File(out_path.joinpath(subset_dict[subset], label_hdf5_file), 'w')
        for i in range(n_split):
            path_str = data_root + f'/{subset}/split{n_split}/{i}'
            feats_file  = feat_template.format(path_str)            
            logger.info(feats_file)
            for key, value in tqdm(kaldi_io.read_mat_ark(feats_file)):
                feats_hdf5[key] = value
            ali_file = ali_template.format(path_str)
            logger.info(ali_file)
            for key, value in tqdm(kaldi_io.read_vec_int_ark(ali_file)):
                label_hdf5[key] = value


def prepare_subset_hdf5(conf):
    out_format = conf.get('out_format', 'flac')  # flac, wav
    data_info = {
        'train': {'csv_file': '/public/home/wenbin_jiang/workspace/AlphaSE_hdu/data/aispeech/cn_fan_Midea-XiDun_v2_clean.csv',
                  'group_list': ['clean']},
        'valid': {'csv_file': '/public/home/wenbin_jiang/workspace/AlphaSE_hdu/data/aispeech/aispeech_valid.csv',
                  'group_list': ['clean', 'noisy']},
        'test': {'csv_file': '/public/home/wenbin_jiang/workspace/AlphaSE_hdu/data/aispeech/testset/cn_fan_Midea-XiDun.csv',
                 'group_list': ['noisy']},
        'noise': {'csv_file': '/public/home/wenbin_jiang/workspace/AlphaSE_hdu/data/aispeech/aispeech_noise.csv',
                  'group_list': ['noise']}
        }
    root_dir = Path(conf['data']['data_dir'])
    for subset in data_info:
        logger.debug(subset)
        csv_file = Path(data_info[subset]['csv_file'])
        store_file = csv_file.with_suffix('.hdf5')
        wavid_list = get_wav_id_list(csv_file)
        df = pd.read_csv(csv_file, index_col=0, sep=' ', names=['wav_id', 'wav_path', 'duration'])
        out_csv = root_dir.joinpath(subset, csv_file.name)
        if out_csv.exists():
            logger.info(f'{out_csv} exists')
            continue
        df.to_csv(out_csv)
        for group in data_info[subset]['group_list']:
            if len(data_info[subset]['group_list']) > 1:
                out_store_file =root_dir.joinpath(subset, f'{store_file.stem}_{group}.{out_format}.hdf5')
            else:
                out_store_file = root_dir.joinpath(subset, f'{store_file.stem}.{out_format}.hdf5')
            if out_store_file.exists():
                logger.info(f'{out_store_file} exists')
                continue
            logger.debug(f'{out_store_file}, {store_file}, {group}')
            export_hdf5_group(out_store_file, store_file, wavid_list, group, desc=f'{subset}/{group}')


def prepare_aispeech_noise(conf):

    def _get_wavlist(wavlist):
        wav_list = []
        with open(wavlist, 'r') as f:
            for item in f.readlines():
                item = item.strip()
                if not item.startswith('#') and item.endswith('.wav'):
                    item_path = Path(item)
                    if item_path.exists():
                        wav_list.append(item_path)
                    else:
                        print(item_path, 'not exists')
        return wav_list

    wavlist = conf.get('wavlist',
                       '/public/share/kaitao_mu/sharedata/aispeech/corpus/yuntai/data/noise/CustNoiseFromProjs/20240524/readme.wavlist')
    sr = conf.get('sample_rate', 16000)
    out_path = Path(conf['data_dir']).joinpath(conf['name'])
    noise = conf.get('noise', 'aispeech')

    # write csv
    wav_list = _get_wavlist(wavlist)
    print('wav_list:', len(wav_list))
    noise_csv = out_path.joinpath(f'{noise}_noise.csv')
    if not noise_csv.exists():
        print('preparing', noise_csv)
        df = get_wav_dur(wav_list, pool)
        df['id'] = df['id'].apply(get_md5_code)  # file_name to md5_code
        df.to_csv(noise_csv, sep=' ', header=None, index=None)
    else:
        df = pd_read_csv(noise_csv)
    print('total: {:.2f} s, max: {:.2f} s, min: {:.2f} s, mean: {:.2f} s'.format(
        df.dur.sum()/sr, df.dur.max()/sr, df.dur.min()/sr, df.dur.mean()/sr))

    # write hdf5
    h5_file = out_path.joinpath(f'{noise}_noise.hdf5')
    if h5_file.exists():
        print(h5_file, 'exists')
        return
    print('generating ', h5_file)
    future_tasks = []
    for item in wav_list:
        future_tasks.append(pool.submit(
            get_wav, item, 'noise', get_md5_code(item.stem)))
    writer = h5py.File(h5_file, 'w')
    for f in tqdm(future_tasks):
        wav_id, wav_data, flag = f.result()
        writer[f'{flag}/{wav_id}'] = wav_data
    writer.close()


def prepare_aispeech_clean(conf):

    def _get_wav_info(wav_scp):
        if isinstance(wav_scp, str):
            wav_scp = [item.strip() for item in wav_scp.split(',')]
        df_list = []
        for _wav_scp in wav_scp:
            _df = pd.read_csv(_wav_scp, sep='|', names=['id', 'text', 'wav_path'], index_col=0)
            df_list.append(_df)
        df = pd.concat(df_list)
        wav_set, id_set = set(), set()  # unique
        for _, row in tqdm(df.iterrows(), total=len(df)):
            wav_path = Path(row['wav_path'])
            if wav_path.exists():
                wav_set.add(row['wav_path'])
                id_set.add(get_md5_code(wav_path.stem))
        return list(wav_set), list(id_set)

    sr = conf.get('sample_rate', 16000)
    n_concat = conf.get('n_concat', 0)
    root_path = Path(conf['data_dir']).joinpath(conf['name'])
    clean = conf.get('clean', 'cn_fan_Midea-XiDun')
    wav_scp = conf.get('wav_scp', 
        '''/public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/phone2/phone-wkp.scp,
            /public/share/kaitao_mu/sharedata/aispeech/codes/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/phone2/phone-asr.scp''')
    if wav_scp is None:
        with open(root_path.joinpath('wav_scp.list')) as f:
            wav_scp = [x.strip() for x in f.readlines()]

    # write csv
    clean_csv = root_path.joinpath(f'{clean}_clean.csv')
    if not clean_csv.exists():
        print('preparing', clean_csv)
        wav_list, id_list = _get_wav_info(wav_scp)
        print('wav_list:', len(wav_list), len(id_list))
        df = get_wav_dur(wav_list, pool, id_list=id_list)        
        if n_concat > 1:
            group_id_list = sum([[i]*n_concat for i in range(math.ceil(len(df)/n_concat))], [])
            group_id_list = group_id_list[:len(df)]
            df_group = df.sample(frac=1.0) # shuffle
            df_group['group_id'] = group_id_list
            df = df_group.groupby('group_id').agg(
                {'id': lambda x: '|'.join(x), 'wav_path': lambda x: '|'.join(x), 'dur': 'sum'})
        df.to_csv(clean_csv, sep=' ', header=None, index=None)
    else:
        df = pd.read_csv(clean_csv, sep=' ', names=['id', 'wav_path', 'dur'])
    print('total: {:.2f} s, max: {:.2f} s, min: {:.2f} s, mean: {:.2f} s'.format(
        df.dur.sum()/sr, df.dur.max()/sr, df.dur.min()/sr, df.dur.mean()/sr))
    
    # write hdf5
    h5_file = root_path.joinpath(f'{clean}_clean.hdf5')
    if h5_file.exists():
        print(h5_file, 'exists')
        return
    print('generating ', h5_file)
    future_tasks = []
    path_sep = None if n_concat == 0 else '|'
    for index, row in df.iterrows():
        future_tasks.append(pool.submit(
            get_wav, row['wav_path'], 'clean', row['id'], path_sep))
    writer = h5py.File(h5_file, 'w')
    for f in tqdm(future_tasks):
        wav_id, wav_data, flag = f.result()
        writer[f'{flag}/{wav_id}'] = wav_data
    writer.close()


def prepare_aispeech_valid(conf):
    params = conf.mix_param
    n_valid = conf.get('n_valid', 1000)
    dur_min = conf.get('dur_min', 2.1)
    sr = conf.get('sample_rate', 16000)
    root_path = Path(conf['data_dir']).joinpath(conf['name'])
    clean = conf.get('clean', 'aispeech')

    # clean speech
    clean_csv = root_path.joinpath(f'{clean}_clean.csv')
    df_clean = pd_read_csv(clean_csv)
    df_clean = df_clean[df_clean['dur'] > dur_min*sr]  # filter by dur_min
    df_valid = df_clean.sample(n=n_valid)
    df_valid.to_csv(root_path.joinpath(
        f'{clean}_valid.csv'), sep=' ', header=None)
    valid_id_list = df_valid.index.tolist()
    with open(root_path.joinpath('valid_id.txt'), 'w') as f:
        f.write('\n'.join(valid_id_list))

    # noise
    noise_csv = root_path.joinpath(f'{clean}_noise.csv')
    df_noise = pd_read_csv(noise_csv)

    # generate noisy
    writer = h5py.File(root_path.joinpath(f'{clean}_valid.hdf5'), 'w')
    for wav_id in tqdm(valid_id_list):
        x = audio.audioread(df_clean.loc[wav_id]['wav_path'], sr)
        _clean_dur = df_clean.loc[wav_id]['dur']
        _df = df_noise[df_noise['dur'] > _clean_dur]  # filter by clean dur
        _df_noise_sample = _df.sample(n=1)
        n = audio.audioread(_df_noise_sample.iloc[0]['wav_path'], sr)
        begin = random.randint(0, len(n) - len(x))  # random sample
        n_seg = n[begin: begin + len(x)]
        snr = np.random.randint(params['snr_lower'], params['snr_upper'])
        clean_wav, _, noisy_wav, _ = audio.snr_mixer(params, x, n_seg, snr)
        audio.audiowrite(root_path.joinpath(f'{clean}_valid', 'clean', f'{wav_id}.wav'), clean_wav, sr)
        audio.audiowrite(root_path.joinpath(f'{clean}_valid', 'noisy', f'{wav_id}.wav'), noisy_wav, sr)
        writer[f'clean/{wav_id}'] = (clean_wav*INT16_MAX).astype(np.int16)
        writer[f'noisy/{wav_id}'] = (noisy_wav*INT16_MAX).astype(np.int16)
    writer.close()


def prepare_aispeech_testset(conf):

    def _get_wav_info(wavlist):
        if isinstance(wavlist, str):
            wavlist = [item.strip() for item in wavlist.split(',')]
        wav_list, id_list = [], []
        for _wavlist in wavlist:
            f = open(_wavlist)
            content = f.readlines()
            wav_list.extend([item.strip() for item in content])
            id_list.extend([Path(item.strip()).stem for item in content])
        return wav_list, id_list

    sr = conf.get('sample_rate', 16000)
    root_path = Path(conf['data_dir']).joinpath(conf['name'], 'testset')
    testset = conf.get('testset', 'cn_fan_Midea-XiDun')
    wavlist = conf.get('wavlist', 
        '''/public/home/wenbin_jiang/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/wkp/use.wavlist,
            /public/home/wenbin_jiang/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/data/testset/客户测试集/asr/use.wavlist''')

    # write csv
    clean_csv = root_path.joinpath(f'{testset}.csv')
    if not clean_csv.exists():
        print('preparing', clean_csv)
        wav_list, id_list = _get_wav_info(wavlist)
        print('wav_list:', len(wav_list))
        df = get_wav_dur(wav_list, pool, id_list=id_list)
        df.to_csv(clean_csv, sep=' ', header=None, index=None)
    else:
        df = pd.read_csv(clean_csv, sep=' ', names=['id', 'wav_path', 'dur'])
    print('total: {:.2f} s, max: {:.2f} s, min: {:.2f} s, mean: {:.2f} s'.format(
        df.dur.sum()/sr, df.dur.max()/sr, df.dur.min()/sr, df.dur.mean()/sr))

    # write hdf5
    h5_file = root_path.joinpath(f'{testset}.hdf5')
    if h5_file.exists():
        print(h5_file, 'exists')
        return
    print('generating ', h5_file)
    future_tasks = []
    for index, row in df.iterrows():
        future_tasks.append(pool.submit(
            get_wav, row['wav_path'], 'noisy', row['id']))
    writer = h5py.File(h5_file, 'w')
    for f in tqdm(future_tasks):
        wav_id, wav_data, flag = f.result()
        writer[f'{flag}/{wav_id}'] = wav_data
    writer.close()


def prepare_noise_subset(conf):
    sr = conf.get('sample_rate', 16000)
    n_hours = conf.get('n_hours', 15)
    n_samples = int(n_hours * 3600 / 10)  # 10s per sample
    root_path = Path(conf['data_dir']).joinpath(conf['name'])
    aispeech_noise_df = pd_read_csv(root_path.joinpath('aispeech_noise.csv'))
    dns_noise_df = pd_read_csv(root_path.joinpath('DNS2021_noise.csv'))
    dns_noise_sample = dns_noise_df[dns_noise_df['dur'] > 9.95*sr].sample(n=n_samples)
    df = pd.concat([aispeech_noise_df, dns_noise_sample])
    df.to_csv(root_path.joinpath(f'merge_noise_{n_hours}.csv'), sep=' ', header=None)


class PrepareData:
    def __init__(self, conf):
        self.name = conf['name']
        self.sample_rate = conf['data']['sample_rate']
        self.subset = conf.get('subset', None)
        self.out_path = Path(conf['data']['data_dir'])
        if self.subset is not None:
            self.out_path = self.out_path.joinpath(self.subset)
        self.pool = get_pool_executor()
    
    @property
    def sr(self):
        return self.sample_rate
    
    def get_wav_list(self, *args, **kwargs):
        # return: list of wav path
        raise NotImplementedError
    
    def prepare_csv(self, wav_list, csv_file=None):
        # prepare csv inf file
        if csv_file is None:
            csv_file = self.out_path.joinpath(f'{self.name}.csv')
        if not csv_file.exists():
            logger.info(f'preparing {csv_file}')
            df = get_wav_dur(wav_list, self.pool)
            # df['wav_id'] = df['wav_id'].apply(get_md5_code)  # file_name to md5_code
            df.to_csv(csv_file, index=None)
        else:
            df = pd.read_csv(csv_file, index_col=0)
        logger.info('total: {:.2f} s, max: {:.2f} s, min: {:.2f} s, mean: {:.2f} s'.format(
            df['duration'].sum()/self.sr, df['duration'].max()/self.sr, 
            df['duration'].min()/self.sr, df['duration'].mean()/self.sr))
    
    def prepare_hdf5(self, wav_list, h5_file=None, group=None):
        # write hdf5
        if h5_file is None:
            h5_file = self.out_path.joinpath(f'{self.name}.hdf5')
        if h5_file.exists():
            logger.info(f'{h5_file} exists')
            return
        logger.info(f'preparing {h5_file}')
        try:
            _, wav_format, _ = wav_list[0].name.rsplit('.', 2)
        except:
            wav_format = 'wav'
        _, out_format, _ = h5_file.name.rsplit('.', 2)        
        if wav_format != out_format:
            logger.info(f'format: {wav_format} -> {out_format}')
        
        future_tasks = []
        for item in wav_list:
            future_tasks.append(self.pool.submit(get_wav, item, group, item.stem))
        writer = h5py.File(h5_file, 'w')
        for f in tqdm(future_tasks):
            wav_id, wav_data, group = f.result()
            key = f'{group}/{wav_id}' if group else f'{wav_id}'
            if out_format == wav_format:          
                writer[key] = wav_data
            elif wav_format == 'wav' and out_format == 'flac':
                try:
                    with io.BytesIO() as f:
                        sf.write(f, wav_data, self.sr, format='flac')
                        writer[key] = np.void(f.getvalue())
                except Exception as e:
                    logger.error(f'{wav_id}: {e}')
            elif wav_format == 'flac' and out_format == 'wav':
                try:
                    with io.BytesIO() as f:
                        data, _ = sf.read(io.BytesIO(np.void(wav_data)), dtype='int16')                        
                        writer[key] = data
                except Exception as e:
                    logger.error(f'{wav_id}: {e}')
            else:
                logger.error(f'format error: {wav_format} -> {out_format}')
        writer.close()
        
    def append_text_info(self, text_info_file, csv_file, phone_id=True):
        if phone_id is True: # phone_id e.g. 19_3_ ...
            phone_info = get_phone_info(text_info_file, 'data/units/units_phn121.txt')
            label_dict = {}
            for key in phone_info:
                label_dict[key] = '_'.join(phone_info[key].astype(str)) # convert numpy arrary to string and concat with '_'
            text_info = pd.DataFrame.from_dict(label_dict, orient='index', columns=['text'])
            text_info.index.name = 'wav_id'
        else: # phone, e.g. d_a3_k_ai1_f_e1_ng_sh_a4_n|d_a3_k_ai1_y_u3_y_i1_n
            text_info = pd.read_csv(text_info_file, index_col=0, sep=' ', names=['wav_id', 'text', 'concat'])
            text_info.index = text_info.index.to_series().apply(lambda x: x.split('.')[0])
        df = pd.read_csv(csv_file, index_col=0)
        if 'text' not in df:
            df_concat = pd.concat([df, text_info[['text']]], axis=1)
            df_concat.to_csv(csv_file)
    
    def append_label_info(self, label_file, csv_file):
        # append phone_id info, e.g. 116_28_63_24_82_102_19_70_66
        reader = h5py.File(label_file)
        key_list = list(reader.keys())
        label_dict = {}
        for key in tqdm(key_list):
            label_dict[key] = '_'.join(reader[key][()].astype(str)) # convert numpy arrary to string and concat with '_'
        label_info = pd.DataFrame.from_dict(label_dict, orient='index', columns=['text'])
        label_info.index.name = 'wav_id'
        df = pd.read_csv(csv_file, index_col=0)
        assert set(df.index.to_list()) == set(key_list)
        if 'text' not in df:
            df_concat = pd.concat([df,label_info[['text']]], axis=1)
            df_concat.to_csv(csv_file)


def prepare_csv_hdf5(conf):
    # prepare csv and hdf5 from augment wav
    wav_list = []
    with open(conf['wav_scp']) as f:
        for wav_file in f.readlines():
            wav_list.append(Path(wav_file.strip()))
    logger.info(f'wav_list:{len(wav_list)}')
    prepare_data = PrepareData(conf)
    prepare_data.prepare_csv(wav_list)
    prepare_data.prepare_hdf5(wav_list)


def append_text_info(conf):
    prepare_data = PrepareData(conf)
    prepare_data.append_text_info(conf['text_info_file'], conf['csv_file'])


def append_label_info(conf):
    prepare_data = PrepareData(conf)
    prepare_data.append_label_info(conf['label_file'], conf['csv_file'])


def check_csv_info(conf):
    data_path = Path(conf['data']['data_dir']).joinpath(conf.get('subset', 'train'))
    csv_template = 'cn_fan_Midea-XiDun_augment_{}.flac.csv'
    csv_path_template = str(data_path.joinpath(csv_template))
    noisy_df = pd.read_csv(csv_path_template.format('noisy'), index_col=0)
    clean_df = pd.read_csv(csv_path_template.format('clean'), index_col=0)
    clean_rir_df = pd.read_csv(csv_path_template.format('clean_rir'), index_col=0)    
    assert noisy_df['duration'].to_dict() == clean_df['duration'].to_dict()
    assert noisy_df['duration'].to_dict() == clean_rir_df['duration'].to_dict()


def split_list(data, split=0.05):
    '''Randomly split a list into two parts.
    data (List): The list to split.
    split (Union[float, int]): Split point as a ratio (float) or an index (int)
    '''
    n_split = int(len(data) * split) if isinstance(split, float) else split
    random.shuffle(data)
    part1, part2 = data[:n_split], data[n_split:]
    return part1, part2


def path_replace(path, old_part, new_part):
    new_path_str = str(path).replace(old_part, new_part)
    return Path(new_path_str)


def split_train_valid_csv(conf):
    split = conf.get('split', 0.1)
    scene_name  = conf.get('scene_name', 'cn_fan_Midea-XiDun_augment')
    root_path = Path(conf['data']['data_dir'])
    valid_csv = list(root_path.glob(f'valid/{scene_name}*.csv'))
    if len(valid_csv) > 0:
        logger.info(f'[{scene_name}] valid csv exists: {str(valid_csv)}')
        return
    csv_list = list(root_path.glob(f'train/{scene_name}*.csv'))
    if len(csv_list) == 0:
        logger.error(f'[{scene_name}] train csv not exists')
        return
    data_info = []
    for csv_file in csv_list:
        df = pd.read_csv(csv_file, index_col=0)
        data_info.append((csv_file, df))
    wav_id_list = data_info[0][1].index.to_list()
    valid_id_list, train_id_list = split_list(wav_id_list, split)
    for csv_file, df in data_info:
        train_df = df[df.index.isin(train_id_list)]
        valid_df = df[df.index.isin(valid_id_list)]
        valid_csv = path_replace(csv_file, 'train', 'valid')
        logger.info(f'write: {valid_csv}')
        valid_df.to_csv(valid_csv)
        logger.info(f'overwrite: {csv_file}')
        train_df.to_csv(csv_file)

def wav2flac(conf):
    src_path = Path(conf.get('src')).rglob('*.wav')
    dest_path = Path('./data/temp')
    for wav_path in tqdm(src_path):
        wav_data, sr = sf.read(wav_path)
        out_path = dest_path / f'{get_md5_code(wav_path.stem)}.flac'
        sf.write(out_path, wav_data, sr, format='flac')


def df_to_csv(df, dest_file, n_split=1):
    if n_split == 1:
        df.to_csv(dest_file)
    else:
        x = Path(dest_file)
        output_path = x.parent / f'{x.stem}.split{n_split}'
        output_path.mkdir(parents=True, exist_ok=True)
        rows_per_file = (len(df) + n_split - 1) // n_split
        for i in range(n_split):
            start_row = i * rows_per_file
            end_row = min(start_row + rows_per_file, len(df))
            df.iloc[start_row:end_row].to_csv(output_path.joinpath(f'{i}.csv'))


def prepare_BigData_csv(conf):
    subset = conf.get('subset', 'clean')
    src_path = list(Path(conf.get('src')).rglob(f'?/{subset}?.csv'))
    src_path.sort()
    dest_file = Path(conf.get('dest_file', './data/BigData/1kw/train_clean.csv'))
    df_list = []
    for src_file in tqdm(src_path):
        print(src_file)
        _df = read_csv(src_file)
        df_list.append(_df)
    df = pd.concat(df_list)
    df.index = df.index.map(get_md5_code)
    base_dir = conf['data']['train']['base_dir']
    df['wav_path'] = df['wav_path'].str.replace(base_dir, '')
    df_to_csv(df, dest_file, conf.get('split_dataset', 1))


def prepare_common_32w_csv(conf):
    src_path = Path(conf.get('src', '/public/home/haoyi_zhang/AlphaSpeech/examples/aispeech/data/Bigdata/tr/32w/clean_32w.csv'))
    dest_file = Path(conf.get('dest_file', './data/BigData/32w/train_clean.csv'))
    df = read_csv(src_path)
    df.index = df.index.map(get_md5_code)
    base_dir = conf['data']['train']['base_dir']
    df['wav_path'] = df['wav_path'].str.replace(base_dir, '')
    df_to_csv(df, dest_file, conf.get('split_dataset', 1))


def prepare_valid_aispeech_csv(conf):
    src_path = Path(conf.get('src', '/public/home/wenbin_jiang/workspace/AlphaSpeech/examples/aispeech/data/valid/aispeech.csv'))
    df = read_csv(src_path)
    dest_dir = Path(conf.get('dest_dir', './data/valid'))
    for subset in ['clean', 'noisy']:
        df['wav_path'] = df.index.map(lambda x: f'{subset}/{x}.wav')
        dest_file = dest_dir.joinpath(f'aispeech_{subset}.csv')
        df_to_csv(df, dest_file, conf.get('split_dataset', 1))


def prepare_BigData_2kw(conf):
    src_path = Path(
        conf.get('src', '/public/share/kaitao_mu/sharedata/aispeech/corpus/BigData/common-huge-data-init-2000w/tr/wav_clean.scp'))
    temp_csv = Path(conf.get('temp_csv', './exp/debug/2kw_train_clean.csv'))
    dest_file = Path(conf.get('dest_file', './data/BigData/2kw/train_clean.csv'))
    base_dir = conf['data']['train']['base_dir']
    
    # stage 1: convert wav_path to md5_code, and remove the base_dir
    if not temp_csv.exists():
        print('generate', temp_csv)
        df = pd.read_csv(src_path, index_col=0, sep=' ', names=['id', 'wav_path'])
        df.index = df.index.map(get_md5_code)
        df['wav_path'] = df['wav_path'].str.replace(base_dir, '')
        df_to_csv(df, temp_csv)
    df = read_csv(temp_csv)
    
    # stage 2
    n_split = 16
    csv_base = Path(f'./exp/debug/2kw_train_clean.split{n_split}')
    if not csv_base.joinpath(f'0.csv').exists():
        df_to_csv(df, temp_csv, n_split)
    
    # stage 3: get duration of all wav
    if dest_file.exists():
        return
    for idx in range(n_split):
        src_csv = csv_base.joinpath(f'{idx}.csv')
        dest_csv = csv_base.joinpath(f'{idx}_dur.csv')
        if not dest_csv.exists():
            print(src_csv)
            _df = read_csv(src_csv)
            wav_list = _df['wav_path'].to_list()
            wav_list = [base_dir + x for x in wav_list]
            id_list = _df.index.to_list()
            df = get_wav_dur(wav_list, pool, fs=None, id_list=id_list)
            df = df.set_index('wav_id')
            df['wav_path'] = df['wav_path'].str.replace(base_dir, '')
            df_to_csv(df, dest_csv)
    df_list = []
    for idx in range(n_split):
        df_list.append(read_csv(csv_base.joinpath(f'{idx}_dur.csv')))
    df = pd.concat(df_list)
    for n_split in [1, 4, 8, 16]:
        df_to_csv(df, dest_file, n_split)


def prepare_merge_data(conf):
    # prepare merge data of a scence
    scence_dict = {'cn_fengshan_Midea-XiDun': 'cn_fengshan_Midea-XiDun__YJGGZC-27436',
                   'cn_chabaji_DongHengSheng': 'cn_chabaji_DongHengSheng_Sanye__YJGGZC-22241',
                   'cn_fanlight_Demo01': 'cn_fanlight_Demo01',
                   'cn_fengshan_DongHengSheng': 'cn_fengshan_DongHengSheng-P02__YJGGZC-19560'}
    dest_dir = Path(conf.get('dest_dir', './data/BigData/merge_data'))
    debug = conf['system'].get('debug')
    base_dir = conf['data']['train']['base_dir']
    for key, value in scence_dict.items():
        for x in ['noisy', 'clean']:
            src_path = Path(base_dir).joinpath(
                value, f'exp/baseline/train/dataset/merged/tr/{x}/{x}.scp')
            dest_csv = dest_dir.joinpath(f'{key}_{x}.csv')
            if not dest_csv.exists():
                print('generate', dest_csv)
                # stage 1: convert wav_path to md5_code
                df = pd.read_csv(src_path, index_col=0, sep=' ', names=['id', 'wav_path'])
                if debug:
                    df = df[:1000]
                df.index = df.index.map(get_md5_code)
                # stage 2: get duration of all wav
                wav_list = df['wav_path'].to_list()
                id_list = df.index.to_list()
                df = get_wav_dur(wav_list, pool, fs=None, id_list=id_list)
                df = df.set_index('wav_id')
                df['wav_path'] = df['wav_path'].str.replace(base_dir, '')
                df_to_csv(df, dest_csv)


def prepare_phone_data(conf):
    wavlist = Path(conf.get('wavlist', 'data/BigData/phone_data/phone_data_uniq.wavlist.path_fixed'))
    csv_file = Path(conf.get('csv_file', 'data/BigData/phone_data/train_clean.csv'))
    h5_file = Path(conf.get('h5_file', 'data/BigData/phone_data/train_clean.hdf5'))
    if not csv_file.exists():
        df = pd.read_csv(wavlist, sep='|', names=['wav_id', 'phone', 'wav_path'], index_col=0)
        df = df.drop_duplicates(subset='wav_path')
        df.index = df['wav_path'].map(get_md5_code)
        wav_list = df['wav_path'].to_list()
        id_list = df.index.to_list()
        df = get_wav_dur(wav_list, pool, fs=conf['data']['sample_rate'], id_list=id_list)
        df = df.set_index('wav_id')
        df = df[df['duration'] > 8000]
        df_to_csv(df, csv_file)
        
    # check file again
    df = read_csv(csv_file)
    file_paths = df['wav_path'].tolist()
    exists = check_files(file_paths)
    df_filtered = df[exists]
    df_filtered.to_csv(csv_file)
    
    # write hdf5
    obj = WavHDF5()
    obj.wav_to_hdf5(csv_file, h5_file, n_split=conf.get('n_split', 1))

    
def concat_split_csv(conf):
    # concatenate and split wav.csv
    csv_file = Path(conf.get('csv_file', 'data/BigData/phone_data/temp_hdf5.csv'))
    n_concat = conf.get('n_concat', 2) # concatenate the wav
    n_split = conf.get('n_split', 1) # split the dataset
    out_csv = csv_file.with_stem((f'{csv_file.stem}_c{n_concat}'))
    df = read_csv(csv_file)
    df_concat = group_and_concat(df, n_concat)
    df_to_csv(df_concat, out_csv, n_split)


def concat_split_csv_clean_noisy(conf):
    clean_csv = Path(conf.get('clean_csv', 'data/BigData/1kw/train_clean.csv'))
    noisy_csv = Path(conf.get('noisy_csv', 'data/BigData/1kw/train_noisy.csv'))
    n_concat = conf.get('n_concat', 2) # concatenate the wav
    n_split = conf.get('n_split', 1) # split the dataset
    clean_out_csv = clean_csv.with_stem((f'{clean_csv.stem}_c{n_concat}'))
    noisy_out_csv = noisy_csv.with_stem((f'{noisy_csv.stem}_c{n_concat}'))
    df_clean = read_csv(clean_csv)
    df_noisy = read_csv(noisy_csv)
    df_clean_concat, df_noisy_concat = group_and_concat_two(df_clean, df_noisy, n_concat)
    df_to_csv(df_clean_concat, clean_out_csv, n_split)
    df_to_csv(df_noisy_concat, noisy_out_csv, n_split)


def test_phone_data(conf):
    csv_file = Path(conf.get('csv_file', 'data/BigData/phone_data/temp_hdf5.csv'))
    obj = WavHDF5()
    df = read_csv(csv_file)
    wav_list = df['wav_path'].to_list()
    for wav_path in tqdm(wav_list):
        data = obj.get_data(wav_path)
        # print(data.shape)
    

def convert_hdf5(conf):
    # convert the old version (src_file) to the new version (hdf5_file)
    src_file = conf.get('src_file', '/public/home/wenbin_jiang/data/workspace/data/DNS2021/merge_noise.hdf5')
    csv_file = conf.get('csv_file', '/public/home/wenbin_jiang/data/workspace/data/DNS2021/merge_noise.csv')
    dest_file = conf.get('dest_file', 'data/noise/merge_noise.hdf5')
    df = pd_read_csv(csv_file)
    df = df.rename(columns={'dur': 'duration'})
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    temp_file_path = temp_file.name
    print(f"temp csv: {temp_file_path}")
    df.to_csv(temp_file_path)
    temp_file.close()
     # write hdf5
    obj = WavHDF5()
    obj.convert_hdf5(temp_file_path, dest_file, src_file, 
                     n_split=conf.get('n_split', 1), group=conf.get('group', 'noise'))


def sample_DNS_noise(conf):
    src_file = Path(conf.get('src_file', 'data/noise/merge_noise_hdf5.csv'))
    hours = conf.get('hours', 5)
    df = read_csv(src_file)
    n_aispeech = 77
    df_DNS_noise = df[:-n_aispeech]
    sampled_df = sample_audio(df_DNS_noise, hours)
    df_new = pd.concat([sampled_df, df.tail(n_aispeech)])
    df_new.to_csv(src_file.with_stem(f'{src_file.stem}_{hours}h'))


if __name__ == '__main__':
    conf = OmegaConf.create({
        'conf': 'conf/asr.yaml',
        'cmd': 'append_label_info',
        'num_workers': 8
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    eval(conf.cmd)(conf)
