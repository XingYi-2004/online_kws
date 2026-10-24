import shutil
import h5py
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from modules.utils.logging import logger
from tools.prepare_data import export_hdf5_group, get_wav_id_list, pd_read_csv
from modules.dataset.wav_hdf5 import WavHDF5
from modules.dataset.common import read_csv


def prepare_subset_hdf5(conf):
    store_file = conf.get('store_file', '~/workspace/AlphaSE/data/voicebank/wav.hdf5')
    out_format = conf.get('out_format', 'flac')  # flac, wav
    wavid_dict = {
        'train': get_wav_id_list('data/train/train.csv'),
        'valid': get_wav_id_list('data/valid/valid.csv'),
        'test': get_wav_id_list('data/test/test.csv')
    }
    for subset in wavid_dict:
        wavid_list = wavid_dict[subset]
        for group in {'clean', 'noisy'}:
            hdf5_file = Path(f'data/{subset}/{group}.{out_format}.hdf5')
            if hdf5_file.exists():
                logger.info(f'{hdf5_file} exists')
                continue
            export_hdf5_group(hdf5_file, store_file, wavid_list, group, desc=f'{subset}/{group}')

    # noise
    noise_list = get_wav_id_list('data/noise/noise.csv')
    hdf5_file = Path(f'data/noise/noise.{out_format}.hdf5')
    if hdf5_file.exists():
        logger.info(f'{hdf5_file} exists')
    else:
        export_hdf5_group(hdf5_file, store_file, noise_list, 'noise', desc='noise')

    # rir
    rir_store_file = conf.get('rir_store_file', '~/workspace/AlphaSE/data/DNS2021/datasets/rir.hdf5')
    rir_list = get_wav_id_list('data/rir/rir.csv')
    hdf5_file = Path(f'data/rir/rir.{out_format}.hdf5')
    if hdf5_file.exists():
        logger.info(f'{hdf5_file} exists')
    else:
        export_hdf5_group(hdf5_file, rir_store_file, rir_list, 'rir', desc='rir')
        
        
def prepare_hdf5(conf):
    store_file = conf.get('store_file', '~/workspace/AlphaSE/data/voicebank/wav.hdf5')
    out_format = conf.get('out_format', 'flac')  # flac, wav    
    wav_store = h5py.File(Path(store_file).expanduser(), 'r')
    
    # clean and noisy
    wavid_list = list(wav_store['noisy'].keys())
    for group in {'clean', 'noisy'}:
        hdf5_file = Path(f'data/train/{group}.{out_format}.hdf5')
        if hdf5_file.exists():
            logger.info(f'{hdf5_file} exists')
            continue
        export_hdf5_group(hdf5_file, store_file, wavid_list, group, desc=group)
    
    # noise
    noise_list = list(wav_store['noise'].keys())
    hdf5_file = Path(f'data/noise/noise.{out_format}.hdf5')
    if hdf5_file.exists():
        logger.info(f'{hdf5_file} exists')
    else:
        export_hdf5_group(hdf5_file, store_file, noise_list, 'noise', desc='noise')

    # rir
    rir_store_file = conf.get('rir_store_file', '~/workspace/AlphaSE/data/DNS2021/datasets/rir.hdf5')
    rir_store = h5py.File(Path(rir_store_file).expanduser(), 'r')
    rir_list = list(rir_store['rir'].keys())
    hdf5_file = Path(f'data/rir/rir.{out_format}.hdf5')
    if hdf5_file.exists():
        logger.info(f'{hdf5_file} exists')
    else:
        export_hdf5_group(hdf5_file, rir_store_file, rir_list, 'rir', desc='rir')
    

def export_group(conf):
    store_file = conf.get('store_file', '~/shared/workspace/AlphaSE/data/voicebank/wav_spp.hdf5')
    group = conf.get('group', 'omlsa')
    subset = conf.get('subset', 'train')
    out_format = conf.get('out_format', 'flac')  # flac, wav    
    wav_store = h5py.File(Path(store_file).expanduser(), 'r')
    wavid_list = list(wav_store[group].keys())    
    hdf5_file = Path(f'data/{subset}/{group}.{out_format}.hdf5')
    if hdf5_file.exists():
        logger.info(f'{hdf5_file} exists')
    else:
        export_hdf5_group(hdf5_file, store_file, wavid_list, group, desc=group)
    

def prepare_csv(conf):
    csv_dir = conf.get('csv_dir', 'data')
    for csv in Path(csv_dir).rglob('*.csv'):
        if csv.name in ['RIR_table_simple.csv']:
            continue
        dest = csv.with_suffix('.csv.bak')
        shutil.copyfile(csv, dest)
        df = pd.read_csv(csv, index_col=0, sep=' ', names=['wav_id', 'wav_path', 'duration'])
        df_filter = df[df['duration'] > 0]
        logger.info(f'{csv}: {len(df)} -> {len(df_filter)}')
        df_filter.to_csv(csv)


def convert_hdf5(conf):
    # convert the old version (src_file) to the new version (hdf5_file)
    src_file = conf.get('src_file', 'data/noise/noise.wav.hdf5')
    csv_file = conf.get('csv_file', 'data/noise/noise.csv')
    dest_file = conf.get('dest_file', 'data/noise/DEMAND.wav.hdf5')
     # write hdf5
    obj = WavHDF5()
    obj.convert_hdf5(csv_file, dest_file, src_file)


if __name__ == '__main__':
    conf = OmegaConf.create({
        'yaml': 'conf/config.yaml',
        'cmd': 'export_group',
        'num_workers': 8
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
