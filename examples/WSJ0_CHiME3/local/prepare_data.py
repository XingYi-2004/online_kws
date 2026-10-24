from tqdm import tqdm
import h5py
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from modules.utils.logging import logger


def merge_hdf5(conf):
    # 1. check wav_id
    data_path = Path('data')
    csv_list = data_path.rglob('*/clean_wav_*.csv')
    df_list = []
    num_item = 0
    for csv in csv_list:
        df = pd.read_csv(csv, index_col=0)
        num_item += len(df)
        df_list.append(df)
    df_all = pd.concat(df_list)
    print('Num of item equal:', num_item == len(df_all))
    print('Index duplicated: ', df_all.index.duplicated().any())
    
    # 2. merge_hdf5
    for x in ['clean', 'noisy']:
        hdf5_list = data_path.rglob(f'*/{x}.wav.hdf5')
        out_store_file = data_path.joinpath(f'{x}.wav.hdf5')
        out_store = h5py.File(out_store_file, 'w')
        print('out_store:', out_store_file)
        for hdf_file in hdf5_list:
            in_store = h5py.File(hdf_file, 'r')
            key_list = in_store.keys()
            for wav_id in tqdm(key_list, desc=str(hdf_file)):
                wav_data = in_store[wav_id][()]
                out_store[wav_id] = wav_data
            in_store.close()
        out_store.close()
                                

if __name__ == '__main__':
    conf = OmegaConf.create({
        'yaml': 'conf/config.yaml',
        'cmd': 'merge_hdf5',
        'num_workers': 8
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
