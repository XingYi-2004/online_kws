import io
import numpy as np
import h5py
from tqdm import tqdm
import librosa
import soundfile as sf
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from modules.utils.logging import logger
from tools.prepare_data import get_md5_code, get_wav_dur
from modules.utils.common import get_pool_executor

pool = get_pool_executor()


def prepare_hdf5(conf):
    csv_root = Path(conf.get('csv_root', '~/shared/workspace/ConferencingSpeech2022/Training/Dev datasets/')).expanduser()
    data_root = Path(conf.get('csv_root', '~/shared/data/ConferencingSpeech2022/')).expanduser()    
    fs = conf.get('fs', 16000)
    wav_df_list = []

    # NISQA_Corpus
    wav_csv = 'data/train/NISQA.csv'
    if not Path(wav_csv).exists():
        df = pd.read_csv(csv_root.joinpath('NISQACorpus/NISQA_corpus_mos.csv'))
        df['wav_path'] = df['filepath_deg'].apply(lambda x: str(data_root.joinpath('NISQA_Corpus', x)))
        df['wav_path_16k'] = df['wav_path'].apply(lambda x: x.replace("NISQA_Corpus", "NISQA_Corpus_16k"))        
        for _, row in tqdm(df.iterrows()):
            data, _ = librosa.load(row['wav_path'], sr=fs)
            Path(row['wav_path_16k']).parent.mkdir(parents=True, exist_ok=True)
            sf.write(row['wav_path_16k'], data, sr=fs)
        df['wav_id'] = df['wav_path'].apply(get_md5_code)
        df = df.set_index('wav_id')
        wav_df = get_wav_dur(df['wav_path_16k'].to_list(), pool, id_list=df.index.to_list(), fs=fs)
        wav_df = wav_df.set_index('wav_id')
        wav_df['mos'] = df['mos']
        wav_df.to_csv(wav_csv)
    else:
        wav_df = pd.read_csv(wav_csv, index_col=0)
        wav_df_list.append(wav_df)

    # TencentCorpus
    wav_csv = 'data/train/Tencent.csv'
    if not Path(wav_csv).exists():
        df = pd.concat([pd.read_csv(csv_root.joinpath('TencentCorpus/withoutReverberationTrainDevMOS.csv')),
                        pd.read_csv(csv_root.joinpath('TencentCorpus/withReverberationTrainDevMOS.csv'))], 
                       axis=0)
        df['wav_path'] = df['deg_wav'].apply(lambda x: str(data_root.joinpath('TencentCorups', x)))
        df['wav_id'] = df['wav_path'].apply(get_md5_code)
        df = df.set_index('wav_id')        
        wav_df = get_wav_dur(df['wav_path'].to_list(), pool, id_list=df.index.to_list(), fs=fs)
        wav_df = wav_df.set_index('wav_id')
        wav_df['mos'] = df['mos']
        wav_df.to_csv(wav_csv)
    else:
        wav_df = pd.read_csv(wav_csv, index_col=0)
        wav_df_list.append(wav_df)

    # IU Bloomington Corpus
    wav_csv = 'data/train/Bloomington.csv'
    if not Path(wav_csv).exists():
        df = pd.concat([pd.read_csv(data_root.joinpath('MOS_Quality_INTERSPEECH_2020/audio_scaled_mos_cosine.csv')),
                        pd.read_csv(data_root.joinpath('MOS_Quality_INTERSPEECH_2020/audio_scaled_mos_voices.csv'))], 
                       axis=0)
        df['wav_path'] = df['AudioName'].apply(
            lambda x: str(data_root.joinpath('MOS_Quality_INTERSPEECH_2020/16k_speech', x.split('/')[-1])))
        df['wav_id'] = df['wav_path'].apply(get_md5_code)
        df = df.set_index('wav_id')
        wav_df = get_wav_dur(df['wav_path'].to_list(), pool, id_list=df.index.to_list(), fs=fs)
        wav_df = wav_df.set_index('wav_id')
        wav_df['mos'] = df['ScaledMOS']/2.0
        wav_df.to_csv(wav_csv)
    else:
        wav_df = pd.read_csv(wav_csv, index_col=0)
        wav_df_list.append(wav_df)

    # all data
    wav_csv = 'data/train/noisy.csv'
    if not Path(wav_csv).exists():
        wav_df_all = pd.concat(wav_df_list, axis=0)
        wav_df_all.to_csv('data/train/noisy.csv')
    else:
        wav_df_all = pd.read_csv(wav_csv, index_col=0)

    # data store
    hdf5_file = Path(f"data/train/noisy.flac.hdf5")
    if hdf5_file.exists():
        logger.info(f"{hdf5_file} exists")
    else:
        writer = h5py.File(hdf5_file, "w")
        for _, row in tqdm(wav_df_all.iterrows()):
            wav_data, _ = librosa.load(row["wav_path"], sr=fs)
            with io.BytesIO() as f:
                sf.write(f, wav_data, fs, format="flac")
                writer[row["wav_id"]] = np.void(f.getvalue())


if __name__ == '__main__':
    conf = OmegaConf.create({
        'yaml': 'conf/config.yaml',
        'cmd': 'prepare_hdf5',
        'num_workers': 8
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
