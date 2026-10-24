# pytorch map-style dataset
from torch.utils.data import Dataset
import torch
import random
import numpy as np
from omegaconf import OmegaConf
from modules.utils.logging import logger
from modules.utils import audio
from modules.utils.common import get_rank
from .common import read_csv, get_chunk_info, check_wav_value
from .data_store import WavStore, FeatStore
from .dataset_base import DatasetBase

try:
    from .spike_dataset import GSCDataset, SHDDataset, SSCDataset, SpikeDataset
except Exception as exc:
    class _MissingSpikingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Spiking datasets require spikingjelly and torchaudio. '
                'Please install them before using SHD/SSC/GSC datasets.'
            ) from exc

    SHDDataset = _MissingSpikingDependency
    SSCDataset = _MissingSpikingDependency
    GSCDataset = _MissingSpikingDependency
    SpikeDataset = _MissingSpikingDependency


class DatasetMap(Dataset, DatasetBase):
    def __init__(self, conf, subset):
        Dataset.__init__(self)
        DatasetBase.__init__(self, conf, subset)
        
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError


class WavDataset(DatasetMap):
    '''map-style dataset'''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noisy']
        self.normalize = conf.get('normalize') # for the diffusion based model
        
    def __getitem__(self, index):
        wav_id, begin, end = self.get_chunk(index)
        noisy_wav = self.get_wav_data(wav_id, 'noisy')
        noisy_wav = self.check_wav_data(noisy_wav[begin:end])
        if self.normalize:
            normfac = noisy_wav.abs().max()
            noisy_wav = noisy_wav / normfac
        data = {'noisy_wav': noisy_wav, 'wav_id': wav_id}
        if self.unpaired is True:
            wav_id = self.data_store['clean'].sample_id(deny_list=[wav_id])
        if self.chunk_info['clean'] is not None:
            clean_wav = self.get_wav_data(wav_id, 'clean')
            clean_wav = self.check_wav_data(clean_wav[begin:end])
            if self.normalize:
                clean_wav = clean_wav / normfac
            data['clean_wav'] = clean_wav
        if self.label_dict is not None and len(self.label_dict) > 0:
            data['label'] = self.label_dict[wav_id]
        if self.check_value:
            if check_wav_value(clean_wav) is False or check_wav_value(noisy_wav) is False:
                new_index = np.random.randint(0, len(self)-1) # sampling new index
                logger.info(f'check value failed, {index} -> {new_index}')
                data = self[new_index]
        return data

    def __len__(self):
        return len(self.chunk_info['clean'] if self.chunk_info['clean'] else self.chunk_info['noisy'])


class WavDataSetTeacher(DatasetMap):
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        teacher_csv = self.resolve_path('teacher_csv', subset)
        teacher_store = self.resolve_path('teacher_store', subset)
        self.data_info['teacher'] = read_csv(teacher_csv) if teacher_csv else None
        self.data_store['teacher'] = WavStore(
            teacher_store, self.data_info['teacher'], name=self.conf['name'], 
            cache=self.cache) if teacher_store or teacher_csv else None
        self.chunk_info['teacher'] = get_chunk_info(self.data_info['teacher'], self.chunk_samples)
        
    def __getitem__(self, index):
        wav_id, begin, end = self.get_chunk(index)
        noisy_wav = self.get_wav_data(wav_id, 'noisy')
        noisy_wav = self.check_wav_data(noisy_wav[begin:end])
        data = {'noisy_wav': noisy_wav, 'wav_id': wav_id}
        if self.unpaired is True:
            wav_id = self.data_store['clean'].sample_id(deny_list=[wav_id])
        if self.chunk_info['clean'] is not None:
            clean_wav = self.get_wav_data(wav_id, 'clean')
            clean_wav = self.check_wav_data(clean_wav[begin:end])
            data['clean_wav'] = clean_wav
        if self.chunk_info['teacher'] is not None:
            teacher_wav = self.get_wav_data(wav_id, 'teacher')
            teacher_wav = self.check_wav_data(teacher_wav[begin:end])
            data['teacher_wav'] = teacher_wav
        if self.label_dict is not None and len(self.label_dict) > 0:
            data['label'] = self.label_dict[wav_id]
        return data

    def __len__(self):
        return len(self.chunk_info['clean'] if self.chunk_info['clean'] else self.chunk_info['noisy'])


class NoiseDataset(DatasetMap):
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noise']

    def get_noise(self, index=None):
        if not index:
            index = random.randint(0, len(self))
        wav_id, begin, end = self.get_chunk(index, 'noise')
        noise = self.get_wav_data(wav_id, 'noise')
        return self.check_wav_data(noise[begin:end])

    def __getitem__(self, index):
        noise = self.get_noise()
        return noise
    
    def __len__(self):
        # should larger than batch size
        return len(self.chunk_info['noise'])


class OnlineNoisyDataset(DatasetMap):
    '''
    noisy2clean: clean + noise -> clean
    noisy2noisy: clean + noise1 -> clean + noise2
    noisier2noisy: clean + noise1 + noise2 -> clean + noise1
    noisier2noisy_v2: noisy + noise -> noisy
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        if self.pairs == 'noisier2noisy_v2':
            assert (self.data_store['noisy'] and self.data_store['noise'])
        else:
            assert (self.data_store['clean'] and self.data_store['noise'])
        if self.add_rir > 0:
            assert self.data_store['rir']
        logger.info(f"pairs: {self.pairs}, [{self.mix_param.snr_lower}, {self.mix_param.snr_upper}] dB")
        self.suppression_ratio = self.mix_param.get('suppression_ratio')
        if self.suppression_ratio:
            logger.info(f'suppression_ratio: {self.suppression_ratio} dB')
        
    def get_clean(self, index):
        key = 'noisy' if self.pairs == 'noisier2noisy_v2' else 'clean'
        wav_id, begin, end = self.get_chunk(index, key)
        clean = self.get_wav_data(wav_id, key)
        clean = clean[begin:end]
        clean = self.check_wav_data(clean)
        if random.random() < self.add_rir:
            clean = self.add_pyreverb(clean)
        if clean.sum() == 0:
            new_index = np.random.randint(0, index)
            logger.error(f'clean_wav is 0, {index} -> {new_index}')
            return self.get_clean(new_index)
        return wav_id, clean

    def get_noise(self, index=None):
        if not index:
            index = random.randint(0, len(self))
        wav_id, begin, end = self.get_chunk(index, 'noise')
        noise = self.get_wav_data(wav_id, 'noise')
        return self.check_wav_data(noise[begin:end])

    def __getitem__(self, index):
        wav_id, clean = self.get_clean(index)
        noise = self.get_noise()
        snr = self.snr_n1
        clean, _, noisy, _ = audio.snr_mixer(self.mix_param, clean, noise, snr)
        if self.pairs in ['noisy2clean', 'noisier2noisy_v2']:
            if self.suppression_ratio:
                target_snr = snr + self.suppression_ratio
                _, _, target, target_rms_level = audio.snr_mixer(self.mix_param, clean, noise, target_snr)
                noisy = audio.normalize(noisy, target_rms_level)    
            else:
                target = clean
        elif self.pairs == 'noisy2noisy':
            noise = self.get_noise()
            _, _, target, target_rms_level = audio.snr_mixer(self.mix_param, clean, noise, self.snr)
            noisy = audio.normalize(noisy, target_rms_level)
        elif self.pairs == 'noisier2noisy':
            noise = self.get_noise()
            _, _, noisier, noisier_rms_level = audio.snr_mixer(self.mix_param, noisy, noise, self.snr)
            target = noisy
            noisy = noisier
            target = audio.normalize(target, noisier_rms_level)
        else:
            raise ValueError(f'pairs: {self.pairs}')
        data = {'clean_wav': target.astype(np.float32), 
                'noisy_wav': noisy.astype(np.float32), 
                'wav_id': wav_id}
        return data
        # return target.astype(np.float32), noisy.astype(np.float32), wav_id

    def __len__(self):
        if self.pairs == 'noisier2noisy_v2':
            length = len(self.chunk_info['noisy'])
        else:
            length = len(self.chunk_info['clean'])
        return length


class FeatLableDataset(DatasetMap):
    '''FeatLableDataset
    read feat and label from kaldi data (hdf5)
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        feats_csv = self.resolve_path('feats_csv', subset)
        feats_store = self.resolve_path('feats_store', subset)
        self.data_info['feats'] = read_csv(feats_csv) if feats_csv else None
        self.data_store['feats'] = FeatStore(feats_store, self.data_info['feats'], 
                                             cache=self.cache) if feats_store or feats_csv else None
        assert self.label_dict is not None
        assert set(self.data_info['feats'].index) == set(self.label_dict.keys())
        assert self.feat_extractor is not None
        logger.info(self.get_feats_msg())
        self.filter_feats()
        
    def get_feats_msg(self):
        n_frames = self.data_info['feats']['n_frames']
        msg = '[{}] utterance: {:d}, max: {:d} frames, min: {:.2f} frames, mean: {:.2f} frames'.format(
            self.subset, len(n_frames), n_frames.max(), n_frames.min(), n_frames.mean())
        return msg
    
    def filter_feats(self, key='feats', name='n_frames'):
        # Filter feats, and update data_info and data_store
        filter_config = OmegaConf.to_container(self.conf['data'][self.subset]).get('filter')
        if not filter_config:
            return
        df = self.data_info[key]
        assert name in df.columns
        if isinstance(filter_config, list):
            lower_bound, upper_bound = filter_config
        elif isinstance(filter_config, float):
            mid_quantile = filter_config
            lower_q = (1 - mid_quantile) / 2
            upper_q = mid_quantile + lower_q
            lower_bound = df[name].quantile(lower_q)
            upper_bound = df[name].quantile(upper_q)
        else:
            raise ValueError(f'Invalid filter configuration: {filter_config}')
        df_filtered = df[(df[name] >= lower_bound) & (df[name] <= upper_bound)]
        self.data_info[key] = df[df.index.isin(df_filtered.index)]
        self.data_store[key].update_info(self.data_info[key])
        logger.info(f'[Filtered] {self.get_feats_msg()}')
    
    def get_feats(self, index):
        wav_id, data = self.data_store['feats'].get_data_by_index(index)
        feats = self.feat_extractor.post_process(data)
        return wav_id, feats
    
    def __getitem__(self, index):
        wav_id, feats = self.get_feats(index)
        label = self.label_dict[wav_id]
        data = {'wav_id': wav_id, 'feats': feats, 'label': label}
        return data
    
    def __len__(self):
        return len(self.data_info['feats'])


class WavTextDataset(DatasetMap):
    '''WavTextDataset
    Extract FBank from wav, and obtain CTC lable from text
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.label_dict is not None
        assert self.feat_extractor is not None
    
    def __getitem__(self, index):        
        wav_id, begin, end = self.get_chunk(index)
        noisy_wav = self.get_wav_data(wav_id, 'noisy')
        noisy_wav = self.check_wav_data(noisy_wav[begin:end])        
        feats = self.feat_extractor.get_feat(torch.from_numpy(noisy_wav))
        label = self.label_dict[wav_id]
        data = {'wav_id': wav_id, 'noisy_wav': noisy_wav, 'feats': feats, 'label': label}
        return data        
    
    def __len__(self):
        return len(self.data_info['noisy'])


class ToyDataset(DatasetMap):
    # for test
    def __init__(self, conf, subset):
        data = list(range(10))
        if conf['data'][subset].get('split_dataset', 1) > 1:
            rank = get_rank()
            self.data = [f'rank{rank}_{x}' for x in data]
        else:
            self.data = data
            
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
