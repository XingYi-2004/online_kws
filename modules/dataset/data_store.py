import h5py
import io
import soundfile as sf
import random
import numpy as np
import pandas as pd
import math
import kaldiio
from pathlib import Path
from . import kaldi_io_cn as kaldi_io
from modules.utils.common import INT16_MAX, read_wav_scp
from .common import random_choice, feature_splice, apply_skip_frame, create_cache
from modules.utils.logging import logger
from collections import OrderedDict
from .wav_hdf5 import WavHDF5


class DataStore:
    def __init__(self, store_file=None, data_info=None, **kwargs):
        '''
        Gets the store type from store_file
        store_type: hdf5, kaldi, lmdb, wavlist
        sub_type: wav, flac
        '''
        self.data_info = data_info
        cache = kwargs.get('cache')
        self.buffer_size = kwargs.get('buffer_size', 16)
        self.data_buffer = OrderedDict() if self.buffer_size > 0 else None
        name = kwargs.get('name')
        if isinstance(self.data_info, pd.DataFrame):
            self.data_info = self.data_info.to_dict(orient='index')
        if store_file is not None:
            if isinstance(store_file, list):
                self.data_store = [self.init_data_store(item, cache, name) for item in store_file]
            else:
                self.data_store = self.init_data_store(store_file, cache, name)
        else:
            logger.warning('store_file is None')
            self.store_type = 'wavlist'
            self.sub_type = None
            assert self.data_info is not None
            self.data_store = self.data_info
        if self.data_info:
            self.id_list = list(self.data_info.keys())
        else:
            logger.warning('data_info is None')
            self.id_list = list(self.data_store.keys()) # bugs ! when data_store is a list
        self.index = 0
    
    def init_data_store(self, store_file, cache, name):
        _, self.sub_type, self.store_type = store_file.name.rsplit('.', 2)
        logger.debug(f'store_file: {store_file}, store_type: {self.store_type}, sub_type: {self.sub_type}')
        if cache:
            store_file = create_cache(store_file, dest=cache, name=name)
        if self.store_type == 'hdf5':
            data_store = h5py.File(store_file, 'r')
        elif self.store_type == 'ark': # kaldi ark
            data_store = open(store_file, 'rb') # binary mode !
        elif self.store_type == 'lmdb':
            raise NotImplementedError
        else:
            logger.error(f'store_type: {self.store_type} error')
            raise NotImplementedError
        return data_store
            
    def get_data_store(self, wav_id):
        # self.data_store may be a list of data_store
        store_idx = self.data_info[wav_id].get('store_idx') if self.data_info else None
        if store_idx is not None and isinstance(self.data_store, list):
            return self.data_store[store_idx]
        else:
            return self.data_store
    
    def update_info(self, data_info):
        self.data_info = data_info
        if isinstance(self.data_info, pd.DataFrame):
            self.data_info = self.data_info.to_dict(orient='index')
        self.id_list = list(self.data_info.keys())

    def get_data(self, *args ,**kwargs):
        raise NotImplementedError
    
    def sample_data(self, n=1):
        # sample n data from id_list
        sampled_id_list = random.sample(self.id_list, n)
        data_list = []
        for wav_id in sampled_id_list:
            data_list.append(self[wav_id])
        return data_list[0] if n == 1 else data_list

    def sample_id(self, n=1, deny_list=None):
        # sample n id from id_list except deny_list
        if deny_list and not isinstance(deny_list, list):
            deny_list = list(deny_list)
        return random_choice(self.id_list, deny_list, n)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.id_list):
            raise StopIteration
        data = self[self.id_list[self.index]]
        self.index += 1
        return data
    
    def get_data_by_index(self, index):
        wav_id = self.id_list[index]
        return wav_id, self[wav_id]

    def __getitem__(self, wav_id):
        if self.data_buffer is None:
            return self.get_data(wav_id)
        if wav_id in self.data_buffer:
            data = self.data_buffer[wav_id]
            self.data_buffer.move_to_end(wav_id) # mark as recently used
        else:
            data = self.get_data(wav_id) # read from the data_store
            self.data_buffer[wav_id] = data # add to buffer
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer.popitem(last=False) # remove the oldest item
        return data

    def __len__(self):
        return len(self.id_list)


class WavStore(DataStore):
    def __init__(self, store_file=None, data_info=None, **kwargs):
        super().__init__(store_file, data_info, **kwargs)
        self.base_dir = kwargs.get('base_dir')
        if self.store_type == 'wavlist':
            wav_id = self.id_list[0]
            data_store = self.get_data_store(wav_id)
            wav_path = data_store[wav_id]['wav_path']
            if wav_path.endswith(('.wav', '.flac', '.mp3')):
                self.sub_type = 'audio'
            elif 'ark:' in wav_path:
                self.sub_type = 'ark'
            elif 'hdf5:' in wav_path:
                self.sub_type = 'hdf5'
                self.wavHDF5 = WavHDF5(base_dir=self.base_dir)
            else:
                logger.error(f'wav_path: {wav_path} type error')
                raise NotImplementedError
            logger.info('sub_type: {}'.format(self.sub_type))
        
    def load_mat(self, wav_path):
        if '|' in wav_path:
            return np.concatenate([self.load_mat(x) for x in wav_path.split('|')])
        if self.base_dir:
            wav_path = self.base_dir + wav_path
        _, wave = kaldiio.load_mat(wav_path)
        return wave
    
    def read_wav(self, wav_path):
        if '|' in wav_path:
            return np.concatenate([self.load_mat(x) for x in wav_path.split('|')])
        if self.base_dir:
            wav_path = self.base_dir + wav_path
        wave, _ = sf.read(wav_path, dtype='float32')
        return wave
    
    def get_data(self, wav_id):
        data_store = self.get_data_store(wav_id)
        if self.store_type == 'wavlist':
            wav_path = data_store[wav_id]['wav_path']
            if self.sub_type == 'audio':
                wave = self.read_wav(wav_path)
            elif self.sub_type == 'ark':
                wave = self.load_mat(wav_path)
            elif self.sub_type == 'hdf5':
                wave = self.wavHDF5.get_data(wav_path)
            else:
                logger.error(f'sub_type: {self.sub_type} error')
                raise NotImplementedError
        else: # ['hdf5', 'kaldi', 'lmdb']
            if self.sub_type == 'flac':
                bin_data = data_store[wav_id][()]  # binary data
                wave, _ = sf.read(io.BytesIO(np.void(bin_data)), dtype='float32')
            elif self.sub_type == 'wav':
                wave = data_store[wav_id][:]
            else:
                logger.error(f'sub_type: {self.sub_type} error')
                raise NotImplementedError
        if wave.dtype == np.int16:
            wave = (wave/INT16_MAX).astype(np.float32)
        return wave


class FeatStore(DataStore):
    def __init__(self, store_file=None, data_info=None, **kwargs):
        super().__init__(store_file, data_info, **kwargs)
        
    def get_data(self, wav_id):
        data_store = self.get_data_store(wav_id)
        return data_store[wav_id][()]  # binary data


class LabelStore(DataStore):
    def __init__(self, store_file=None, data_info=None, **kwargs):
        super().__init__(store_file, data_info, **kwargs)
        
    def get_data(self, wav_id):
        data_store = self.get_data_store(wav_id)
        label = data_store[wav_id][()]  # binary data
        # label.astype(np.int64) ?
        return label


def kaldi_read_label(label_scp, n_split=None):
    ''' read label from ark
    return: dict {wav_id: label}, label is a list of phone id (i.e., CTC label)
    '''
    # get scp file
    label_scp_list = []
    if n_split is not None:
        for i in range(n_split):
            label_scp_list.append(label_scp.format(n_split, i))
    else:
        label_scp_list = [label_scp]

    # read all scp file
    wav_scp_dict = {}
    for wav_scp in label_scp_list:
        wav_scp_dict.update(read_wav_scp(wav_scp))  # merge dict

    # get all ark file
    ark_list = []
    for wav_id in wav_scp_dict:
        data_path = wav_scp_dict[wav_id]
        ark_file, _ = str(data_path).split(':')
        if ark_file not in ark_list:
            ark_list.append(ark_file)

    # read label data
    all_label_data = {}
    for ark_file in ark_list:
        for key, value in kaldi_io.read_vec_int_ark(ark_file):
            all_label_data[key] = value
    label_data = {}  # subset of all_label_data
    for wav_id in wav_scp_dict:
        label_data[wav_id] = all_label_data[wav_id]
    # logger.debug(f'ark_list: {len(ark_list)}, wav_scp_dict: {len(wav_scp_dict)}, label_scp_list: {len(label_scp_list)}, all_label_data: {len(all_label_data)}')

    return label_data


class KaldiFeatStore:
    '''Read the feature from kaldi ark for pytorch iter-style dataset
    Args:
        rspec (str): the feature rspec string
        transform (str or None): the global transform matrix file path, first row
                                 `Addshift`, second row `Rescale`. No ``[`` or ``]`` is needed.
        splice (tuple(int, int)): context length for previous and future frames
        stack_frame (int, optional): Google style stacking+decimation,
        skip_frame (int, optional): number of frames to skip out-side the model
                                    it is used to skip the label if skip_label is True
        target_delay (int, optional): delay the label to utilize future frames,
                                      it's applied after `skip_frame`
    '''

    def __init__(self, rspec, transform=None, splice=(0, 0), skip_frame=1, **kwargs):
        # pylint: disable=unused-argument
        self.feat_rspec = rspec
        self.splice = splice
        self.skip_frame = skip_frame
        if transform:
            logger.info(f'Use feature transform from {transform}')
            self.transform = np.recfromtxt(transform).astype('float32')
        else:
            self.transform = None

    def __iter__(self):
        '''Read the feature rspec and post-process
        Yield:
            utterence_id (str): the utterence ID of current sample
            feat (np.array): the feature vector of current sample
        '''
        for utterence_id, feat in kaldi_io.read_mat_ark(self.feat_rspec):
            feat = feature_splice(feat, self.splice)
            if self.transform is not None:
                feat = (feat + self.transform[0]) * self.transform[1]
            feat = apply_skip_frame(feat, self.skip_frame)
            yield utterence_id, feat
            
