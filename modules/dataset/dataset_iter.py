import math
import time
import torch
import random
import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from modules.utils.common import get_rank
from modules.utils.logging import logger
from modules.utils import audio
from .dataset_base import DatasetBase
from .common import check_wav_value


class DatasetIterable(IterableDataset, DatasetBase):
    def __init__(self, conf, subset):
        IterableDataset.__init__(self)
        DatasetBase.__init__(self, conf, subset)
        self.buffer_size = conf['data'].get('shuffle_buffer_size', 1000)
        self.sample_density = conf['data'].get('sample_density', 1.0)

    def _get_partition(self, full_list):
        """
        Partition data across GPUs and workers.
        """
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = max(0, get_rank())
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        total_samples = len(full_list)
        total_processes = world_size * num_workers
        g_worker_id = rank * num_workers + worker_id
        base_size = total_samples // total_processes
        remainder = total_samples % total_processes
        size = base_size + (1 if g_worker_id < remainder else 0)
        offset = (base_size * g_worker_id) + min(g_worker_id, remainder)
        partition = full_list[offset: offset + size]
        logger.debug(f'[{g_worker_id}/{total_processes}] {len(partition)}: [{offset}, {offset + size}]')
        assert len(partition) > 0, f'[{g_worker_id}/{total_processes}] {world_size} {rank} {total_samples} {offset} {size}'
        return partition

    def _shuffle_stream(self, iterator):
        """
        Perform stream-based shuffling using a buffer.
        """
        buf = []
        for item in iterator:
            buf.append(item)
            if len(buf) >= self.buffer_size:
                idx = random.randint(0, len(buf) - 1)
                yield buf[idx]
                buf[idx] = buf[-1] # last item to idx
                buf.pop()
        random.shuffle(buf)
        for item in buf:
            yield item

    def __iter__(self):
        def generator():
            run_loop = True
            while run_loop:
                yield from self._yield_data()
                if not self.infinite:
                    run_loop = False
        return self._shuffle_stream(generator())

    def _yield_data(self, *args, **kwargs):
        raise NotImplementedError
    

class NoiseDataset(DatasetIterable):
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noise']
        self.infinite = conf['data'].get('infinite', True)
        # self.chunk_list = self._get_partition(self.chunk_info.get('noise', []))
    
    def _yield_data(self):
        chunk_list = self._get_partition(self.chunk_info.get('noise', []))
        for wav_id, begin, end in chunk_list:
            noise = self.get_wav_data(wav_id, 'noise')
            yield self.check_wav_data(noise[..., begin : end])


class WavDataset(DatasetIterable):
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noisy'] 
        self.normalize = conf.get('normalize')
        self.infinite = conf['data'].get('infinite', False)
    
    def _yield_data(self):
        chunk_list = self._get_partition(self.chunk_info.get('noisy', []))
        for noisy_wav_id, begin, end in chunk_list:
            clean_wav_id = noisy_wav_id
            if self.unpaired is True:
                clean_wav_id = self.data_store['clean'].sample_id(deny_list=[noisy_wav_id])
            yield self._process_wav(
                self.get_wav_data(noisy_wav_id, 'noisy')[begin:end],
                self.get_wav_data(clean_wav_id, 'clean')[begin:end] if self.data_store.get('clean') else None,
                noisy_wav_id, clean_wav_id)
            
    def _process_wav(self, noisy_wav, clean_wav, noisy_wav_id, clean_wav_id):
        noisy_wav = self.check_wav_data(noisy_wav)
        normfac = noisy_wav.abs().max() if self.normalize else 1.0
        data = {'noisy_wav': noisy_wav, 'wav_id': noisy_wav_id}
        if self.normalize:
            data['noisy_wav'] = data['noisy_wav'] / normfac
        if clean_wav is not None:
            clean_wav = self.check_wav_data(clean_wav)
            if self.normalize:
                clean_wav = clean_wav / normfac
            data['clean_wav'] = clean_wav
        if self.label_dict is not None and len(self.label_dict) > 0:
            data['label'] = self.label_dict[clean_wav_id]
        return data


class OnlineNoisyDataset(DatasetIterable):
    '''
    noisy2clean: clean + noise -> clean
    noisy2noisy: clean + noise1 -> clean + noise2
    noisier2noisy: clean + noise1 + noise2 -> clean + noise1
    noisier2noisy_v2: noisy + noise -> noisy
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        if self.pairs == 'noisier2noisy_v2':
            assert (self.data_store.get('noisy') and self.data_store.get('noise'))
            self.main_key = 'noisy'
        else:
            assert (self.data_store.get('clean') and self.data_store.get('noise'))
            self.main_key = 'clean'
        # self.chunk_list = self._get_partition(self.chunk_info.get(self.main_key, []))
        
        self.infinite = conf['data'].get('infinite', False)
        self.mix_param = conf['data']['mix_param']
        self.add_rir = conf['data'].get('add_rir', 0)
        if self.add_rir > 0:
            assert self.data_store.get('rir')
        logger.info(f"pairs: {self.pairs}, [{self.mix_param.snr_lower}, {self.mix_param.snr_upper}] dB")
        self.suppression_ratio = self.mix_param.get('suppression_ratio')
        if self.suppression_ratio:
            logger.info(f'suppression_ratio: {self.suppression_ratio} dB')
        
        noise_conf = conf.copy()
        noise_conf['data']['infinite'] = True
        self.noise_dataset = NoiseDataset(noise_conf)
        self.noise_iter = iter(self.noise_dataset)

    def _yield_data(self):
        chunk_list = self._get_partition(self.chunk_info.get(self.main_key, []))
        for wav_id, begin, end in chunk_list:
            data = self._online_mix_chunk(
                self.get_wav_data(wav_id, self.main_key)[..., begin : end], 
                wav_id)
            if data is not None:
                yield data

    def _online_mix_chunk(self, chunk, wav_id):
        clean = self.check_wav_data(chunk)
        if random.random() < self.add_rir:
            clean = self.add_pyreverb(clean)
        # if clean.sum() == 0:
        if check_wav_value(clean) is False:
            return None
        
        noise1 = next(self.noise_iter)
        snr = self.snr_n1
        clean, _, noisy, _ = audio.snr_mixer(self.mix_param, clean, noise1, snr)
        if self.pairs in ['noisy2clean', 'noisier2noisy_v2']:
            if self.suppression_ratio:
                target_snr = snr + self.suppression_ratio
                _, _, target, target_rms_level = audio.snr_mixer(self.mix_param, clean, noise1, target_snr)
                noisy = audio.normalize(noisy, target_rms_level)    
            else:
                target = clean
        elif self.pairs == 'noisy2noisy':
            noise2 = next(self.noise_iter)
            _, _, target, target_rms_level = audio.snr_mixer(self.mix_param, clean, noise2, self.snr)
            noisy = audio.normalize(noisy, target_rms_level)
        elif self.pairs == 'noisier2noisy':
            noise2 = next(self.noise_iter)
            _, _, noisier, noisier_rms_level = audio.snr_mixer(self.mix_param, noisy, noise2, self.snr)
            target = noisy
            noisy = noisier
            target = audio.normalize(target, noisier_rms_level)
        else:
            raise ValueError(f'pairs: {self.pairs} is not supported')

        data = {'clean_wav': target.astype(np.float32), 
                'noisy_wav': noisy.astype(np.float32), 
                'wav_id': wav_id}
        return data


def test_dataset(conf):
    from modules.utils.logging import logger
    from tqdm import tqdm
    subset = conf.get('subset', 'train')
    kwargs = OmegaConf.to_container(conf['data']['dataloader'], resolve=True) # dict
    kwargs.pop('collate_fn')
    kwargs.pop('shuffle')
    kwargs['drop_last'] = True
    kwargs['persistent_workers'] = True
    p_dataset = eval(conf.dataset_name or conf['data'][subset]['name'])(conf, subset)
    dataloader = DataLoader(p_dataset, **kwargs)

    batch = next(iter(dataloader))
    logger.debug(type(batch))
    if type(batch) is torch.Tensor:
        logger.debug(batch.shape)
    else:
        logger.debug(batch.keys())
        logger.debug(batch["noisy_wav"].shape)
    cnt = 0
    start_time = time.perf_counter()
    for _ in tqdm(dataloader):
        cnt += 1
    elapsed_time = time.perf_counter() - start_time
    logger.debug(f"Total batches: {cnt}")
    logger.debug(f"Total time: {elapsed_time:.4f} seconds")
    logger.debug(f"Average time per batch: {elapsed_time / cnt:.4f} seconds")


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    conf = OmegaConf.create({
         "conf": "conf/config.yaml",
         "cmd": "test_dataset",
         "dataset_name": "NoiseDataset",
         })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    eval(conf.cmd)(conf)