# Author: JiangWenbin, 2022.02
# Self-supervised learning Speech Enhancement
import torch
import copy
import random
import importlib
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from modules.utils.logging import logger
from modules.utils.common import torch_float32, merge_dicts
from modules.utils import audio, torch_audio
from .system import BaseSE, ComplexSE, UniSE
from .. import loss as loss_module


class Noise2NoiseBase(BaseSE):
    def __init__(self, conf):
        super().__init__(conf)        
        self.mix_param = conf['data'].get('mix_param')
        self.pairs = conf['data'].get('pairs', 'noisy2clean')
        self.remix = conf['data'].get('remix', 0)
        self.remix_norm = conf['data'].get('remix_norm', False)
        self.PHA_ReMixIT = conf['data'].get('PHA_ReMixIT', False)
        self.warm_up = conf['system'].get('warm_up', 0)
        self.n_history = conf['system'].get('n_history', 1)
        if self.n_history == 1:
            self.model_pre = None
            self.iter_p = conf['system'].get('iter_p', 0.0)
            self.init_p = self.iter_p
            self.delta_p = conf['system'].get('delta_p', 0)
            if self.delta_p == 'auto':
                self.delta_p = (1.0 - self.iter) / (self.conf.trainer.max_epochs - self.warm_up)
        else: # p_list, the prob of using unprocessed data, the history model, the latest model
            self.p_list = [float(item) for item in conf['system'].get('p_list', '0.1,0.1,0.8').split(',')]
            self.model_list = []
        
        self.embedding_loss = None
        if 'embedding_loss' in conf and conf['embedding_loss']['weight'] > 0:
            self.embedding_loss = loss_module.EmbeddingLoss(conf['embedding_loss'])
        self.pesq_loss = None
        if 'pesq_loss' in conf and conf['pesq_loss']['weight'] > 0:
            self.pesq_loss = loss_module.PESQLoss(conf['pesq_loss'])
    
    def setup(self, stage):
        super().setup(stage)
        self.init_noise_iterator()
        
    def init_noise_iterator(self):
        # initialize noise iterator
        module = importlib.import_module('modules.dataset.dataset_iter')
        noise_conf = self.conf.copy()
        noise_conf['data']['infinite'] = True
        p_data = getattr(module, 'NoiseDataset')(noise_conf)
        logger.info(f'{type(p_data).__name__}')
        kwargs = OmegaConf.to_container(noise_conf['data']['dataloader'], resolve=True) # dict
        kwargs.pop('collate_fn', None)
        kwargs.pop('shuffle', None)
        self.noise_dataloader = DataLoader(p_data, **kwargs)
        self.noise_iterator = iter(self.noise_dataloader)
    
    def get_noise_batch(self, ref_wav, remix_noise=None):
        n_batch, _ = ref_wav.shape
        if remix_noise is not None:
            return remix_noise[torch.randperm(n_batch)]
        noise = next(self.noise_iterator)
        noise = torch_float32(noise).type_as(ref_wav)
        noise = noise[:n_batch,:] # the last data
        return noise
    
    def on_train_epoch_start(self):
        if self.n_history == 1:
            if self.iter_p > 0 and self.current_epoch > self.warm_up:
                self.model_pre = copy.deepcopy(self.model)
                self.model_pre.eval()
                self.iter_p = self.init_p + (self.current_epoch - self.warm_up) * self.delta_p
                # print(f"epoch: {self.current_epoch}, iter: {self.iter}")
            else:
                self.model_pre = None
        else: # new strategy
            if self.current_epoch > self.warm_up:
                model = copy.deepcopy(self.model)
                model.eval()
                if len(self.model_list) > self.n_history:
                    self.model_list.pop(0) # pop the first model
                self.model_list.append(model) # push the latest model
        
    def snr_mixer(self, params, clean, noise, value=0):
        clean, _, noisy, rms_level = torch_audio.snr_mixer(params, clean, noise)
        if self.remix_norm:
            noisy = torch_audio.normalize_with_ref(noisy, clean)
        if torch.isnan(noisy).any():
            logger.warning(f"snr_mixer: {self.global_step}, noisy nan, fill with {value}")
            noisy = torch.where(torch.isnan(noisy), torch.full_like(noisy, value), noisy)
        return clean, noisy, rms_level
    
    def get_PHA_noise(self, ref_wav, remix_noise):
        n_batch, _ = ref_wav.shape
        noise = self.get_noise_batch(ref_wav)
        combine_noise = torch.concat([noise, remix_noise], dim=0)
        idx = torch.randperm(2 * n_batch)[:n_batch]
        return combine_noise[idx]
        
    def get_data_pair(self, est, clean=None, remix_noise=None):
        if self.PHA_ReMixIT and remix_noise is not None:
            noise = self.get_PHA_noise(est, remix_noise)
        else:
            noise = self.get_noise_batch(est, remix_noise)
        target, noisy, _ = self.snr_mixer(self.mix_param, est, noise)        
        return noisy, target
    
    def n2n_test(self, noisy_wav, clean_wav):
        noisy_wav = noisy_wav.squeeze().cpu()
        noise = self.online_noise.get_noise(n_sample=len(noisy_wav))
        _, _, noisier_wav, _ = audio.snr_mixer(self.mix_param, noisy_wav, noise, self.snr)
        noisier_wav = torch_float32(noisier_wav).type_as(clean_wav).unsqueeze(0)
        noisier_spec = self.stft.apply_stft(noisier_wav)
        est_spec = self.model(noisier_spec)
        n2n_alpha = self.conf.get('n2n_alpha', 1.0)
        final_spec = ((1 + n2n_alpha ** 2) * est_spec - noisier_spec) / (n2n_alpha ** 2)
        est_wav = self.stft.apply_istft(final_spec)
        return est_wav, None
    
            
class N2N_ComplexSE(Noise2NoiseBase, ComplexSE):
    def __init__(self, conf):
        super().__init__(conf)
        
    def forward(self, noisy_wav, clean_wav=None, train=True):
        if self.conf.get('n2n_test', False) is True:
            return self.n2n_test(noisy_wav, clean_wav)
        else:
            if self.model_pre is not None and random.random() < self.iter_p: # Iterative Data Refinement
                noisy_spec = self.stft.apply_stft(noisy_wav)
                est_spec = self.model_pre(noisy_spec)
                est_wav = self.stft.apply_istft(est_spec)
                noisy_wav, clean_wav = self.get_data_pair(est_wav.detach(), clean_wav)
            return super().forward(noisy_wav, clean_wav, train)

Noise2Noise = N2N_ComplexSE

class N2N_UniSE(Noise2NoiseBase, UniSE):
    def __init__(self, conf):
        default_conf = {
            # 'mag_data': False,
            # 'mag_decoder': False,
            # 'spec_decoder': True,
            'rnn': {'bidirectional': True}
        }
        conf['model'] = merge_dicts(default_conf, conf['model']) # merge
        super().__init__(conf)
    
    def denoise_addnoise(self, model, noisy_wav, clean_wav):
        # first denoising with the model, then add noise
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_input = torch.cat([noisy_spec.real, noisy_spec.imag], dim=1)
        with torch.no_grad():
            spec_mask, _ = model(noisy_input)
        spec_mask = torch.complex(spec_mask[:,0,...], spec_mask[:,1,...]).unsqueeze(1)
        est_spec = spec_mask * noisy_spec if self.mask else spec_mask
        est_wav = self.stft.apply_istft(est_spec)        
        remix_noise = noisy_wav - est_wav if random.random() < self.remix else None
        return self.get_data_pair(est_wav.detach(), clean_wav, remix_noise)
    
    def forward(self, noisy_wav, clean_wav=None, train=True):
        if (not self.model.training):
            return super().forward(noisy_wav, clean_wav, train)
        if self.n_history == 1:
            if self.model_pre is not None and random.random() < self.iter_p:
                noisy_wav, clean_wav = self.denoise_addnoise(self.model_pre, noisy_wav, clean_wav)
        else: # new strategy
            if len(self.model_list) > 0:
                p = random.random() # U[0, 1)
                if p < self.p_list[-1]: # the latest model
                    model = self.model_list[-1]
                    noisy_wav, clean_wav = self.denoise_addnoise(model, noisy_wav, clean_wav)
                # the history model
                elif (p >= self.p_list[-1] and p < (self.p_list[-1] + self.p_list[-2])):
                    # model_list may only has one model
                    model = random.choice(self.model_list[:-1] or self.model_list)
                    noisy_wav, clean_wav = self.denoise_addnoise(model, noisy_wav, clean_wav)
        _loss, est_wav = super().forward(noisy_wav, clean_wav, train)
        if self.embedding_loss is not None:
            _loss = _loss + self.embedding_loss(clean_wav, est_wav) * \
                self.conf['embedding_loss']['weight']
        if self.pesq_loss is not None:
            _loss = _loss + self.pesq_loss(clean_wav, est_wav) * \
                self.conf['pesq_loss']['weight']
        return _loss, est_wav
    
    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)        
        state = self.state_dict()
        for k in list(state.keys()):
            if k.startswith('model_pre.'):
                state.pop(k)
        checkpoint['state_dict'] = state

class Self2SelfSE(Noise2NoiseBase, UniSE):
    # note: It is not the source code for SelfSE paper !
    def __init__(self, conf):
        default_conf = {
            'rnn': {'bidirectional': True}
        }
        conf['model'] = merge_dicts(default_conf, conf['model']) # merge
        super().__init__(conf)
        
        self.supervised = conf['system'].get('supervised', False)
        self.SE_loss_w = conf['loss'].get('weight', 1.0)
        
    def forward(self, noisy_wav, clean_wav=None, train=True):
        if (not self.model.training):
            return super().forward(noisy_wav, clean_wav, train)
        ref_wav = clean_wav if self.supervised else noisy_wav
        SE_loss, est_wav = super().forward(noisy_wav, ref_wav, train)
        _loss = 0
        if self.SE_loss_w is not None:
            _loss = _loss + SE_loss * self.SE_loss_w
        if self.embedding_loss is not None:
            _loss = _loss + self.embedding_loss(clean_wav, est_wav) * \
                self.conf['embedding_loss']['weight']
        return _loss, est_wav
    
        
class Self2Self(ComplexSE):
    def __init__(self, conf):
        super().__init__(conf)
    
    def train_dataloader(self):
        input = self.conf.get('train_input', self.conf.get('input', 'noisy'))
        output = self.conf.get('train_output', self.conf.get('output', 'noisy'))
        p_data = WavDatasetRandom(self.conf, subset='train', 
                                            input=input, output=output)
        return DataLoader(p_data, shuffle=self.conf["shuffle"], 
                          batch_size=self.batch_size, num_workers=4)
        
    def val_dataloader(self):
        input = self.conf.get('valid_input', self.conf.get('input', 'noisy'))
        output = self.conf.get('valid_output', self.conf.get('output', 'clean'))
        p_data = WavDatasetRandom(self.conf, subset='valid', split_wav=True, 
                                            input=input, output=output)
        return DataLoader(p_data, batch_size=self.batch_size, num_workers=4)
                
    def test_dataloader(self):
        input = self.conf.get('test_input', self.conf.get('input', 'noisy'))
        output = self.conf.get('test_output', self.conf.get('output', 'clean'))
        p_data = WavDatasetRandom(self.conf, subset='test', split_wav=False, 
                                            input=input, output=output)
        return DataLoader(p_data, batch_size=1, num_workers=1)  
