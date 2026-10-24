import torch
import numpy as np
from modules.utils.common import torch_float32, refine_state_dict
from omegaconf import OmegaConf
from pathlib import Path
from modules.system import BaseModule
import modules.model.arch as model_arch
from modules.stft import STFT
import alpha.enh.model as enh_models
import alpha.asr.models as asr_models
import alpha.enh.loss as enh_loss_module
import alpha.asr.loss as asr_loss_module
from modules.utils.logging import logger


class BaseEnhASR(BaseModule):
    def __init__(self, conf):
        super().__init__(conf)
        self.automatic_optimization = False # !!!
        self.stft = STFT(self.stft_conf)
        self.feature = None
        self.deepfilter = self.init_deepfilter(conf)
        assert self.feat_extractor is not None
        
        enh_model_conf, asr_model_conf = conf['model']['enh'], conf['model']['asr']
        self.enh_model_class = getattr(enh_models, enh_model_conf['name'])
        self.enh_model_detach = enh_model_conf.get('detach', True)
        self.enh_model_freeze = enh_model_conf.get('freeze', False)
        self.enh_param = self.get_model_param(enh_model_conf)        
        self.asr_model_class = getattr(asr_models, asr_model_conf['name'])
        self.asr_model_freeze = asr_model_conf.get('freeze', False)
        self.asr_param = self.get_model_param(asr_model_conf)        
        self.enh_model, self.asr_model = None, None
        
        enh_loss_conf, asr_loss_conf = conf['loss']['enh'], conf['loss']['asr']
        self.enh_loss_name = enh_loss_conf['name']
        self.enh_loss = self.get_loss(enh_loss_conf)
        self.asr_loss_name = asr_loss_conf['name']
        self.asr_loss = self.get_loss(asr_loss_conf)

    def get_model_param(self, model_conf):
        name = model_conf.get('name')
        param = model_conf.get('param')
        if not param:
            return None
        if param.endswith('.yaml'):
            return OmegaConf.load(Path('conf') / param).get(name)
        return getattr(model_arch, param, None)

    def get_loss(self, loss_conf):
        name = loss_conf.pop('name', None)
        if not name:
            raise ValueError('loss name is required in loss_conf')
        if hasattr(enh_loss_module, name):
            return getattr(enh_loss_module, name)(loss_conf)
        if hasattr(asr_loss_module, name):
            return getattr(asr_loss_module, name)(loss_conf)
        if hasattr(torch.nn, name):
            return getattr(torch.nn, name)()
        raise ValueError(f'loss name error: {name}')
    
    def init_model(self, model_conf):
        # init enh model
        enh_model_path = model_conf['enh'].get('init')
        if enh_model_path:
            logger.info(f'Enh model initialize with: {enh_model_path}')
            enh_model_dict = torch.load(enh_model_path)
            enh_state_dict = enh_model_dict.get('state_dict')
            assert enh_state_dict, "Key 'state_dict' not found in enhancement model dictionary"
            enh_state_dict = refine_state_dict(enh_state_dict)
            self.enh_model.load_state_dict(enh_state_dict)
        # init asr model
        asr_model_path = model_conf['asr'].get('init')
        if asr_model_path:
            logger.info(f'ASR model initialize with: {asr_model_path}')
            asr_model_dict = torch.load(asr_model_path)
            asr_state_dict = asr_model_dict.get('model')
            assert asr_state_dict, "Key 'model' not found in ASR model dictionary"
            self.asr_model.load_state_dict(asr_state_dict)

    def init_deepfilter(self, conf):
        deepfilter = conf.get('deepfilter', None)
        if deepfilter:
            assert self.mask is True
            deepfilter.update({
                'n_freq': conf['feature']['n_feature'] if self.feature else
                conf['stft']['n_fft']//2 + 1})
        return deepfilter

    def denoise(self, y, chunk=-1):
        # y: shape of [1, n_samples], chunk: in seconds, -1, use all data
        y = torch_float32(y, self.device)
        if chunk > 0:
            counter = 0
            chunk_samples = self.sample_rate * chunk
            n_chunk = int(np.ceil(len(y)/chunk_samples))
            est_wav = torch.zeros_like(y)
            for idx in range(n_chunk):
                start = idx * chunk_samples
                end = None if idx == n_chunk - 1 else (idx+1) * chunk_samples
                y_chunk = y[start:end]
                with torch.no_grad():
                    est_wav_chunk = self.forward(
                        self.padding(y_chunk.unsqueeze(0)), train=False).squeeze()
                est_wav_chunk = est_wav_chunk[0:len(y_chunk)]  # remove the padded
                est_wav[start:end] = est_wav_chunk
                counter += len(y_chunk)
            assert len(y) == counter
        else:
            with torch.no_grad():
                est_wav = self.forward(
                    self.padding(y.unsqueeze(0)), train=False).squeeze()
            est_wav = est_wav[0:len(y)]  # remove the padded
        return est_wav
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def enh_forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def asr_forward(self, *args, **kwargs):
        raise NotImplementedError
        
    def training_step(self, batch, batch_idx):
        enh_opt, asr_opt = self.optimizers()
        
        # Optimize enh_model
        enh_loss, _  = self.enh_forward(batch['noisy_wav'], batch['clean_wav'])
        enh_opt.zero_grad()
        self.manual_backward(enh_loss)
        enh_opt.step()
        
        # Optimize asr_model
        est_wav = self.enh_forward(batch['noisy_wav'], train=False)
        if self.enh_model_detach:
            est_wav = est_wav.detach()
        feats = self.feat_extractor.get_feat_batch(est_wav, batch['wav_lengths'])
        asr_loss, asr_loss_info = self.asr_forward(feats, batch['label'])
        asr_opt.zero_grad()
        self.manual_backward(asr_loss)
        asr_opt.step()
        
        asr_loss_stat = self.asr_loss.log_stat(asr_loss_info)
        self.log_dict({"train/enh_loss": enh_loss, 'train/asr_loss': asr_loss, 
                       'train/loss_per': asr_loss_stat['loss']})
        self.log('train/acc', asr_loss_stat['token_acc'], prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        enh_loss, est_wav = self.enh_forward(batch['noisy_wav'], batch['clean_wav'])
        feats = self.feat_extractor.get_feat_batch(est_wav, batch['wav_lengths'])
        asr_loss, asr_loss_info = self.asr_forward(feats, batch['label'])
        asr_loss_stat = self.asr_loss.log_stat(asr_loss_info)
        self.valid_log('valid/enh_loss', enh_loss)
        self.valid_log('valid/asr_loss', asr_loss)
        self.valid_log('valid/loss_per', asr_loss_stat['loss'])
        self.valid_log('valid/acc', asr_loss_stat['token_acc'], prog_bar=True)
    
    def configure_optimizers(self):        
        opt_conf = OmegaConf.to_container(self.conf['optimizer'])
        assert isinstance(opt_conf, dict) and 'enh' in opt_conf and 'asr' in opt_conf
        assert self.enh_model and self.asr_model
        models = {'enh': self.enh_model, 'asr': self.asr_model}
        return [self.get_optimizer(opt_conf[key], model.parameters()) for key, model in models.items()]
            

class LiteEnhASR(BaseEnhASR):
    def __init__(self, conf):
        super().__init__(conf)
        # asr model
        self.asr_model = self.asr_model_class(**self.asr_param)
        
        # enh model
        self.enh_model_conf = {
            'complex_mask': True,
            'mag_mask': True,
            'norm': 'instance',
            'deepfilter': self.deepfilter,
            'merge': 0.5,
            'debug': self.debug
        }
        self.conf['model']['enh']['param'] = self.enh_param # update
        self.enh_model_conf.update(**self.conf['model']['enh']) # merge
        self.enh_model = self.enh_model_class(self.enh_model_conf)
        assert self.feature is None
        
    def forward(self, batch):
        pass
    
    def asr_forward(self, feats, label):
        model_out = self.asr_model(feats)
        _loss, loss_info = self.asr_loss(model_out, feats, label)
        return _loss, loss_info
    
    def enh_forward(self, noisy_wav, clean_wav=None, train=True):
        # use forward for inference/predictions
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)

        noisy_input = torch.cat([noisy_spec.real, noisy_spec.imag, noisy_mag], dim=1)
        est_spec, est_mag = self.enh_model(noisy_input)

        # complex head
        if self.enh_model_conf['complex_mask']:
            if self.deepfilter:
                est_spec = self.enh_model.deepFilter.apply_deepfilter(noisy_input, coeff=est_spec)
            else:
                est_spec = noisy_spec * est_spec
        est_spec = torch.complex(est_spec[:, 0, ...], est_spec[:, 1, ...]).unsqueeze(1)

        # magnitude head
        if self.enh_model_conf['mag_mask']:
            est_mag = noisy_mag * est_mag
        est_mag_spec = est_mag * torch.exp(1.0j*noisy_phase)

        # merge
        final_spec = (1 - self.enh_model_conf['merge']) * est_spec + \
            self.enh_model_conf['merge'] * est_mag_spec

        est_wav = self.stft.apply_istft(final_spec)
        if train is True:
            pre_spec = final_spec.squeeze(1)
            ref_spec = self.stft.apply_stft(clean_wav).squeeze(1)
            if self.enh_loss_name == 'HybridLoss':  # require spec
                _loss = self.enh_loss(torch.stack([pre_spec.real, pre_spec.imag], dim=-1),
                                  torch.stack([ref_spec.real, ref_spec.imag], dim=-1))
            else:
                _loss = self.enh_loss(clean_wav, est_wav)
            return _loss, est_wav
        else:
            return est_wav