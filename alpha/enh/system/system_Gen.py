# Author: JiangWenbin, 2025.07
# Generative Speech Enhancement
import importlib
import sys
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from modules.utils.logging import logger
from modules.utils.common import merge_dicts, pad2even, check_model_isnan
from .. import model as models
from .system import BaseSE

def pad_frame(Y):
    # [B C F T], T必须是64的整数倍
    T = Y.shape[-1]
    if T % 64 != 0:
        num_pad = 64-T%64
    else:
        num_pad = 0
    pad2d = torch.nn.ZeroPad2d((0, num_pad, 0, 0))
    return pad2d(Y)

class MatchaSE(BaseSE):
    # based on Matcha-TTS decoder
    def __init__(self, conf):
        super().__init__(conf)
        self.spec_type = self.model_conf.get('spec_type', 'mag') # 'mag' or 'complex'
        if self.spec_type == 'mag':
            in_channels = self.model_conf['n_freqs']
            out_channel = self.model_conf['n_freqs']
        else:
            in_channels = 2*self.model_conf['n_freqs']
            out_channel = 2*self.model_conf['n_freqs']
        if self.model_conf['cfm_params'].get('condition', True):
            in_channels = 2 * in_channels
        self.model = self.model_class(
            in_channels=in_channels, out_channel=out_channel,
            cfm_params=OmegaConf.create(self.model_conf["cfm_params"]),
            decoder_params=self.model_conf["decoder_params"],
            n_spks=0, spk_emb_dim=None)
        assert self.model_conf['mask'] == False
        
    def forward(self, noisy_wav, clean_wav=None, **kwargs):
        noisy_spec = self.stft.apply_stft(noisy_wav)
        n_batch, _, _, n_frames = noisy_spec.shape
        processed_noisy = noisy_spec.squeeze(1)[:, :-1, :]
        if self.spec_type == 'mag':
            noisy_mag = torch.abs(processed_noisy)
            noisy_phase = torch.angle(processed_noisy)
            noisy_input = pad2even(noisy_mag)
        else:
            noisy_input = torch.cat([processed_noisy.real, processed_noisy.imag], dim=1)
            noisy_input = pad2even(noisy_input)
        mask = pad2even(torch.ones((n_batch, 1, n_frames), device=noisy_wav.device))
        
        if self.trainer.training:
            assert clean_wav is not None, "Training requires clean_wav"
            clean_spec = self.stft.apply_stft(clean_wav).squeeze(1)[:, :-1, :]
            if self.spec_type == 'mag':
                clean_input = torch.abs(clean_spec)
            else:
                clean_input = torch.cat([clean_spec.real, clean_spec.imag], dim=1)
            clean_input = pad2even(clean_input)
            diff_loss, _ = self.model.compute_loss(clean_input, mask, noisy_input)
            return diff_loss, None
        
        model_out = self.model(noisy_input, mask, **self.model_conf['forward_params'])
        if n_frames % 2 != 0:
            model_out = model_out[..., :-1]
        if self.spec_type == 'mag':
            est_spec = model_out.unsqueeze(1) * torch.exp(1.0j * noisy_phase)
        else:
            est_spec = torch.complex(*torch.chunk(model_out, 2, dim=1)).unsqueeze(1)
            
        if self.model_conf['pad_value'] in [0, 1]:
            est_spec = F.pad(est_spec, [0, 0, 0, self.model_conf['pad_value']])
        else:
            est_spec = torch.cat([est_spec, noisy_spec[..., -1:, :]], dim=-2)
        est_wav = self.stft.apply_istft(est_spec)
        _loss = self.loss(clean_wav, est_wav) if clean_wav is not None else None
        return _loss, est_wav


class SpecFlowSE(BaseSE):
    '''Spectral-domain Flow Matching Speech Enhancement'''
    def __init__(self, conf):
        super().__init__(conf)
        self.model = self.model_class(self.model_conf)
        self.ode_param = self.model_conf['ode_param']
        self.ode_param['fn'] = self.model
        self.ode_name = self.ode_param.get('name', 'FlowMSE') # FlowMSE, Naive
        self.ode = getattr(models, self.ode_name)(self.ode_param)
        self.negative = self.ode_param.get('negative', False)
        self.valid_wav_every_n_epoch = self.model_conf.get('valid_wav_every_n_epoch', 4)
        self.complex = self.model_conf.get('complex', True) # complex or mag
        self.flow_loss = self.model_conf.get('flow_loss', 'MSE') # Flow Matching loss
        self.wav_loss = self.model_conf.get('wav_loss', False)
        self.flow_loss_w = self.model_conf.get('flow_loss_w', 2e-4)
        self.pesq_weight = self.conf['loss'].get('pesq_weight', 0.0)
        if self.pesq_weight > 0:
            self.pesq_loss = self.get_loss({'name':'PESQLoss'})
        logger.info(f'wav_loss: {self.wav_loss}, flow_loss_w: {self.flow_loss_w}, '
                    f'pesq_weight: {self.pesq_weight}')
    
    def get_flow_loss(self, clean, y, **kwargs):
        # y: noisy
        rdm = (1-torch.rand(y.shape[0], device=y.device)) * \
            (self.ode_param['T_rev'] - self.ode_param['t_eps']) + self.ode_param['t_eps']
        t = torch.min(rdm, torch.tensor(self.ode_param['T_rev']))
        if self.ode_name == 'FlowMSE':
            x0 = clean
            mean, std = self.ode.marginal_prob(x0, t, y)
            z = torch.randn_like(y)
            xt = mean + std[:, None, None, None] * z
            target = self.ode.der_mean(x0, y) + self.ode.der_std() * z
            model_input = self.ode.get_model_input(xt, y)
            vectorfield = self.model(model_input, t)
        elif self.ode_name == 'Naive':
            T = t[:, None, None, None]
            x0 = self.ode.prior_sampling(y)
            # logger.info(f'{t.shape}, {x0.shape}, {clean.shape}')
            xt = (1 - T) * x0 + T * clean
            target = x0 - clean if self.negative else clean - x0
            model_input = self.ode.get_model_input(xt, y)
            vectorfield = self.model(model_input, t)
        else:
            raise NotImplementedError(f'{self.ode_name}')
        if self.flow_loss == 'MSE':
            losses = torch.abs(vectorfield - target)**2  # power of the complex value
            per_sample_loss = 0.5 * losses.flatten(1).sum(dim=1)
        elif self.flow_loss == 'euclidean':
            per_sample_loss = torch.norm(torch.view_as_real(
                vectorfield) - torch.view_as_real(target), p=2, dim=-1)
        return per_sample_loss.mean()
    
    def get_wav_loss(self, clean_wav, noisy_spec, pad):
        est_wav = self.get_est_wav(noisy_spec, pad)
        min_len = min(est_wav.shape[-1], clean_wav.shape[-1])
        est_wav, clean_wav = est_wav[..., :min_len], clean_wav[..., :min_len]
        # logger.info(f'clean_wav: {clean_wav.shape}, est_wav: {est_wav.shape}')
        _loss = self.loss(clean_wav, est_wav)
        if self.pesq_weight > 0:
            _pesq_loss = self.pesq_loss(clean_wav, est_wav)
            if self.trainer.training:
                self.log("train/pesq_loss", _pesq_loss)
            _loss = _loss + self.pesq_weight * _pesq_loss
        return _loss
    
    def get_est_wav(self, noisy_spec, pad=True):
        n_frame = noisy_spec.shape[-1]
        noisy_data = noisy_spec if self.complex else torch.abs(noisy_spec)
        if pad:
            noisy_data = pad_frame(noisy_data)
        est_data = self.ode.solve_euler(noisy_data)
        est_data = est_data[..., :n_frame]
        # logger.info(f'est_data: {est_data.shape}, noisy_spec: {noisy_spec.shape}')
        est_wav = self.stft.apply_istft(
            est_data if self.complex else est_data * torch.exp(1j * torch.angle(noisy_spec)))
        return est_wav
    
    def forward(self, noisy_wav, clean_wav=None, **kwargs):
        noisy_spec = self.stft.apply_stft(noisy_wav)
        _loss, est_wav = None, None
        if self.trainer.training:
            assert clean_wav is not None, "Training requires clean_wav"
            clean_spec = self.stft.apply_stft(clean_wav)
            _loss = self.get_flow_loss(clean_spec if self.complex else torch.abs(clean_spec), 
                                       noisy_spec if self.complex else torch.abs(noisy_spec))
            if self.wav_loss:
                wav_loss = self.get_wav_loss(clean_wav, noisy_spec, pad=False)
                self.log("train/wav_loss", wav_loss)
                _loss = self.flow_loss_w * _loss + wav_loss
            return _loss, est_wav
        
        if clean_wav is not None:
            clean_spec = self.stft.apply_stft(clean_wav)
            _loss = self.get_flow_loss(pad_frame(clean_spec if self.complex else torch.abs(clean_spec)), 
                                       pad_frame(noisy_spec if self.complex else torch.abs(noisy_spec)))
            if self.wav_loss:
                wav_loss = self.get_wav_loss(clean_wav, noisy_spec, pad=True)
                _loss = self.flow_loss_w * _loss + wav_loss
        if (self.trainer.validating and ((self.current_epoch + 1) % self.valid_wav_every_n_epoch == 0)) \
            or self.trainer.testing:
            est_wav = self.get_est_wav(noisy_spec)
        
        return _loss, est_wav    


      