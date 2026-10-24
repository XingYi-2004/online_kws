import torch
import sys
import random
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler as LR_scheduler
import modules.model.arch as model_arch
from modules.utils.common import EPS, torch_float32, refine_state_dict, \
    check_model_isnan, merge_dicts, resolve_path, pad2even
from modules.utils.logging import logger
from modules.utils import torch_audio
from modules.system import BaseModule
from modules.stft import STFT
from omegaconf import OmegaConf
from modules.utils import metrics
from .. import loss as loss_module
from .. import model as models


class BaseSE(BaseModule):
    def __init__(self, conf):
        super().__init__(conf)
        self.stft = STFT(self.stft_conf)
        self.model_class = getattr(models, conf['model']['name'], None)
        self.resolve_model_param()
        self.mix_param = conf['data'].get('mix_param', None)
        self.loss_name = conf['loss'].get('name')
        self.loss = self.get_loss(conf['loss'])
        self.valid_loss = self.init_valid_loss(conf.get('valid_loss', None))
        self.feature = None
        self.deepfilter = self.init_deepfilter(conf)
        self.model_conf = OmegaConf.to_container(self.conf['model'], resolve=True)
        self.check_nan = conf['system'].get('check_nan', False)
        self.rms_norm = conf['system'].get('rms_norm', False) # apply RMS normalization

    def resolve_model_param(self):
        model_param = self.conf['model'].get('param', None)
        if type(model_param) is str:
            self.conf['model']['param'] = getattr(model_arch, model_param, None)
            
    def get_loss(self, loss_conf):
        name = loss_conf.get('name')
        if not name:
            return None
        if hasattr(loss_module, name):
            loss = getattr(loss_module, name)(loss_conf)
        elif hasattr(torch.nn, name):
            loss = getattr(torch.nn, name)()
        else:
            raise ValueError(f'Loss name error: {name}')
        return loss

    def init_valid_loss(self, loss_conf):        
        return self.get_loss(loss_conf) if loss_conf else None
    
    def init_model(self, model_conf):
        model_path = model_conf.get('init')
        if model_path:
            model_path = resolve_path(model_path)
            logger.info(f'Model initialize with: {model_path}')
            model_dict = torch.load(model_path)
            if 'state_dict' in model_dict:  # pytorch lightning ckpt
                model_dict = refine_state_dict(model_dict['state_dict'])
            self.model.load_state_dict(model_dict)
            
    def init_deepfilter(self, conf):
        deepfilter = conf.get('deepfilter', None)
        if deepfilter:
            assert self.mask is True
            deepfilter.update({
                'n_freq': conf['feature']['n_feature'] if self.feature else
                conf['stft']['n_fft']//2 + 1})
        return deepfilter

    def forward(self, *args, **kwargs):
        raise NotImplementedError

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
                    _, est_chunk = self.forward(self.padding(y_chunk.unsqueeze(0)))
                    est_chunk = est_chunk.squeeze()
                est_chunk = est_chunk[0:len(y_chunk)]  # remove the padded
                est_wav[start:end] = est_chunk
                counter += len(y_chunk)
            assert len(y) == counter
        else:
            with torch.no_grad():
                _, est_wav = self.forward(self.padding(y.unsqueeze(0)))
                est_wav = est_wav.squeeze()
            est_wav = est_wav[0:len(y)]  # remove the padded
        return est_wav

    def configure_optimizers(self):
        if self.conf.get('xformer_opt', False):
            self.conf['warmup'] = True  # set the warmup flag
            optimizer = torch.optim.Adam(self.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
            def lr_lambda(epoch): return 4e-4 * (0.98 ** (max((epoch-1)//2, 0)))
            scheduler = LR_scheduler.LambdaLR(optimizer, lr_lambda)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        else:
            return super().configure_optimizers()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if self.conf.get('warmup', False):
            k, d_model, warmup_steps = 0.2, 64, 4000
            n_step = self.trainer.global_step + 1  # start from 1
            if n_step <= warmup_steps:
                lr = k * d_model**(-0.5) * min(n_step ** (-0.5), n_step * (warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                # print('Adjusting learning rate to {:.4e}.'.format(lr))

    @property
    def snr(self):
        return random.randint(self.mix_param.snr_lower, self.mix_param.snr_upper)
    
    def _calc_stft(self, x):
        x_stft = self.stft.apply_stft(x)
        x_mag = torch.abs(x_stft)
        x_phase = torch.angle(x_stft)
        return x_mag, x_phase
    
    def _calc_istft(self, x_mag, x_phase, **kwargs):
        x_stft = x_mag * torch.exp(1.0j*x_phase)
        x = self.stft.apply_istft(x_stft, **kwargs)
        return x
    
    def test_stft(self, x):
        # x: wave
        x_mag, x_phase = self._calc_stft(x)
        x_r = self._calc_istft(x_mag, x_phase)
        score = metrics.eval(x, x_r, self.sample_rate)
        print(score)


class OracleSE(BaseSE):
    # using the ground true information, just for test
    def __init__(self, conf):
        super().__init__(conf)
        assert 'oracle' in conf
        self.oracle_mag = conf['oracle'].get('mag', False)
        self.oracle_phase = conf['oracle'].get('phase', False)
        
    def forward(self, noisy_wav, clean_wav=None, train=True):
        noisy_mag, noisy_phase = self._calc_stft(noisy_wav)
        clean_mag, clean_phase = self._calc_stft(clean_wav)
        est_mag = clean_mag if self.oracle_mag else noisy_mag
        est_phase = clean_phase if self.oracle_phase else noisy_phase
        est_wav = self._calc_istft(est_mag, est_phase)
        return None, est_wav


class MagSE(BaseSE):
    # Magnitude SE
    def __init__(self, conf):
        super().__init__(conf)
        
        # the default model configuration
        model_conf = {"in_channels": 1, 'debug': self.debug}
        self.model_conf = merge_dicts(model_conf, self.conf['model'])  # merge
        self.model = self.model_class(self.model_conf)
        
        # use MIMONet and number of decoder should be 1
        assert self.model_conf['name'] == 'MIMONet'
        assert len(self.model_conf['decoder_list']) == 1 

    def forward(self, noisy_wav, clean_wav=None, train=True):
        noisy_mag, noisy_phase = self._calc_stft(noisy_wav)
        est_mag = self.model(noisy_mag)[0] # only one decoder
        est_mag = est_mag * noisy_mag if self.model_conf['decoder_list'][0]['mask'] else est_mag
        est_wav = self._calc_istft(est_mag, noisy_phase)
        if not train:
            return est_wav
        _loss = self.loss(clean_wav, est_wav)
        return _loss, est_wav
    

class ComplexSE(BaseSE):
    # Complex spectrum speech enhancement
    def __init__(self, conf):
        super().__init__(conf)
        kwargs = {'mask': self.mask, 'mask_bound': conf.get('mask_bound', 'tanh'), 
                  'debug': self.debug, 'rnn_layers': conf.get('rnn_layers', 2), 
                  'bidirectional': conf.get('bidirectional', False)}
        self.model = self.model_class(self.model_param, **kwargs)
        self.complex_nn = conf.get('complex_nn', False)

    def forward(self, noisy_wav, clean_wav=None, train=True):
        # use forward for inference/predictions
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_input = noisy_spec if self.complex_nn else torch.cat(
            [noisy_spec.real, noisy_spec.imag], dim=1)
        est_out = self.model(noisy_input)
        est_out = noisy_input * est_out if self.mask is True else est_out
        est_spec = est_out if self.complex_nn else torch.complex(
            est_out[:, 0, ...], est_out[:, 1, ...]).unsqueeze(1)
        est_wav = self.stft.apply_istft(est_spec)
        # est_wav = est_wav[:, 0:noisy_wav.shape[1]] # the wav might be padded
        if train is True:
            if self.loss_name == 'wSDRLoss':  # require noisy wav
                _loss = self.loss(clean_wav, est_wav, noisy_wav)
            # frequency domain loss
            elif self.loss_name in ['CMSELoss']:
                _loss = self.loss(self.stft.apply_stft(clean_wav), est_spec)
            else:
                _loss = self.loss(clean_wav, est_wav)
            return _loss, est_wav
        else:
            return est_wav


class LiteSE(BaseSE):
    # light-weight speech enhancement
    def __init__(self, conf):
        super().__init__(conf)
        # the default configuration
        default_conf = {}
        self.model_conf = merge_dicts(default_conf, self.model_conf) # merge
        self.model = self.model_class(self.model_conf)

    def forward(self, noisy_wav, clean_wav=None, train=True):
        noisy_spec = self.stft.apply_stft(noisy_wav).squeeze(1).unsqueeze(-1)
        noisy_input = torch.cat([noisy_spec.real, noisy_spec.imag], dim=-1)
        est_out = self.model(noisy_input)
        est_spec = torch.complex(est_out[..., 0], est_out[..., 1])
        est_wav = self.stft.apply_istft(est_spec)
        if train is True:
            clean_spec = self.stft.apply_stft(clean_wav).squeeze(1).unsqueeze(-1)
            ref_spec = torch.cat([clean_spec.real, clean_spec.imag], dim=-1)
            if self.loss_name == 'HybridLoss':  # require spec
                _loss = self.loss(est_out, ref_spec, est_wav, clean_wav)
            else:
                _loss = self.loss(clean_wav, est_wav)  # time domain loss
            return _loss, est_wav
        else:
            return est_wav


class PhaseSE(BaseSE):
    def __init__(self, conf):
        super().__init__(conf)
        in_channels = 0
        self.phase_data = conf['model'].get('phase_data', True)
        self.mag_data = conf['model'].get('mag_data', False)
        self.spec_data = conf['model'].get('spec_data', False)
        if self.phase_data:
            in_channels = in_channels + 1
        if self.mag_data:
            in_channels = in_channels + 1
        if self.spec_data:
            in_channels = in_channels + 2
        # the default model configuration
        model_conf = {
            "in_channels": in_channels,
            'debug': self.debug,
            'merge': 0.5
        }
        self.model_conf = merge_dicts(model_conf, self.conf['model'])  # merge
        self.model = self.model_class(self.model_conf)
        self.n_decoder = len(self.model_conf['decoder_list'])
        self.unwarp = conf['system'].get('unwarp', None)
        self.unwrapping = signal.Unwrapping(conf)

    def forward(self, noisy_wav, clean_wav=None, train=True, ret_mask=False):
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = self.unwrapping(
            noisy_wav if self.unwarp == 'fast_unwrap' else torch.angle(noisy_spec))
        data_list = []
        if self.phase_data:
            data_list.append(noisy_phase/torch.pi)
        if self.mag_data:
            data_list.append(noisy_mag)
        if self.spec_data:
            data_list.extend([noisy_spec.real, noisy_spec.imag])
        noisy_input = torch.cat(data_list, dim=1)
        
        est_out = self.model(noisy_input)
        if self.n_decoder == 1:  # phase
            est_phase = est_out[0] * torch.pi
            est_phase = est_phase * noisy_phase if self.model_conf['decoder_list'][0]['mask'] else est_phase
            final_spec = noisy_mag * torch.exp(1.0j*signal.wrap(est_phase))
        elif self.n_decoder == 2:  # phase, magnitude
            est_phase, est_mag = est_out
            est_phase = est_phase * torch.pi
            est_phase = est_phase * noisy_phase if self.model_conf['decoder_list'][0]['mask'] else est_phase
            est_mag = est_mag * noisy_mag if self.model_conf['decoder_list'][1]['mask'] else est_mag
            final_spec = est_mag * torch.exp(1.0j*signal.wrap(est_phase))
        elif self.n_decoder == 3:  # phase, magnitude, complex
            est_phase, est_mag, est_spec = est_out
            est_phase = est_phase * torch.pi
            est_phase = est_phase * noisy_phase if self.model_conf['decoder_list'][0]['mask'] else est_phase
            est_mag = est_mag * noisy_mag if self.model_conf['decoder_list'][1]['mask'] else est_mag
            est_spec = torch.complex(est_spec[:, 0, ...], est_spec[:, 1, ...]).unsqueeze(1)
            est_spec = est_spec * noisy_spec if self.model_conf['decoder_list'][2]['mask'] else est_spec
            final_spec = (1 - self.model_conf['merge']) * est_spec + \
                self.model_conf['merge'] * est_mag * torch.exp(1.0j*signal.wrap(est_phase))
        
        est_wav = self.stft.apply_istft(final_spec)
        if self.rms_norm is True:
            est_wav = torch_audio.normalize_with_ref(est_wav, noisy_wav)
        if not train:
            return est_wav
        
        if (not self.model.training) and self.valid_loss is not None:
            _loss = self.valid_loss(clean_wav, est_wav)
        else:
            if self.loss_name == 'wSDRLoss': # require noisy wav
                _loss = self.loss(clean_wav, est_wav, noisy_wav)
            elif self.loss_name in ['CMSELoss']: # frequency domain loss
                _loss = self.loss(self.stft.apply_stft(clean_wav), est_spec)
            else:
                _loss = self.loss(clean_wav, est_wav)
        if ret_mask is True:
            return _loss, est_wav, est_spec, est_mag
        return _loss, est_wav


class UniSE(BaseSE):
    # Single head speech enhancement, using magnitude and/or complex stft as input
    def __init__(self, conf):
        super().__init__(conf)
        in_channels = 0
        self.spec_data = conf['model'].get('spec_data', True)
        self.mag_data = conf['model'].get('mag_data', False)
        if self.spec_data:
            in_channels = in_channels + 2
        if self.mag_data:
            in_channels = in_channels + 1
        
        # the default model configuration
        model_conf = {
            "in_channels": in_channels,
            "mag_decoder": False,
            "spec_decoder": True,
            'debug': self.debug,
            'rnn': {'bidirectional': True}
        }
        self.model_conf = merge_dicts(model_conf, self.conf['model']) # merge
        self.model = self.model_class(self.model_conf)
        
        # extra loss
        self.ext_loss = {}
        if conf.get('mag_loss', None):            
            self.ext_loss['mag_loss'] = (getattr(torch.nn, conf.get('mag_loss_fun', 'MSELoss'))(), 
                                         conf['mag_loss'])
        if conf.get('spec_loss', None):
            self.ext_loss['spec_loss'] = (loss_module.CMSELoss(), conf['spec_loss'])
        if self.ext_loss:
            logger.info(f"ext_loss: {self.ext_loss}")
        self.extra_loss_log2 = conf.get('extra_loss_log2', False)
        
    def apply_loss_fun(self, clean_wav, est_wav, clean_spec, est_spec,
                       clean_mag=None, est_mag=None):
        loss = 0.0
        loss_fun_w = self.conf.get('loss_fun_w', 1.0)
        if loss_fun_w > 0: # time-domain loss
            _loss = loss_fun_w * self.loss(clean_wav, est_wav) 
            self.log("train/time_loss", _loss, sync_dist=True, batch_size=self.batch_size)
            loss = loss + _loss
        if 'spec_loss' in self.ext_loss: # complex spectrum loss
            _loss_fun, loss_w = self.ext_loss['spec_loss']
            _spec_loss = _loss_fun(clean_spec, est_spec)
            if self.extra_loss_log2 is True:
                _spec_loss = torch.log2(_spec_loss.clip(EPS))
            _spec_loss = loss_w * _spec_loss
            self.log("train/spec_loss", _spec_loss, sync_dist=True, batch_size=self.batch_size)
            loss = loss + _spec_loss
        if 'mag_loss' in self.ext_loss: # magnitude spectrum loss
            _loss_fun, loss_w = self.ext_loss['mag_loss']
            if est_mag is None:
                est_mag = torch.abs(est_spec)
            _mag_loss = _loss_fun(clean_mag, est_mag)
            if self.extra_loss_log2 is True:
                _mag_loss = torch.log2(_mag_loss.clip(EPS))
            _mag_loss = loss_w * _mag_loss
            self.log("train/mag_loss", _mag_loss, sync_dist=True, batch_size=self.batch_size)
            loss = loss + _mag_loss
        return loss
    
    def forward(self, noisy_wav, clean_wav=None, train=True, ret_mask=False):
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        data_list = []
        if self.spec_data:
            data_list.extend([noisy_spec.real, noisy_spec.imag])
        if self.mag_data:
            data_list.append(noisy_mag)
        noisy_input = torch.cat(data_list, dim=1)
        est_spec, est_mag = self.model(noisy_input)
        if est_spec is not None: # output of spec_decoder
            est_spec = torch.complex(est_spec[:,0,...], est_spec[:,1,...]).unsqueeze(1)
            est_spec = est_spec * noisy_spec if self.mask else est_spec
            if est_mag is not None: # output of mag_decoder
                est_mag = est_mag * noisy_mag if self.mask else est_mag
                if self.conf.get('noisy_phase', True):
                    noisy_phase = torch.angle(noisy_spec)
                    final_spec = (est_spec + est_mag * torch.exp(1.0j*noisy_phase))/2
                else: # don't use it !
                    final_mag = (torch.abs(est_spec) + est_mag) / 2
                    est_phase = torch.angle(torch.complex(est_spec.real.clip(EPS), 
                                                          est_spec.imag.clip(EPS))) # fix nan
                    final_spec = final_mag * torch.exp(1.0j*est_phase)
            else:
                est_mag = None
                final_spec = est_spec
        else:
            est_spec = None # don't set spec_loss !!!
            if est_mag is not None: # output of mag_decoder
                est_mag = est_mag * noisy_mag if self.mask else est_mag
                noisy_phase = torch.angle(noisy_spec) # use noisy phase
                final_spec =  est_mag * torch.exp(1.0j*noisy_phase)
            else:
                raise ValueError('both est_spec and est_mag are None')
        
        est_wav = self.stft.apply_istft(final_spec)
        if self.rms_norm is True:
            est_wav = torch_audio.normalize_with_ref(est_wav, noisy_wav)
        if not train:
            return est_wav
        
        if (not self.model.training) and self.valid_loss is not None:
            _loss = self.valid_loss(clean_wav, est_wav)
        else:
            if len(self.ext_loss) > 0:
                clean_spec = self.stft.apply_stft(clean_wav)
                clean_mag = torch.abs(clean_spec)
                _loss = self.apply_loss_fun(clean_wav, est_wav, clean_spec, est_spec,
                                            clean_mag, est_mag)
            else:
                if self.loss_name == 'wSDRLoss': # require noisy wav
                    _loss = self.loss(clean_wav, est_wav, noisy_wav)
                elif self.loss_name in ['CMSELoss']: # frequency domain loss
                    _loss = self.loss(self.stft.apply_stft(clean_wav), est_spec)
                else:
                    _loss = self.loss(clean_wav, est_wav)
        if self.check_nan:
            data_file = f"exp/debug/{self.global_step}.pt"
            if check_model_isnan(self.model.state_dict()) or torch.isnan(est_wav).any() or _loss.isnan():
                logger.info("check_nan, model: {}, est_wav: {}, loss: {}".format(
                    check_model_isnan(self.model.state_dict()), torch.isnan(est_wav).any(), _loss.isnan()))
                torch.save([_loss, noisy_wav, clean_wav, est_wav, self.model.state_dict()], data_file)
                sys.exit(0)
        if ret_mask is True:
            return _loss, est_wav, est_spec, est_mag
        return _loss, est_wav


class DualSE(BaseSE):
    # Speech enhancement with Dual Head
    def __init__(self, conf):
        super().__init__(conf)
        # the default configuration
        default_conf = {
            'complex_mask': True,
            'mag_mask': True,
            'norm': 'instance',
            'deepfilter': self.deepfilter,
            'merge': 0.5,
            'debug': self.debug
        }
        self.model_conf = merge_dicts(default_conf, self.model_conf) # merge
        self.model = self.model_class(self.model_conf)
        assert self.feature is None

    def forward(self, noisy_wav, clean_wav=None, train=True):
        # use forward for inference/predictions
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)

        noisy_input = torch.cat([noisy_spec.real, noisy_spec.imag, noisy_mag], dim=1)
        est_spec, est_mag = self.model(noisy_input)

        # complex head
        if self.model_conf['complex_mask']:
            if self.deepfilter:
                est_spec = self.model.deepFilter.apply_deepfilter(noisy_input, coeff=est_spec)
            else:
                est_spec = noisy_spec * est_spec
        est_spec = torch.complex(est_spec[:, 0, ...], est_spec[:, 1, ...]).unsqueeze(1)

        # magnitude head
        if self.model_conf['mag_mask']:
            est_mag = noisy_mag * est_mag
        est_mag_spec = est_mag * torch.exp(1.0j*noisy_phase)

        # merge
        final_spec = (1 - self.model_conf['merge']) * est_spec + \
            self.model_conf['merge'] * est_mag_spec

        est_wav = self.stft.apply_istft(final_spec)
        if train is True:
            pre_spec = final_spec.squeeze(1)
            ref_spec = self.stft.apply_stft(clean_wav).squeeze(1)
            if self.loss_name == 'HybridLoss':  # require spec
                _loss = self.loss(torch.stack([pre_spec.real, pre_spec.imag], dim=-1),
                                  torch.stack([ref_spec.real, ref_spec.imag], dim=-1),
                                  est_wav, clean_wav)
            else:
                _loss = self.loss(clean_wav, est_wav)
            return _loss, est_wav
        else:
            return est_wav


class FBankSE(BaseSE):
    # Speech enhancement with FBankNet
    def __init__(self, conf):
        super().__init__(conf)
        # the default configuration
        default_conf = {
            'debug': self.debug,
            'nn_linear': True
        }
        self.model_conf = merge_dicts(default_conf, self.model_conf)  # merge
        self.model = self.model_class(self.model_conf)
        mel_conf = {'n_mels': self.model_conf.get('n_mels', 40), 
                    'n_stft': self.stft_conf['n_fft']//2 + 1}
        if not self.feat_extractor: # if Kaldi feature extractor is not used
            self.melSpec = models.MelSpec(**mel_conf)
        self.inverseMelSpec = models.InvMelSpec(**mel_conf, 
                                                nn_linear=self.model_conf['nn_linear'])
        self.wav_loss_conf = conf['loss']['wav']
        self.mel_loss_conf = conf['loss']['mel']
        self.wav_loss_name = conf['loss']['wav']['name']
        self.mel_loss_name = conf['loss']['mel']['name']
        self.wav_loss = self.get_loss(self.wav_loss_conf)
        self.mel_loss = self.get_loss(self.mel_loss_conf)
        self.mask_spec = self.model_conf.get('mask_spec', 'mag') # mag or complex
        self.pesq_weight = 0
        if 'pesq' in self.conf['loss'] and conf['loss']['pesq']['weight'] > 0:
            self.pesq_loss = self.get_loss(conf['loss']['pesq'])
            self.pesq_weight = conf['loss']['pesq']['weight']

    def getMelSpec(self, wav, spec):
        if not self.feat_extractor:
            return self.melSpec(spec)
        else:
            feat_batch = self.feat_extractor.get_feat_batch(wav)
            return feat_batch.data.permute(0, 2, 1).unsqueeze(1)
    
    def forward(self, noisy_wav, clean_wav=None, train=True):
        # use forward for inference/predictions
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        noisy_fbank = self.getMelSpec(noisy_wav, noisy_mag)
        est_fbank = self.model(noisy_fbank)
        est_mag = self.inverseMelSpec(est_fbank)
        if self.mask_spec == 'complex':
            est_spec = est_mag * noisy_spec            
            est_wav = self.stft.apply_istft(est_spec, length=noisy_wav.shape[-1])
        else:
            if self.mask:
                est_mag = est_mag * noisy_mag
            est_wav = self._calc_istft(est_mag, torch.angle(noisy_spec), length=noisy_wav.shape[-1])
        
        if not train:
            return est_wav
        _loss = 0.0
        if self.wav_loss_conf['weight'] > 0:
            _wav_loss = 0.0
            if self.wav_loss_name == 'HybridLoss':  # require spec
                est_spec = self.stft.apply_stft(est_wav).squeeze(1)
                ref_spec = self.stft.apply_stft(clean_wav).squeeze(1)
                _wav_loss = self.wav_loss(torch.stack([est_spec.real, est_spec.imag], dim=-1),
                                          torch.stack([ref_spec.real, ref_spec.imag], dim=-1),
                                          est_wav, clean_wav)
            else:
                _wav_loss = self.loss(clean_wav, est_wav)
            _loss = _loss + self.wav_loss_conf['weight'] * _wav_loss
        if self.mel_loss_conf['weight'] > 0:
            clean_fbank = self.getMelSpec(clean_wav, self._calc_stft(clean_wav)[0])
            _loss = _loss + self.mel_loss_conf['weight'] * self.mel_loss(clean_fbank, est_fbank)
        if self.pesq_weight > 0:
            _loss = _loss + self.pesq_weight * self.pesq_loss(clean_wav, est_wav)
            if self.check_nan:
                data_file = f"exp/debug/{self.global_step}.pt"
                if check_model_isnan(self.model.state_dict()) or torch.isnan(est_wav).any() or _loss.isnan():
                    logger.info("check_nan, model: {}, est_wav: {}, loss: {}".format(
                        check_model_isnan(self.model.state_dict()), torch.isnan(est_wav).any(), _loss.isnan()))
                    torch.save([_loss, noisy_wav, clean_wav, est_wav, self.model.state_dict()], data_file)
                    sys.exit(0)
        return _loss, est_wav


class DualFBankSE(BaseSE):
    # Speech enhancement with DualFBankNet
    def __init__(self, conf):
        super().__init__(conf)        
        # the default model configuration
        model_conf = {
            'debug': self.debug,
            'merge': 0.5
        }
        self.model_conf = merge_dicts(model_conf, self.conf['model'])  # merge
        self.model = self.model_class(self.model_conf)
        mel_conf = {'n_mels': self.model_conf['n_mels'], 'n_stft': self.stft_conf['n_fft']//2 + 1}
        self.melSpec = models.MelSpec(**mel_conf)
        self.inverseMelSpec = models.InvMelSpec(
            **mel_conf, nn_linear=self.model_conf.get('mel_inv_linear', False))
        self.wav_loss_conf = conf['loss']['wav']
        self.mel_loss_conf = conf['loss']['mel']
        self.wav_loss_name = conf['loss']['wav']['name']
        self.mel_loss_name = conf['loss']['mel']['name']
        self.wav_loss = self.get_loss(self.wav_loss_conf)
        self.mel_loss = self.get_loss(self.mel_loss_conf)

    def forward(self, noisy_wav, clean_wav=None, train=True):
        # use forward for inference/predictions
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)
        if self.model_conf['in_channels'] == 3:
            noisy_data = torch.cat([noisy_mag, noisy_spec.real, noisy_spec.imag], dim=1)
            data_sign = torch.sign(noisy_data)
            noisy_fbank = self.melSpec(torch.abs(noisy_data))
            est_fbank = self.model(noisy_fbank) # ([B, 1, n_mels, T], [B, 2, n_mels, T])        
            est_data = self.inverseMelSpec(torch.cat(est_fbank, dim=1))
            est_data = data_sign * est_data
        elif self.model_conf['in_channels'] == 1:
            noisy_fbank = self.melSpec(noisy_mag)
            est_fbank = self.model(noisy_fbank) # ([B, 1, n_mels, T], [B, 2, n_mels, T])        
            est_data = self.inverseMelSpec(torch.cat(est_fbank, dim=1))
        est_mag = est_data[:, :1, :, :]  # [B, 1, n_stft, T]
        est_spec = est_data[:, 1:, :, :] # [B, 2, n_stft, T]
        
        # magnitude head
        est_mag = noisy_mag * est_mag if self.model_conf['decoder_list'][0]['mask'] else est_mag
        est_mag_spec = est_mag * torch.exp(1.0j*noisy_phase)
        
        # complex head
        est_spec = torch.complex(est_spec[:, 0, ...], est_spec[:, 1, ...]).unsqueeze(1)
        est_spec = noisy_spec * est_spec if self.model_conf['decoder_list'][1]['mask'] else est_spec
        
        # merge
        final_spec = (1 - self.model_conf['merge']) * est_spec + \
            self.model_conf['merge'] * est_mag_spec
        est_wav = self.stft.apply_istft(final_spec, length=noisy_wav.shape[-1])
        
        if not train:
            return est_wav
        _loss = 0.0
        ref_spec = self.stft.apply_stft(clean_wav).squeeze(1)
        if self.wav_loss_conf['weight'] > 0:
            _wav_loss = 0.0
            if self.wav_loss_name == 'HybridLoss':  # require spec
                est_spec = (est_mag * torch.exp(1.0j * noisy_phase)).squeeze(1)                
                _wav_loss = self.wav_loss(torch.stack([est_spec.real, est_spec.imag], dim=-1),
                                          torch.stack([ref_spec.real, ref_spec.imag], dim=-1),
                                          est_wav, clean_wav)
            else:
                _wav_loss = self.loss(clean_wav, est_wav)
            _loss = _loss + self.wav_loss_conf['weight'] * _wav_loss
        if self.mel_loss_conf['weight'] > 0:
            ref_fbank = self.melSpec(torch.cat([ref_spec.real, ref_spec.imag, torch.abs(ref_spec)], dim=1))
            _loss = _loss + self.mel_loss_conf['weight'] * self.mel_loss(ref_fbank, 
                                                                         torch.cat(est_fbank, dim=1))
        return _loss, est_wav


class MambaSE(BaseSE):
    # Speech enhancement with Mamba
    def __init__(self, conf):
        super().__init__(conf)        
        # the default model configuration
        default_conf = {
            'n_fft': self.stft_conf['n_fft'],
            'input_channel': 2, # Magnitude and Phase
            'output_channel': 1  # Single Channel Speech Enhancement
        }
        self.model_conf = merge_dicts(default_conf, self.model_conf) # merge
        self.model = self.model_class(self.model_conf)
        self.phase_weight = self.conf['loss'].get('phase_weight', 0.1)

    def forward(self, noisy_wav, clean_wav=None, **kwargs):
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)
        noisy_input = torch.cat([noisy_mag, noisy_phase], dim=1)
        
        est_mask, est_phase = self.model(noisy_input)
        est_mag = est_mask * noisy_mag
        est_spec = torch.complex(est_mag * torch.cos(est_phase), est_mag * torch.sin(est_phase))
        est_wav = self.stft.apply_istft(est_spec)        
        
        _loss = None
        if self.model.training:
            _, clean_phase = self._calc_stft(clean_wav)
            _loss = self.loss(clean_wav, est_wav, clean_phase, est_phase)
        else:
            if clean_wav is not None:
                _, clean_phase = self._calc_stft(clean_wav)
                _loss = self.loss(clean_wav, est_wav, clean_phase, est_phase)
        return _loss, est_wav    