import math
import torch
import torchaudio
from torch import nn
import modules.model.arch as model_arch
from torchaudio.transforms import MelScale
from .blocks import *
# librosa.feature.melspectrogram
# librosa.feature.inverse.mel_to_stft
# torchaudio.transforms.MelSpectrogram
# torchaudio.transforms.InverseMelScale

class ComplexMelSpectrogram(torchaudio.transforms.MelSpectrogram):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['power'] == None # complex spectrogram
        
    def forward(self, waveform):
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        return torch.complex(self.mel_scale(specgram.real), self.mel_scale(specgram.imag))
    
    
class MelSpec(nn.Module):
    def __init__(self, n_mels=128, sample_rate=48000, n_fft=1024, **kwargs):
        super().__init__()
        self.mel_scale = MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_fft//2+1)
        
    def forward(self, specgram, dim=1):
        r"""
        Args:
            specgram (Tensor): Tensor of specgram of dimension (..., c, n_freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., c, ``n_mels``, time).
        """
        # Unbind the tensor along the specified dimension and apply mel_scale
        mel_spec_list = [self.mel_scale(s) for s in torch.unbind(specgram, dim=dim)]
        # Stack the processed tensors back along the same dimension
        return torch.stack(mel_spec_list, dim=dim)
    
    
class FilterNet(nn.Module):
    def __init__(self, param, **kwargs):
        super().__init__()
        norm = kwargs.get('norm', 'batch')
        inv = kwargs.get('inv', False)
        n_freqs = kwargs.get('n_fft', 1024)//2 + 1
        self.inv_linear = kwargs.get('inv_linear', False)
        self.debug = kwargs.get('debug', False)
        if inv:
            self.net = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param])
            if self.inv_linear:
                self.inv_linear = nn.Linear(n_freqs, n_freqs, bias=False)
        else:
            self.net = nn.ModuleList([ConvBlock(*item, norm=norm) for item in param])
    
    def forward(self, x):
        # x: shape of (B, C, F, T)
        for i, layer in enumerate(self.net):
            x = layer(x)
            if self.debug:
                print(f"layer_{i}: {x.shape}")
        if self.inv_linear:
            x = self.inv_linear(x.permute(0,1,3,2)).permute(0,1,3,2)
        return x
        

class FreqSample(nn.Module):
    def __init__(self, sample, org_bins, target_bins, low_freq=0, 
                 n_fft=1024, fs=48000, **kwargs):
        '''frequency sampling, down or up'''
        super().__init__()
        assert sample in ['down', 'up']
        if sample == 'down':
            assert org_bins > target_bins
        else:
            assert target_bins > org_bins
        self.low_bin = math.ceil(low_freq / (fs/n_fft))
        self._filters = nn.Parameter(torch.ones(org_bins-self.low_bin,
                                                target_bins-self.low_bin))
        if kwargs.get('debug', False):
            print(self._filters.shape)
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, x):
        """
        x: shape of [n_batch, n_channel, ..., n_freq, n_frame]
        return: shape of [n_batch, n_channel, ..., n_freqDown/n_freqUp, n_frame]
        """
        x_low = x[:, :, ..., 0:self.low_bin,:]
        x_high = x[:, :, ..., self.low_bin:,:]
        x_high_filtered = torch.einsum('bc...ft,fg->bc...gt', [x_high, self._filters])
        return torch.cat([x_low, x_high_filtered], dim=-2)


class Feature(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        conf_feature = conf['feature']
        n_feature = conf_feature['n_feature']
        filter = conf_feature['filter'] # mel, erb, bark, nn        
        inv_filter = conf_feature['inv_filter']
        inv_linear = conf_feature.get('inv_linear', False)
        debug = conf.get('debug', False)
        param = getattr(model_arch, conf_feature.get('nn_param', 'filterNet_2'))        
        if filter == 'nn':
            self._filter = FilterNet(param["filter"], debug=debug)
        elif filter == 'mel':
            kwargs = {'n_mels': n_feature, 
                      'sample_rate': conf['sample_rate'], 
                      'n_fft': conf['stft_kwargs']['n_fft']}
            self._filter = MelSpec(**kwargs)
        elif filter == 'nn_sample':
            kwargs = {'sample': 'down',
                      'org_bins': conf['stft_kwargs']['n_fft'] // 2 + 1,
                      'target_bins': n_feature,
                      'low_freq': conf_feature.get('down_low_freq', 8000),
                      'debug': debug}
            self._filter = FreqSample(**kwargs)
        else:
            raise NotImplementedError
        if inv_filter == 'nn':
            self._inv_filter = FilterNet(param["inv_filter"], inv=True, debug=debug, 
                                         inv_linear=inv_linear)
        elif inv_filter == 'nn_sample':
            kwargs = {'sample': 'up',
                      'org_bins': n_feature,
                      'target_bins': conf['stft_kwargs']['n_fft'] // 2 + 1,
                      'low_freq': conf_feature.get('up_low_freq', 0),
                      'debug': debug}
            self._inv_filter = FreqSample(**kwargs)
        else:
            raise NotImplementedError
        
    def forward(self, x):
        return self.apply_forward(x)
    
    def apply_forward(self, x):
        return self._filter(x)
    
    def apply_inverse(self, x):
        return self._inv_filter(x)
            