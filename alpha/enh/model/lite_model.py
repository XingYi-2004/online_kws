from modules.blocks.blocks import *
import modules.blocks.gtcrn as GBlock
from torchaudio.transforms import MelScale, InverseMelScale
from modules.utils.common import merge_dicts
from modules.utils.logging import logger
from modules.blocks.cbam import CBAM


class MIMOLiteNet(ModelBase):
    '''multi-input multi-output lightweight network
    number of input channels: in_channels
    number of decoder: len(decoder_list)
    '''
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'norm': 'batch', 'block_att': True, 'in_channels': -1,
            'feat_expand': 5, 'rnn': {'name': 'DPGRNN', 'num_layers': 2},
            'decoder_list': [{'mask': True, 'bound': 'tanh', "out_channels": 1}]
        }
        self.model_conf = merge_dicts(default_conf, model_conf) # merge
        logger.debug(self.model_conf)
        rnn_name = self.model_conf['rnn']['name']
        norm = self.model_conf['norm']
        param = self.model_conf['param']
        n_ConvBlock, n_GTConvBlock = param['n_ConvBlock'], param['n_GTConvBlock']         
        param['encoder'][0][0] = self.model_conf['in_channels'] * self.model_conf['feat_expand']
        
        self.sfe = GBlock.SFE(self.model_conf['feat_expand'], 1, axis=-2)        
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder'][:n_ConvBlock]] + 
            [GBlock.GTConvBlock(*item, time_last=True) for item in param['encoder'][n_ConvBlock:]])
        self.decoder_list = nn.ModuleList()
        for i in range(len(self.model_conf['decoder_list'])):
            param['decoder'][-1][1] = self.model_conf['decoder_list'][i].get('out_channels', 1)
            decoder = nn.ModuleList(
            [GBlock.GTConvBlock(*item, time_last=True) for item in param['decoder'][:n_GTConvBlock]] + 
            [TransConvBlock(*item, norm=norm) for item in param['decoder'][n_GTConvBlock:]])
            self.decoder_list.append(decoder)
        self.block_att = nn.ModuleList([CBAM(item[1], min(item[1]//2, 64)) 
                                        if self.model_conf['block_att'] is True 
                                        else nn.Identity() for item in param['encoder']])        
        
        if rnn_name == 'DPGRNN':
            rnn_kwargs = {
                'input_size': param['encoder'][-1][1],                
                'width': self.model_conf['rnn'].get('width', 20),
                'hidden_size': param['encoder'][-1][1],
                'time_last': True
            }
            logger.debug(rnn_kwargs)
            self.rnn_block = nn.Sequential(
                *[GBlock.DPGRNN(**rnn_kwargs) for _ in range(self.model_conf['rnn']['num_layers'])])
        else:
            raise NotImplementedError

    def forward(self, x):
        '''
        x: shape of (B, 1, n_feature, T)
        return: shape of [(B, 1, n_feature, T), ...]
        '''
        e = self.sfe(x)
        logger.debug(f'e: {e.shape}')
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        logger.debug(f'rnn_out: {rnn_out.shape}')
        
        n_layer = len(e_list)
        d_list = [rnn_out] * len(self.decoder_list)
        for j in range(n_layer):
            # i: decoder idx, j: layer idx
            # d_list = [self.decoder_list[i][j](_padded_cat(d, e_list[n_layer-j-1])) for i, d in enumerate(d_list)]
            d_list = [self.decoder_list[i][j](d + e_list[n_layer-j-1]) for i, d in enumerate(d_list)]
            logger.debug(','.join([f'd{i}_{j}: {d.shape}' for i, d in enumerate(d_list)]))
            
        d_list = [apply_mask_bound(d, self.model_conf['decoder_list'][i]['bound'])
                  if self.model_conf['decoder_list'][i]['mask'] else d for i, d in enumerate(d_list)]
        
        return d_list
    

class GTCRN(ModelBase):
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'mask_clamp_type':  None,
            'mask_threshold': 0.0
        }
        self.model_conf = merge_dicts(default_conf, model_conf)  # merge

        self.erb = GBlock.ERB(65, 64)
        self.sfe = GBlock.SFE(3, 1)
        self.encoder = GBlock.Encoder()
        self.dpgrnn1 = GBlock.DPGRNN(16, 33, 16)
        self.dpgrnn2 = GBlock.DPGRNN(16, 33, 16)
        self.decoder = GBlock.Decoder()
        self.mask = GBlock.Mask(self.model_conf)

    def forward(self, spec):
        '''
        spec: (B, F, T, 2)
        '''
        spec_ref = spec  # (B,257,T,2)
        spec_real = spec[..., 0].permute(0, 2, 1)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)
        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)
        feat, en_outs = self.encoder(feat) # (B,16,T,33), (B, 16, T, 65/33/33/33)
        feat = self.dpgrnn1(feat)  # (B,16,T,33)
        feat = self.dpgrnn2(feat)  # (B,16,T,33)
        m_feat = self.decoder(feat, en_outs)  # (B,2,T,129)
        m = self.erb.bs(m_feat)  # (B,2,T,257)
        spec_enh = self.mask(m, spec_ref.permute(0, 3, 2, 1))  # (B,2,T,F)
        spec_enh = spec_enh.permute(0, 3, 2, 1)  # (B,F,T,2)
        return spec_enh
    

class MelSpec(nn.Module):
    def __init__(self, n_mels=40, sample_rate=16000, n_stft=257, **kwargs):
        super().__init__()
        self.mel_scale = MelScale(n_mels=n_mels, sample_rate=sample_rate, n_stft=n_stft)
        
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


class InvMelSpec(nn.Module):
    def __init__(self, n_stft=257, n_mels=40, sample_rate=16000, **kwargs):
        super().__init__()
        self.nn_linear = kwargs.get('nn_linear', False)
        self.mel_transform = (
            nn.Linear(n_mels, n_stft, bias=False)
            if self.nn_linear else
            InverseMelScale(n_stft=n_stft, n_mels=n_mels, sample_rate=sample_rate)
        )

    def forward(self, melspec):
        r"""
        Args:
            melspec (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)
        Returns:
            Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        if self.nn_linear:
            spec = self.mel_transform(melspec.transpose(-1, -2)).transpose(-1, -2)
        else:
            spec = self.mel_transform(melspec)
        return spec

    
class FNet(ModelBase):
    # FBank Net
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'mask_bound': 'relu',
            'norm': 'batch',
            'block_att': True,
            'in_channels': 1,
            'feat_expand': 5,
            'rnn': {'name': 'DPGRNN', 'num_layers': 2}
        }        
        self.model_conf = merge_dicts(default_conf, model_conf) # merge
        logger.debug(self.model_conf)
        rnn_name = self.model_conf['rnn']['name']
        norm = self.model_conf['norm']
        param = self.model_conf['param']
        n_ConvBlock, n_GTConvBlock = param['n_ConvBlock'], param['n_GTConvBlock']         
        param['encoder'][0][0] = self.model_conf['in_channels'] * self.model_conf['feat_expand']
        
        self.sfe = GBlock.SFE(self.model_conf['feat_expand'], 1, axis=-2)        
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder'][:n_ConvBlock]] + 
            [GBlock.GTConvBlock(*item, time_last=True) for item in param['encoder'][n_ConvBlock:]])
        self.decoder = nn.ModuleList(
            [GBlock.GTConvBlock(*item, time_last=True) for item in param['decoder'][:n_GTConvBlock]] + 
            [TransConvBlock(*item, norm=norm) for item in param['decoder'][n_GTConvBlock:]])        
        self.block_att = nn.ModuleList([CBAM(item[1], min(item[1]//2, 64)) 
                                        if self.model_conf['block_att'] is True 
                                        else nn.Identity() for item in param['encoder']])        
        if rnn_name == 'DPGRNN':
            rnn_kwargs = {
                'input_size': param['encoder'][-1][1],
                'width': self.model_conf['rnn'].get('width', 20),
                'hidden_size': param['encoder'][-1][1],
                'time_last': True
            }
            self.rnn_block = nn.Sequential(
                *[GBlock.DPGRNN(**rnn_kwargs) for _ in range(self.model_conf['rnn']['num_layers'])])
        else:
            raise NotImplementedError
        if self.model_conf['mask_bound'] == 'learnable_sigmoid':
            self.learnable_sigmoid = LearnableSigmoid(self.model_conf['n_mels'])

    def forward(self, x):
        '''
        x: shape of (B, 1, n_fbank, T)
        return: shape of (B, 1, n_fbank, T), (B, 1, n_fft//2+1, T)
        '''
        e = self.sfe(x)
        logger.debug(f'e: {e.shape}')
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        logger.debug(f'rnn_out: {rnn_out.shape}')
        d = rnn_out
        for i, d_layer in enumerate(self.decoder):
            # d = d_layer(_padded_cat(d, e_list.pop()))
            d = d_layer(d + e_list.pop())
            logger.debug(f'decoder_{i}: {d.shape}')
        if self.model_conf['mask_bound'] is not None:
            if self.model_conf['mask_bound']  == 'learnable_sigmoid':
                logger.debug(f'd.shape: {d.shape}')
                d = self.learnable_sigmoid(d)
            else:
                d = apply_mask_bound(d, self.model_conf['mask_bound'])
        return d
    