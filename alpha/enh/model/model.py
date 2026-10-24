import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rich import print as rprint
from modules.blocks.blocks import *
from modules.blocks.DPRNN import DPRNN
from modules.blocks.cbam import CBAM
import modules.blocks.gtcrn as GBlock
from modules.utils.logging import logger
from modules.utils.common import merge_dicts
import importlib

class DeepFilter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_inchannel = kwargs.get('n_inchannel', 2)
        self.n_filter_f = kwargs.get('n_filter_f', 3)  # frequency
        self.n_filter_t = kwargs.get('n_filter_t', 5)  # time
        self.n_filter_c = kwargs.get('n_filter_c', 2)  # channel
        self.expand = kwargs.get('expand', 'SPConv')  # SPConv, TransConv
        self.complex = kwargs.get('complex', False)
        self.method = kwargs.get('method', 'masking')  # masking, filtering
        self.reshape = kwargs.get('reshape', True)
        if self.method == 'filtering':
            assert self.reshape == True
        n_freq = kwargs.get('n_freq', 513)
        self.debug = kwargs.get('debug', False)
        if self.complex:
            raise NotImplementedError
        if self.expand == 'SPConv':
            self.expand_module = SubPixelConv2d(self.n_inchannel, self.n_filter_c,
                                                ratio=(self.n_filter_f,self.n_filter_t),
                                                reshape=self.reshape)
        elif self.expand == 'linear':
            assert self.n_filter_t == 1
            self.expand_module = LinearUpSample(n_freq, self.n_filter_f)
        else:
            raise NotImplementedError

    def forward(self, x):
        '''
        x: shape of [n_batch, n_channel, n_freq, n_frame]
        return: shape of [n_batch, n_filter_c, n_filter_f, n_filter_t, n_freq, n_frame]
                      or [n_batch, n_filter_c, n_freq, n_frame]
        '''
        return self.expand_module(x)

    def apply_deepfilter(self, x, coeff):
        '''
        x: shape of [n_batch, n_channel, n_freq, n_frame]
        coeff: shape of [n_batch, n_filter_c, n_filter_f, n_filter_t, n_freq, n_frame]
                     or [n_batch, n_filter_c, n_freq, n_frame]
        return: shape of x
        '''
        if self.method == 'masking':
            if self.reshape:
                return torch.einsum('bcft,bc...ft->bcft', [x, coeff])
            else:
                return torch.einsum('bCft,bcft->bcft', [x, coeff])
        else:
            ''' pad last dim by (self.n_filter_t-1, 0) and 
            2nd to last by (self.n_filter_f-1)//2, (self.n_filter_f-1)//2) '''
            x_pad = F.pad(x, (self.n_filter_t-1, 0, (self.n_filter_f-1)//2, (self.n_filter_f-1)//2))
            x_unfold_t = x_pad.unfold(dimension=-1, size=self.n_filter_t, step=1)  # [..., n_filter_t]
            x_unfold_t_f = x_unfold_t.unfold(dimension=-3, size=self.n_filter_f, step=1)  # [..., n_filter_f]
            return torch.einsum('bcftij,bCjift->bcft', [x_unfold_t_f, coeff])


class UNet1(torch.nn.Module):
    def __init__(self, param, in_channels=2, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        self.mask = kwargs.get('mask', True)
        self.mask_bound = kwargs.get('mask_bound', 'tanh')
        norm = kwargs.get('norm', 'instance')
        block_att = kwargs.get('block_att', True)
        deepfilter = kwargs.get('deepfilter', None)
        self.deepFilter = DeepFilter(**deepfilter) if deepfilter else nn.Identity()

        param['encoder'][0][0] = in_channels
        param['decoder'][-1][1] = in_channels
        self.encoder = nn.ModuleList([ConvBlock(*item, norm=norm) for item in param['encoder']])
        self.decoder = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param['decoder']])
        self.block_att = nn.ModuleList([
            CBAM(item[1]) if block_att is True else nn.Identity() for item in param['encoder']])

        rnn_kwargs = {
            'encoder_dim': param['encoder'][-1][1],
            'num_layers': kwargs.get('rnn_layers', 4),
            'bidirectional': kwargs.get('bidirectional', False)
        }
        self.rnn_block = DPRNN(**rnn_kwargs, hidden_size=kwargs.get('hidden_size', 128))

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            if self.debug:
                logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        if self.debug:
            logger.debug(f'rnn_out: {rnn_out.shape}')
        idx = len(e_list)
        d = rnn_out
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            # torch.cat((d, e_list[idx]), 1))
            d = layer(padded_cat(d, e_list[idx]))
            if self.debug:
                logger.debug(f'decoder_{i}: {d.shape}')
        if self.mask is True:
            d = apply_mask_bound(d, self.mask_bound)
            d = self.deepFilter(d)
        return d


class CRN(nn.Module):
    '''
    Input: [batch size, channels=1, n_fft//2+1, T]
    Output: [batch size, n_fft//2+1, T]
    '''

    def __init__(self, param, **kwargs):
        super(CRN, self).__init__()
        self.param = param
        self.mask = kwargs.get('mask', True)
        self.debug = kwargs.get('debug', False)
        self.mask_bound = kwargs.get('mask_bound', 'sigmoid')
        n_freqs = kwargs.get('n_freqs', 257)
        norm = kwargs.get('norm', 'weight')
        DPRNN_bi = kwargs.get('DPRNN_bi', True)
        lstm_block = kwargs.get('lstm_block', 'LSTM_FC_Block')
        if lstm_block == 'LSTM_FC_Block':
            encoded_shape = (param['encoder'][-1][1], get_encoded_freqs(n_freqs, param['encoder']))
            if self.debug:
                logger.debug(f'encoded_shape: {encoded_shape}')
            self.lstm_block = LSTM_FC_Block(np.prod(encoded_shape))
        elif lstm_block == 'LSTM_Block':
            self.lstm_block = LSTM_Block(*param['lstm'])
        elif lstm_block == 'DPRNN':
            n_channel = param['encoder'][-1][1]
            self.lstm_block = DPRNN(encoder_dim=n_channel, hidden_size=128, output_size=n_channel,
                                    num_layers=2, bidirectional=DPRNN_bi)
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder']])
        self.decoder = nn.ModuleList(
            [TransConvBlock(*item, norm=norm) for item in param['decoder']])

    def forward(self, x):
        if self.debug:
            logger.debug(f'input x: {x.shape}')
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e_list.append(e)
            if self.debug:
                logger.debug(f'encoder_{i}: {e.shape}')

        lstm_out = self.lstm_block(e)

        idx = len(e_list)
        d = lstm_out
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            # d = layer(torch.cat((d, e_list[idx]), 1))
            d = layer(padded_cat(d, e_list[idx]))
            if self.debug:
                logger.debug(f'decoder_{i}: {d.shape}')

        if self.mask is True:
            d = apply_mask_bound(d, self.mask_bound)
            d = x * d

        return d


class DCCRN(nn.Module):
    '''Deep complex convolutional recurrent neural network'''

    def __init__(self, param, **kwargs):
        super(DCCRN, self).__init__()
        self.param = param
        self.mask = kwargs.get('mask', True)
        self.mask_bound = kwargs.get('mask_bound', 'mag_tanh')
        self.debug = kwargs.get('debug', False)
        n_freqs = kwargs.get('n_freqs', 257)
        init = kwargs.get('init', True)
        freq_prod = np.prod([enc_stride for _, _, _, enc_stride, _ in param['encoder']], axis=0)[0]
        encoded_shape = (param['encoder'][-1][1], int(np.floor(n_freqs / freq_prod)))
        if self.debug:
            logger.debug(freq_prod, encoded_shape)
        self.lstm_block = ComplexLSTM_Block(np.prod(encoded_shape))
        self.encoder = nn.ModuleList([ComplexConvBlock(*item, init=init) for item in param['encoder']])
        self.decoder = nn.ModuleList([ComplexTransConvBlock(*item, init=init) for item in param['decoder']])

    def forward(self, x):
        if self.debug:
            logger.debug(f'input x: {x.shape}')
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e_list.append(e)
            if self.debug:
                logger.debug(f'encoder_{i}: {e.shape}')

        lstm_out = self.lstm_block(e)

        idx = len(e_list)
        d = lstm_out
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            # torch.cat((d, e_list[idx]), 1))
            d = layer(padded_cat(d, e_list[idx]))
            if self.debug:
                logger.debug(f'decoder_{i}: {d.shape}')
        # pad constant zero will cause tanh nan! When n_pad < 0, it will truncate the data
        d = F.pad(d, (0, x.shape[-1] - d.shape[-1], 0, 0), mode='reflect')

        if self.mask is True:
            d = apply_mask_bound(d, self.mask_bound)
            d = x * d

        return d


class DenseCRN(torch.nn.Module):
    '''
    Input: [batch size, 3, n_fft//2+1, T]
    Output: [batch size, 1, n_fft//2+1, T], [batch size, 2, n_fft//2+1, T]
    '''

    def __init__(self, param=None, in_channels=3, n_features=64, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        norm = kwargs.get('norm', 'instance')
        n_freqs = kwargs.get('n_freqs', 256)
        rnn_block = kwargs.get('rnn_block', 'DPRNN')
        num_layers = kwargs.get('rnn_layers', 4)
        self.mag_decoder = kwargs.get('mag_decoder', True)
        self.spec_decoder = kwargs.get('spec_decoder', True)

        self.encoder = nn.ModuleList([
            ConvBlock(in_channels, n_features, kernel_size=1, stride=1,
                      padding=0, causal=False, norm=norm, n_freqs=n_freqs),
            DenseBlock(n_features, debug=self.debug,
                       norm=norm, n_freqs=n_freqs),
            ConvBlock(n_features, n_features, kernel_size=(3, 2),
                      stride=(2, 1), padding=(1, 1), norm=norm, n_freqs=n_freqs//2)])
        if rnn_block == 'DPRNN':
            self.rnn_block = DPRNN(encoder_dim=n_features, hidden_size=128,
                                   output_size=n_features, num_layers=2)
        elif rnn_block == 'DPConformer':
            self.rnn_block = DPConformer(encoder_dim=n_features, num_layers=num_layers)
        elif rnn_block == 'DPTransformer':
            self.rnn_block = DPTransformer(encoder_dim=n_features, num_layers=num_layers)

        if self.spec_decoder:
            self.spec_decoder = nn.ModuleList([
                DenseBlock(n_features, debug=self.debug, norm=norm, n_freqs=n_freqs//2),
                SubPixelTransConvBlock(n_features, n_features, norm=norm, n_freqs=n_freqs),
                nn.Conv2d(n_features, 2, kernel_size=1),
                MaskBound('tanh')])

        if self.mag_decoder:
            self.mag_decoder = nn.ModuleList([
                DenseBlock(n_features, debug=self.debug, norm=norm, n_freqs=n_freqs//2),
                SubPixelTransConvBlock(n_features, n_features, norm=norm, n_freqs=n_freqs),
                nn.Conv2d(n_features, 1, kernel_size=1),
                MaskBound('sigmoid')])

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x[..., :-1, :]  # Remove Nyquist frequency bin
        for i, layer in enumerate(self.encoder):
            e = layer(e)
        rnn_out = self.rnn_block(e)
        spec_out, mag_out = None, None
        if self.spec_decoder:
            spec_out = rnn_out
            for i in range(len(self.spec_decoder)):
                spec_out = self.spec_decoder[i](spec_out)
            spec_out = torch.nn.functional.pad(spec_out, [0, 0, 0, 1])  # Pad Nyquist frequency bin
        if self.mag_decoder:
            mag_out = rnn_out
            for i in range(len(self.mag_decoder)):
                mag_out = self.mag_decoder[i](mag_out)
            mag_out = torch.nn.functional.pad(mag_out, [0, 0, 0, 1])  # Pad Nyquist frequency bin
        return spec_out, mag_out


class MIMONet(ModelBase):
    '''multi-input multi-output network
    number of input channels: in_channels
    number of decoder: len(decoder_list)
    '''
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'debug': False, 'norm': 'batch', 'block_att': True, 'in_channels': -1,
            'rnn': {'name': 'DPRNN', 'num_layers': 2, 'bidirectional': False, 'hidden_size': 128},
            'decoder_list': [{'mask': True, 'bound': 'tanh', "out_channels": 1}]
        }
        self.model_conf = merge_dicts(default_conf, model_conf) # merge
        assert len(self.model_conf['decoder_list']) > 0
        param = self.model_conf['param']
        norm = self.model_conf['norm']
        param['encoder'][0][0] = self.model_conf['in_channels']        
        
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder']])
        self.decoder_list = nn.ModuleList()
        for i in range(len(self.model_conf['decoder_list'])):
            param['decoder'][-1][1] = self.model_conf['decoder_list'][i].get('out_channels', 1)
            decoder = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param['decoder']])
            self.decoder_list.append(decoder)
                         
        self.block_att = nn.ModuleList([
            CBAM(item[1]) if self.model_conf['block_att']  is True else nn.Identity() for item in param['encoder']])

        rnn_name = self.model_conf['rnn']['name']
        rnn_kwargs = {
            'encoder_dim': param['encoder'][-1][1],
            'num_layers': self.model_conf['rnn']['num_layers'],
            'bidirectional': self.model_conf['rnn']['bidirectional']
        }
        if rnn_name == 'DPRNN':
            self.rnn_block = DPRNN(**rnn_kwargs, hidden_size=self.model_conf['rnn']['hidden_size'])
        elif rnn_name == 'DPTransformer':
            self.rnn_block = DPTransformer(**rnn_kwargs)
        elif rnn_name == 'DPConformer':
            self.rnn_block = DPConformer(**rnn_kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
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
            d_list = [self.decoder_list[i][j](padded_cat(d, e_list[n_layer-j-1])) for i, d in enumerate(d_list)]
            logger.debug(','.join([f'd{i}_{j}: {d.shape}' for i, d in enumerate(d_list)]))
                
        d_list = [apply_mask_bound(d, self.model_conf['decoder_list'][i]['bound'])
                  if self.model_conf['decoder_list'][i]['mask'] else d for i, d in enumerate(d_list)]
        
        return d_list
    

class MIMONetTimeEmbed(ModelBase):
    '''multi-input multi-output network with time embedding
    number of input channels: in_channels
    number of decoder: len(decoder_list)
    '''
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'debug': False, 'norm': 'batch', 'block_att': True, 'in_channels': -1, 
            'time_emb_dim': 512, 'act_fn': 'PReLU', 
            'rnn': {'name': 'DPTransformerTimeEmbed', 'num_layers': 4},
            'decoder_list': [{"out_channels": 2}]
        }
        self.model_conf = merge_dicts(default_conf, model_conf) # merge
        assert len(self.model_conf['decoder_list']) > 0
        param = self.model_conf['param']
        norm = self.model_conf['norm']
        param['encoder'][0][0] = self.model_conf['in_channels']
        act_fn = self.model_conf['act_fn']
        time_emb_dim = self.model_conf['time_emb_dim']
        
        self.time_embed = TimestepEmbedding(time_emb_dim)
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm, act_fn=act_fn, time_emb_dim=time_emb_dim) for item in param['encoder']])
        self.decoder_list = nn.ModuleList()
        for i in range(len(self.model_conf['decoder_list'])):
            param['decoder'][-1][1] = self.model_conf['decoder_list'][i].get('out_channels', 1)
            decoder = nn.ModuleList([TransConvBlock(
                *item, norm=norm, act_fn=act_fn, time_emb_dim=time_emb_dim) for item in param['decoder']])
            self.decoder_list.append(decoder)
                         
        self.block_att = nn.ModuleList([
            CBAM(item[1]) if self.model_conf['block_att']  is True else nn.Identity() for item in param['encoder']])

        rnn_name = self.model_conf['rnn'].pop('name')
        self.model_conf['rnn'].update({
            'encoder_dim': param['encoder'][-1][1]
        })
        if rnn_name == 'DPTransformerTimeEmbed':
            self.rnn_block = DPTransformerTimeEmbed(**self.model_conf['rnn'])
        else:
            raise NotImplementedError

    def forward(self, x, t):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        time_emb = self.time_embed(t)
        logger.debug(f'time_emb: {time_emb.shape}')
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e, time_emb)
            e = self.block_att[i](e)
            e_list.append(e)
            logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e, time_emb)
        logger.debug(f'rnn_out: {rnn_out.shape}')
        
        n_layer = len(e_list)
        d_list = [rnn_out] * len(self.decoder_list)
        for j in range(n_layer):
            # i: decoder idx, j: layer idx
            d_list = [self.decoder_list[i][j](padded_cat(d, e_list[n_layer-j-1]), 
                                              time_emb) for i, d in enumerate(d_list)]
            logger.debug(','.join([f'd{i}_{j}: {d.shape}' for i, d in enumerate(d_list)]))
        
        return d_list


class LifterNet(MIMONet):
    def __init__(self, model_conf):
        super().__init__(model_conf)
        self.n_decoer = len(self.model_conf['decoder_list'])
        assert self.n_decoer in [1, 2]
        
    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        lifter_coff = super().forward(x)
        if self.n_decoer == 1:
            vocal = x * lifter_coff[0]
            excit = x * (1 - lifter_coff[0])
        else:
            vocal = x * lifter_coff[0]
            excit = x * lifter_coff[1]
        return vocal, excit


class UNet(ModelBase):
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'debug': False, 'mask': True, 'mask_bound': 'tanh', 'mag_decoder': False,
            'spec_decoder': False, 'norm': 'batch', 'block_att': True, 'in_channels': 3,
            'rnn': {'name': 'DPRNN', 'num_layers': 2, 'bidirectional': False, 'hidden_size': 128}
        }
        self.model_conf = merge_dicts(default_conf, model_conf) # merge
        
        assert (self.model_conf['mag_decoder'] ^ self.model_conf['spec_decoder'])
        param = self.model_conf['param']
        norm = self.model_conf['norm']
        self.mask_bound = self.model_conf['mask_bound']

        param['encoder'][0][0] = self.model_conf['in_channels']
        param['decoder'][-1][1] = self.model_conf['in_channels']
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder']])
        self.decoder = nn.ModuleList(
            [TransConvBlock(*item, norm=norm) for item in param['decoder']])
        self.block_att = nn.ModuleList([
            CBAM(item[1]) if self.model_conf['block_att']  is True else nn.Identity() for item in param['encoder']])

        rnn_name = self.model_conf['rnn']['name']
        rnn_kwargs = {
            'encoder_dim': param['encoder'][-1][1],
            'num_layers': self.model_conf['rnn']['num_layers'],
            'bidirectional': self.model_conf['rnn']['bidirectional']
        }
        if rnn_name == 'DPRNN':
            self.rnn_block = DPRNN(**rnn_kwargs, hidden_size=self.model_conf['rnn']['hidden_size'])
        elif rnn_name == 'DPTransformer':
            self.rnn_block = DPTransformer(**rnn_kwargs)
        elif rnn_name == 'DPConformer':
            self.rnn_block = DPConformer(**rnn_kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            if self.model_conf['debug']:
                logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        if self.model_conf['debug']:
            logger.debug(f'rnn_out: {rnn_out.shape}')
        idx = len(e_list)
        d = rnn_out
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            # torch.cat((d, e_list[idx]), 1))
            d = layer(padded_cat(d, e_list[idx]))
            if self.model_conf['debug']:
                logger.debug(f'decoder_{i}: {d.shape}')
        if self.model_conf['mask']:
            d = apply_mask_bound(d, self.mask_bound)
        if self.model_conf['spec_decoder']:
            return d, None
        else:
            return None, d


class TNet(torch.nn.Module):
    # Triple UNet
    def __init__(self, param, in_channels=4, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        self.mask = kwargs.get('mask', True)
        self.homo_mask = kwargs.get('homo_mask', True)
        self.mask_bound = kwargs.get(
            'mask_bound', 'tanh,sigmoid,sigmoid').split(',')
        norm = kwargs.get('norm', 'batch')
        block_att = kwargs.get('block_att', True)
        self.homo_w = kwargs.get('homo_w', 0.5)

        param['encoder'][0][0] = in_channels
        self.encoder = nn.ModuleList([ConvBlock(*item, norm=norm) for item in param['encoder']])
        if self.homo_w == 1:
            self.decoder1 = nn.ModuleList([nn.Identity() for _ in param['decoder']])
        else:
            param['decoder'][-1][1] = 2
            self.decoder1 = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param['decoder']])
        if self.homo_w == 0:
            self.decoder2 = nn.ModuleList([nn.Identity() for _ in param['decoder']])
            self.decoder3 = nn.ModuleList([nn.Identity() for _ in param['decoder']])
        else:
            param['decoder'][-1][1] = 1
            self.decoder2 = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param['decoder']])
            self.decoder3 = nn.ModuleList([TransConvBlock(*item, norm=norm) for item in param['decoder']])
        self.block_att = nn.ModuleList([
            CBAM(item[1]) if block_att is True else nn.Identity() for item in param['encoder']])

        rnn_block = kwargs.get('rnn_block', 'DPRNN')
        rnn_kwargs = {
            'encoder_dim': param['encoder'][-1][1],
            'num_layers': kwargs.get('rnn_layers', 4),
            'bidirectional': kwargs.get('bidirectional', False)
        }
        if rnn_block == 'DPRNN':
            self.rnn_block = DPRNN(**rnn_kwargs, hidden_size=kwargs.get('hidden_size', 128))
        elif rnn_block == 'DPTransformer':
            self.rnn_block = DPTransformer(**rnn_kwargs,)
        elif rnn_block == 'DPConformer':
            self.rnn_block = DPConformer(**rnn_kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            if self.debug:
                logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        idx = len(e_list)
        d1 = d2 = d3 = rnn_out
        if self.debug:
            logger.debug(f'rnn_out: {rnn_out.shape}')
        for i, (d1_layer, d2_layer, d3_layer) in enumerate(zip(self.decoder1, self.decoder2, self.decoder3)):
            idx = idx - 1
            d1 = d1_layer(padded_cat(d1, e_list[idx]))
            d2 = d2_layer(padded_cat(d2, e_list[idx]))
            d3 = d3_layer(padded_cat(d3, e_list[idx]))
            if self.debug:
                logger.debug(f'd1_{i}: {d1.shape}, d2_{i}: {d2.shape}, d3_{i}: {d3.shape}')
        if self.mask is True:
            d1 = apply_mask_bound(d1, self.mask_bound[0])
        if self.homo_mask is True:
            d2 = apply_mask_bound(d2, self.mask_bound[1])
            d3 = apply_mask_bound(d3, self.mask_bound[2])
        if self.homo_w == 1:
            d1 = None
        elif self.homo_w == 0:
            d2 = d3 = None
        return d1, d2, d3


class DNet(ModelBase):
    # Dual UNet
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'debug': False, 'complex_mask': True, 'mag_mask': True, 'mask_bound': 'tanh,sigmoid',
            'norm': 'instance', 'block_att': True, 'deepfilter': None, 'in_channels': 3,
            'rnn': {'name': 'DPRNN', 'num_layers': 2, 'bidirectional': False, 'hidden_size': 128}
        }
        self.model_conf = merge_dicts(default_conf, model_conf) # merge
        self.debug = self.model_conf.get('debug', False)
        param = self.model_conf['param']
        norm = self.model_conf['norm']
        self.mask_bound = self.model_conf['mask_bound'].split(',')
        self.deepFilter = DeepFilter(
            **self.model_conf['deepfilter']) if self.model_conf['deepfilter'] else nn.Identity()

        param['encoder'][0][0] = self.model_conf['in_channels']
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder']])
        param['decoder'][-1][1] = 2  # complex_decoder
        self.decoder1 = nn.ModuleList(
            [TransConvBlock(*item, norm=norm) for item in param['decoder']])
        param['decoder'][-1][1] = 1  # mag_decoder
        self.decoder2 = nn.ModuleList(
            [TransConvBlock(*item, norm=norm) for item in param['decoder']])
        self.block_att = nn.ModuleList([CBAM(item[1], min(item[1]//2, 64)) 
                                        if self.model_conf['block_att'] is True 
                                        else nn.Identity() for item in param['encoder']])

        rnn_name = self.model_conf['rnn']['name']
        if rnn_name == 'DPRNN':
            rnn_kwargs = {
                'encoder_dim': param['encoder'][-1][1],
                'num_layers': self.model_conf['rnn']['num_layers'],
                'bidirectional': self.model_conf['rnn']['bidirectional'],
                'hidden_size': self.model_conf['rnn']['hidden_size']
            }
            self.rnn_block = DPRNN(**rnn_kwargs)
        elif rnn_name == 'DPGRNN':
            rnn_kwargs = {
                'input_size': param['encoder'][-1][1],
                'width': param['encoder'][-1][1],
                'hidden_size': param['encoder'][-1][1],
                'time_last': True
            }
            self.rnn_block = nn.Sequential(
                *[GBlock.DPGRNN(**rnn_kwargs) for _ in range(self.model_conf['rnn']['num_layers'])])
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            if self.debug:
                rprint(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        idx = len(e_list)
        d1 = d2 = rnn_out
        if self.debug:
            rprint(f'rnn_out: {rnn_out.shape}')
        for i, (d1_layer, d2_layer) in enumerate(zip(self.decoder1, self.decoder2)):
            idx = idx - 1
            d1 = d1_layer(padded_cat(d1, e_list[idx]))
            d2 = d2_layer(padded_cat(d2, e_list[idx]))
            if self.debug:
                rprint(f'd1_{i}: {d1.shape}, d2_{i}: {d2.shape}')
        if self.model_conf['complex_mask']:
            d1 = apply_mask_bound(d1, self.mask_bound[0])
            d1 = self.deepFilter(d1)
        if self.model_conf['mag_mask']:
            d2 = apply_mask_bound(d2, self.mask_bound[1])
        return d1, d2


class MergeNet(nn.Module):
    def __init__(self, param, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        norm = kwargs.get('norm', 'batch')
        causal = kwargs.get('causal', True)
        # param = [
        #     # (in_channels, out_channels, kernel_size, stride, padding)
        #     (3, 3, (5, 3), (1, 1), (2, 1)),
        #     (3, 3, (5, 3), (1, 1), (2, 1)),
        #     (3, 1, (5, 3), (1, 1), (2, 1))]
        # self.net = nn.ModuleList([
        #     ConvBlock(3, 6, (5, 2), (1, 1), (2, 1), causal=drop_t, norm=norm),
        #     ConvBlock(6, 3, (5, 2), (1, 1), (2, 1), causal=drop_t, norm=norm),
        #     ConvBlock(3, 1, (5, 2), (1, 1), (2, 1), causal=drop_t, norm=norm)])
        self.net = nn.ModuleList([ConvBlock(*item, causal=causal, norm=norm) for item in param])

    def forward(self, x):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if self.debug:
                logger.debug(f'layer_{i}: {x.shape}')
        return x


class DCUNet(nn.Module):
    '''Deep Complex U_Net'''

    def __init__(self, param, **kwargs):
        super(DCUNet, self).__init__()
        self.param = param
        self.mask = kwargs.get('mask', True)
        self.mask_bound = kwargs.get('mask_bound', 'mag_tanh')
        self.debug = kwargs.get('debug', False)
        self.encoder = nn.ModuleList(
            [ComplexConvBlock(*item, causal=False) for item in param['encoder']])
        if 'd_mode' in param and param['d_mode'] == 'up':
            _dcode_Block = ComplexConvBlock
        else:
            _dcode_Block = ComplexTransConvBlock
        self.decoder = nn.ModuleList([_dcode_Block(*item, causal=False) for item in param['decoder']])

    def forward(self, x):
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e_list.append(e)
            if self.debug:
                logger.debug(f'encoder_{i}: {e.shape}')

        idx = len(e_list)
        d = e
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            data_in = d if self.param['shortPath'][i] is False else padded_cat(d, e_list[idx])
            d = layer(data_in)
            if self.debug:
                logger.debug(f'decoder_{i}: {d.shape}')
        # pad constant zero will cause tanh nan! When n_pad < 0, it will truncate the data
        d = F.pad(d, (0, x.shape[3] - d.shape[3], 0,
                  x.shape[2] - d.shape[2]), mode='reflect')

        if self.mask is True:
            d = apply_mask_bound(d, self.mask_bound)
            d = x * d
        return d


class YNet(nn.Module):
    def __init__(self, param, in_channels=2, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        self.mask = kwargs.get('mask', True)
        self.mask_bound = kwargs.get('mask_bound', 'tanh')
        rnn_block = kwargs.get('rnn_block', 'DPRNN')
        norm = kwargs.get('norm', 'batch')
        block_att = kwargs.get('block_att', True)

        param['encoder'][0][0] = in_channels
        param['decoder'][-1][1] = in_channels
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder']])
        self.decoder1 = nn.ModuleList(
            [TransConvBlock(*item, norm=norm) for item in param['decoder']])
        self.decoder2 = nn.ModuleList(
            [TransConvBlock(*item, norm=norm) for item in param['decoder']])
        self.block_att = nn.ModuleList([
            CBAM(item[1]) if block_att is True else nn.Identity() for item in param['encoder']])

        rnn_kwargs = {
            'encoder_dim': param['encoder'][-1][1],
            'num_layers': kwargs.get('rnn_layers', 4),
            'bidirectional': kwargs.get('bidirectional', True)
        }
        if rnn_block == 'DPRNN':
            self.rnn_block = DPRNN(
                **rnn_kwargs, hidden_size=kwargs.get('hidden_size', 128))
        elif rnn_block == 'DPTransformer':
            self.rnn_block = DPTransformer(**rnn_kwargs,)
        elif rnn_block == 'DPConformer':
            self.rnn_block = DPConformer(**rnn_kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            if self.debug:
                logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        if self.debug:
            logger.debug(f'rnn_out: {rnn_out.shape}')
        idx = len(e_list)
        d1, d2 = rnn_out, rnn_out
        for i, (d1_layer, d2_layer) in enumerate(zip(self.decoder1, self.decoder2)):
            idx = idx - 1
            d1 = d1_layer(padded_cat(d1, e_list[idx]))
            d2 = d2_layer(padded_cat(d2, e_list[idx]))
            if self.debug:
                logger.debug(f'd1_{i}: {d1.shape}, d2_{i}: {d2.shape}')
        if self.mask is True:
            d1 = apply_mask_bound(d1, self.mask_bound)
            d2 = apply_mask_bound(d2, self.mask_bound)
        return d1, d2


class Self2Self(nn.Module):
    def __init__(self, param, debug=False):
        super(Self2Self, self).__init__()
        self.param = param
        self.debug = debug
        self.encoder = nn.ModuleList(
            [Self_EncodeBlock(*item) for item in param['encoder']])
        self.decoder = nn.ModuleList(
            [Self_DecodeBlock(*item) for item in param['decoder']])

    def forward(self, x, mask):
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e, mask = layer(e, mask)
            e_list.append(e)
            if self.debug:
                logger.debug(f'encoder_{i}: {e.shape}')

        idx = len(e_list)
        d = e
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            data_in = d if self.param['shortPath'][i] is False else padded_cat(d, e_list[idx])
            d = layer(data_in)
            if self.debug:
                logger.debug(f'decoder_{i}: {d.shape}')
        # pad constant zero will cause tanh nan! When n_pad < 0, it will truncate the data
        # d = F.pad(d, (0, x.shape[3] - d.shape[3], 0, x.shape[2] - d.shape[2]), mode='reflect')
        return d


class MambaNet(nn.Module):
    """
    https://github.com/RoyChao19477/SEMamba/ 
    Speech Enhancement using Mamba blocks.
    
    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, model_conf):
        super().__init__()
        mamba_modules = importlib.import_module("modules.blocks.mamba_modules")
        self.model_conf = model_conf
        num_tscblocks = model_conf.get('num_tfmamba', 4)
        self.dense_encoder = mamba_modules.DenseEncoder(model_conf)
        self.TSMamba = nn.ModuleList([mamba_modules.TFMambaBlock(model_conf) for _ in range(num_tscblocks)])
        self.mag_decoder = mamba_modules.MagDecoder(model_conf)
        self.phase_decoder = mamba_modules.PhaseDecoder(model_conf)

    def forward(self, x):
        """
        x: shape of [B, C, n_fft//2+1, T]
        Return: shape of x
        """
        x = self.dense_encoder(x.transpose(-1, -2)) # x: [B, 2, F, T] -> [B, 2, T, F]
        
        for block in self.TSMamba:
            x = block(x)
            
        mag = self.mag_decoder(x)
        phase = self.phase_decoder(x)
        
        # [B, 1, T, F] -> [B, 1, F, T]
        return mag.transpose(-1, -2), phase.transpose(-1, -2)