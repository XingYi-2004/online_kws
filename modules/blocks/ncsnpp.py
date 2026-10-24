# NCSN++ model, adapted from https://github.com/yang-song/score_sde repository
import torch.nn as nn
import functools
import torch
import numpy as np
from .ncsnpp_utils import layers, layerspp, normalization

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class NCSNpp(nn.Module):
    # 65.6M
    def __init__(self, scale_by_sigma=True, nonlinearity='swish', nf=128, ch_mult=(1, 1, 2, 2, 2, 2, 2), 
                 num_res_blocks=2, attn_resolutions=(16,), resamp_with_conv=True, conditional=True, 
                 fir=True, fir_kernel=[1, 3, 3, 1], skip_rescale=True, resblock_type='biggan', 
                 progressive='output_skip', progressive_input='input_skip', progressive_combine='sum', 
                 init_scale=0., fourier_scale=16, image_size=256, embedding_type='fourier', dropout=.0, 
                 **kwargs):
        super().__init__()
        self.act = act = get_act(nonlinearity)
        self.nf = nf
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(ch_mult)
        self.all_resolutions = [image_size // (2 ** i) for i in range(self.num_resolutions)]
        self.conditional = conditional  # noise-conditional
        self.scale_by_sigma = scale_by_sigma
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type
        self.progressive = progressive
        self.progressive_input = progressive_input
        self.embedding_type = embedding_type
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combiner = functools.partial(Combine, method=progressive_combine)
        
        self.in_channels = kwargs.get('in_channels', 4) # x.real, x.imag, y.real, y.imag
        self.out_channels = kwargs.get('out_channels', 2)
        self.output_layer = nn.Conv2d(self.in_channels, self.out_channels, 1)
        
        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=fourier_scale))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')
        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
        AttnBlock = functools.partial(layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)
        Upsample = functools.partial(layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            self.pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)
        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)
        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            self.pyramid_downsample = functools.partial(layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True)
        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act,dropout=dropout, init_scale=init_scale,
                                            skip_rescale=skip_rescale, temb_dim=nf * 4)
        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                                            dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                                            init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)
        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block
        channels = self.in_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if self.all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != self.num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if progressive_combine == 'cat':
                        in_ch *= 2
                elif progressive_input == 'residual':
                    modules.append(self.pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch
                hs_c.append(in_ch)
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if self.all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(self.pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')
            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))
        assert not hs_c
        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, t, **kwargs):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        # # Convert real and imaginary parts of (x,y) into four channel dimensions
        # x = torch.cat((x[:, [0], :, :].real, x[:, [0], :, :].imag,
        #                x[:, [1], :, :].real, x[:, [1], :, :].imag), dim=1)
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')
        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x
        # Input layer: Conv2d: 4ch -> 128ch
        hs = [modules[m_idx](x)]
        m_idx += 1
        # Down path in U-Net
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # Attention layer (optional)
                # edit: check H dim (-2) not W dim (-1)
                if h.shape[-2] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)
        h = hs[-1]  # actualy equal to: h = h
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block
        m_idx += 1
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for _ in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')
            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)  # Upspampling
                    m_idx += 1
        assert not hs
        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1
        assert m_idx == len(modules), "Implementation error"
        h = h / used_sigmas[:, None, None, None]
        
        # # Convert back to complex number
        # h = self.output_layer(h)
        # h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        # h = torch.view_as_complex(h)[:, None, :, :]
        return self.output_layer(h)
    

class NCSNppMedium(NCSNpp):
    # 27.8M
    def __init__(self, **kwargs):
        super().__init__(nf=128, ch_mult=(1, 2, 2, 2), num_res_blocks=1, attn_resolutions=(0,), **kwargs)


class NCSNppSmall(NCSNpp):
    # 11.7M
    def __init__(self, **kwargs):
        super().__init__(nf=96, ch_mult=(1, 2, 2, 1), num_res_blocks=1, attn_resolutions=(0,), **kwargs)


class NCSNppTiny(NCSNpp):
    # 5.2M
    def __init__(self, **kwargs):
        super().__init__(nf=96, ch_mult=(1, 1, 1, 1), num_res_blocks=1, attn_resolutions=(0,), **kwargs)