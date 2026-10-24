# from https://github.com/RoyChao19477/SEMamba/tree/main
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm


def create_block(d_model, model_conf, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False):
    d_state = model_conf['d_state']  # 16
    d_conv = model_conf['d_conv']  # 4
    expand = model_conf['expand']  # 4
    norm_epsilon = model_conf['norm_epsilon']  # 0.00001

    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(d_model, mixer_cls, norm_cls=norm_cls, fused_add_norm=fused_add_norm, 
                  residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


class MambaBlock(nn.Module):
    def __init__(self, in_channels, model_conf):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks = nn.ModuleList(create_block(in_channels, model_conf) for _ in range(n_layer))
        self.backward_blocks = nn.ModuleList(create_block(in_channels, model_conf) for _ in range(n_layer))
        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None \
            else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)


class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """

    def __init__(self, model_conf):
        super(TFMambaBlock, self).__init__()
        self.hid_feature = model_conf['hid_feature']

        # Initialize Mamba blocks
        self.time_mamba = MambaBlock(self.hid_feature, model_conf)
        self.freq_mamba = MambaBlock(self.hid_feature, model_conf)

        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        """
        Forward pass of the TFMamba block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class LearnableSigmoid1D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 1D inputs.
    
    This module applies a learnable slope parameter to the sigmoid activation function.
    """

    def __init__(self, in_features, beta=1):
        """
        Initialize the LearnableSigmoid1D module.
        
        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid1D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid1D module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2D(nn.Module):
    """
    Learnable Sigmoid Activation Function for 2D inputs.
    
    This module applies a learnable slope parameter to the sigmoid activation function for 2D inputs.
    """

    def __init__(self, in_features, beta=1):
        """
        Initialize the LearnableSigmoid2D module.
        
        Args:
        - in_features (int): Number of input features.
        - beta (float, optional): Scaling factor for the sigmoid function. Defaults to 1.
        """
        super(LearnableSigmoid2D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        """
        Forward pass for the LearnableSigmoid2D module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after applying the learnable sigmoid activation.
        """
        return self.beta * torch.sigmoid(self.slope * x)


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding size for a convolutional layer.
    
    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
    
    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    """
    Calculate the padding size for a 2D convolutional layer.
    
    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).
    
    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))


class DenseBlock(nn.Module):
    """
    DenseBlock module consisting of multiple convolutional layers with dilation.
    """

    def __init__(self, hid_feature, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = hid_feature

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size,
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        """
        Forward pass for the DenseBlock module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after processing through the dense block.
        """
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """

    def __init__(self, model_conf):
        super(DenseEncoder, self).__init__()
        self.input_channel = model_conf['input_channel']
        self.hid_feature = model_conf['hid_feature']

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature))
        self.dense_block = DenseBlock(self.hid_feature, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature,
                      (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature))

    def forward(self, x):
        """
        Forward pass for the DenseEncoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Encoded tensor.
        """
        x = self.dense_conv_1(x)  # [batch, hid_feature, time, freq]
        x = self.dense_block(x)   # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x


class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """

    def __init__(self, model_conf):
        super(MagDecoder, self).__init__()
        self.hid_feature = model_conf['hid_feature']
        self.output_channel = model_conf['output_channel']
        self.n_fft = model_conf['n_fft']
        self.beta = model_conf.get('beta', 2.0)
        
        self.dense_block = DenseBlock(self.hid_feature, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1)))
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        """
        Forward pass for the MagDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with magnitude information.
        """
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)
        return x


class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """

    def __init__(self, model_conf):
        super(PhaseDecoder, self).__init__()
        self.hid_feature = model_conf['hid_feature']
        self.output_channel = model_conf['output_channel']
        
        self.dense_block = DenseBlock(self.hid_feature, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature))
        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        """
        Forward pass for the PhaseDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with phase information.
        """
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
