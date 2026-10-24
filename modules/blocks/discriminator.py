from modules.stft import STFT
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
import modules.model.arch as model_arch
import modules.blocks as blocks

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, conf, p_list=[2, 3, 5]):  # [2,3,5,7,11]
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in p_list])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


# MetricGAN
def xavier_init_layer(in_size, out_size=None, spec_norm=True, layer_type=nn.Linear, **kwargs):
    "Create a layer with spectral norm, xavier uniform init and zero bias"
    if out_size is None:
        out_size = in_size
    layer = layer_type(in_size, out_size, **kwargs)
    if spec_norm:
        layer = spectral_norm(layer)
    # initialization
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)
    return layer


class MetricDiscriminator(nn.Module):
    """Metric estimator for enhancement training. Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers
    Arguments
    ---------
    kernel_size : tuple, the dimensions of the 2-d kernel used for convolution.
    base_channels : int, number of channels used in each conv layer.
    """

    def __init__(self, conf, kernel_size=(5, 5), base_channels=15, activation=nn.LeakyReLU):
        super().__init__()
        self.activation = activation(negative_slope=0.3)
        self.BN = nn.BatchNorm2d(num_features=2, momentum=0.01)
        self.conv1 = xavier_init_layer(2, base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.conv2 = xavier_init_layer(base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.conv3 = xavier_init_layer(base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.conv4 = xavier_init_layer(base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.Linear1 = xavier_init_layer(base_channels, out_size=50)
        self.Linear2 = xavier_init_layer(in_size=50, out_size=10)
        self.Linear3 = xavier_init_layer(in_size=10, out_size=1)

    def forward(self, x):
        out = self.BN(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.activation(out)
        out = self.conv4(out)
        out = self.activation(out)
        out = torch.mean(out, (2, 3))
        out = self.Linear1(out)
        out = self.activation(out)
        out = self.Linear2(out)
        out = self.activation(out)
        out = self.Linear3(out)
        return out


class MetricDiscriminatorU(nn.Module):
    """Metric estimator for enhancement training.

    Consists of:
     * four 2d conv layers
     * channel averaging
     * three linear layers

    Arguments
    ---------
    kernel_size : tuple
        The dimensions of the 2-d kernel used for convolution.
    base_channels : int
        Number of channels used in each conv layer.
    lin_dim1: int
        Dimensionality of the first linear layer.
    lin_dim2: int
        Dimensionality of the second linear layer.


    Example
    -------
    >>> inputs = torch.rand([1, 1, 100, 257])
    >>> model = MetricDiscriminator()
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([1, 1])
    """

    # FCN
    def __init__(self, conf, kernel_size=(5, 5), base_channels=15, activation=nn.LeakyReLU, 
                 lin_dim1=50, lin_dim2=10):
        super().__init__()
        d_out = conf.get('d_out', 1)
        self.activation = activation(negative_slope=0.3)
        self.BN = nn.BatchNorm2d(num_features=1, momentum=0.01)
        self.conv1 = xavier_init_layer(1, base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.conv2 = xavier_init_layer(base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.conv3 = xavier_init_layer(base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.conv4 = xavier_init_layer(base_channels, layer_type=nn.Conv2d, 
                                       kernel_size=kernel_size)
        self.Linear1 = xavier_init_layer(base_channels, out_size=lin_dim1)
        self.Linear2 = xavier_init_layer(in_size=lin_dim1, out_size=lin_dim2)
        self.Linear3 = xavier_init_layer(in_size=lin_dim2, out_size=d_out)

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        # x = self.BN(x)
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.activation(out)
        out = self.conv4(out)
        out = self.activation(out)
        out = torch.mean(out, (2, 3))
        out = self.Linear1(out)
        out = self.activation(out)
        out = self.Linear2(out)
        out = self.activation(out)
        out = self.Linear3(out)
        return out


class ResNetDiscriminator(nn.Module):
    def __init__(self, model_conf, activation=nn.LeakyReLU) -> None:
        super().__init__()
        self.activation = activation(negative_slope=0.3)
        feat_dim = model_conf.get('feat_dim', 257)
        embedding_size = model_conf.get('embedding_size', 192)
        out_size = model_conf.get('out_size', 1)
        self.random = model_conf.get('ranom', False)
        if self.random:
            feat_dim = feat_dim * 2
        self.backbone = getattr(blocks, model_conf.get('ResNet', 'Res2Net'))(
            feat_dim=feat_dim, embedding_size=embedding_size)
        self.linear = xavier_init_layer(in_size=embedding_size, out_size=out_size)
    
    def forward(self, x):
        if self.random:
            x = torch.cat((x, torch.randn_like(x)), dim=2)
        out = self.backbone(x)
        out = self.activation(out)
        out = self.linear(out)
        return out


# from OTUR
class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        self.features = nn.Sequential(

            # input is (3) x 128 x 128
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            #nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 44 x 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 32 x 32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 16 x 16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out = self.features(input)
        # state size. (512) x 2 x 2
        out = out.view(out.size(0), -1)

        # state size. (512 x 2 x 2)
        out = self.fc1(out)

        # state size. (2048)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        # out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1),
                 padding=(0, 1), is_last=False, d_norm='spectral_norm'):
        super().__init__()
        self.is_last = is_last
        norm_f = getattr(torch.nn.utils, d_norm)
        self.conv = norm_f(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.activation = nn.LeakyReLU(0.1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        # if self.is_last is False:
        x = self.activation(x)
        return x


class Discriminator_STFT(nn.Module):
    def __init__(self, conf, param, **kwargs):
        super().__init__()
        self.conf = conf
        self.stft = STFT(conf)
        self.debug = conf.get('debug', False)
        self.d_data = conf.get('d_data', 'spec')
        param[0][0] = 2 if self.d_data == 'spec' else 1
        # spectral_norm, weight_norm
        d_norm = conf.get('d_norm', 'spectral_norm')
        self.models = nn.ModuleList([Conv2DBlock(*item, d_norm=d_norm) for item in param])
        self.post_models = nn.ModuleList()
        if conf.get('d_linear', False):  # add linear layer ?
            norm_f = getattr(torch.nn.utils, d_norm) if conf.get('d_linear_norm', False) else nn.Identity()
            linear_nodes = [int(item) for item in conf.get('linear_nodes', '256,64,1').split(',')]
            self.post_models.extend([norm_f(nn.Linear(linear_nodes[0], linear_nodes[1])),
                                    #  nn.LeakyReLU(0.1),
                                     norm_f(nn.Linear(linear_nodes[1], linear_nodes[2]))])
            if conf.get('d_sigmoid', False):  # add sigmoid layer ?
                self.post_models.append(nn.Sigmoid())

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        spec = self.stft.apply_stft(x)
        if self.d_data == 'spec':  # complex spectrum
            data_list = [spec.real, spec.imag]
        else:  # magnitude spectrum
            data_list = [torch.abs(spec)]
        data = torch.cat(data_list, dim=1)
        if self.debug:
            print(f"data: {data.shape}")
        for i, layer in enumerate(self.models):
            data = layer(data)
            if self.debug:
                print(f"layer_{i}: {data.shape}")
        data = torch.flatten(data, 1, -1)
        for i, layer in enumerate(self.post_models):
            data = layer(data)
            if self.debug:
                print(f"layer_{i}: {data.shape}")
        return data


class MultiResDiscriminator(nn.Module):
    # multi-resolution stft discriminator
    info_list = (
        # stft_kwargs, param
        ({'n_fft': 2048, 'win_length': 1600, 'hop_length': 400}, 'CNN_6_128_2048'),
        ({'n_fft': 1536, 'win_length': 1200, 'hop_length': 300}, 'CNN_6_128_1536'),
        ({'n_fft': 1024, 'win_length': 800, 'hop_length': 200}, 'CNN_6_128_1024'),
        ({'n_fft': 768, 'win_length': 600, 'hop_length': 150}, 'CNN_6_128_768'),
        ({'n_fft': 512, 'win_length': 400, 'hop_length': 100}, 'CNN_6_128_512'),
        ({'n_fft': 384, 'win_length': 300, 'hop_length': 75}, 'CNN_6_128_384'),
        ({'n_fft': 256, 'win_length': 200, 'hop_length': 50}, 'CNN_6_128_256'),
        ({'n_fft': 128, 'win_length': 100, 'hop_length': 25}, 'CNN_6_128_128'))

    def __init__(self, conf, **kwargs):
        super().__init__()
        fft_list = [int(item) for item in str(conf.get('fft_list', '256,512,1024')).split(',')]
        model_list = []
        for n_fft in fft_list:
            for stft_kwargs, param in self.info_list:
                if stft_kwargs['n_fft'] == n_fft:
                    conf = deepcopy(conf)
                    conf['stft_kwargs'] = stft_kwargs
                    model_list.append(Discriminator_STFT(conf, getattr(model_arch, param)))
        self.discriminators = nn.ModuleList(model_list)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        for d in self.discriminators:
            y_d_r = d(y)
            y_d_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
        return y_d_rs, y_d_gs, None, None
