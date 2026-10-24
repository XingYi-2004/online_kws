import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
from spikingjelly.activation_based import layer

from .conv import (
    DepthwiseConv1d,
    DepthwiseConv2d,
    MultiStepBatchNorm1d,
    PointwiseConv1d,
    Transpose,
)

class MSTASA_v_branch(nn.Module):
    """
    MSTASA captures diverse temporal dependencies from multiple perspectives with shared spiking QKV representations.
    NOTE: This is a partial implementation intended for peer review.
    Full code will be released upon paper acceptance.
    """
    def __init__(
        self,
        dim,
        config,
        num_heads: int = 8,
        init_tau: float = 2.0,
        spike_mode: str = "lif",
        attention_window: int = 20,
        layers: int = 0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.config = config
        self.dataset = self.config['dataset']
        self.use_dp: bool = self.config.get('use_dp', False)
        self.dim = dim
        self.num_heads = num_heads
        self.attention_window = attention_window
        self.layers = layers

        # Transpose helpers: [T,B,D] <-> [B,D,T]
        self.trans1 = Transpose(1, 2, 0)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(2, 0, 1, 3)
        self.local_scale = 1 / math.sqrt((dim // num_heads) * (2 * attention_window + 1))
        if self.dataset == "gsc":
            self.global_scale = 1 / math.sqrt((dim // num_heads) * (800 // self.config['hop_length']))
        else:
            self.global_scale = 1 / math.sqrt((dim // num_heads) * (1000 // self.config['time_step']))
        # Q path
        self.q_conv = nn.Conv1d(self.config['n_hidden_neurons'], 
                                self.config['n_hidden_neurons'], 
                                kernel_size=1, bias=False)
        
        self.q_bn = MultiStepBatchNorm1d(self.config['n_hidden_neurons'])
        if spike_mode == "lif":
            self.q_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, 
                store_v_seq=False, 
                backend=self.config['backend'])
        else:
            self.q_lif = ParametricLIFNode(
                init_tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, 
                store_v_seq=False, 
                backend=self.config['backend'])

        # K path
        self.k_conv = nn.Conv1d(self.config['n_hidden_neurons'], self.config['n_hidden_neurons'], kernel_size=1, bias=False)
        self.k_bn = MultiStepBatchNorm1d(self.config['n_hidden_neurons'])
        if spike_mode == "lif":
            self.k_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])
        else:
            self.k_lif = ParametricLIFNode(
                init_tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])

        # V path
        self.v_conv = nn.Conv1d(self.config['n_hidden_neurons'], self.config['n_hidden_neurons'], kernel_size=1, bias=False)
        self.v_bn = MultiStepBatchNorm1d(self.config['n_hidden_neurons'])
        if spike_mode == "lif":
            self.v_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])
        else:
            self.v_lif = ParametricLIFNode(
                init_tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])

        # Attention dropout & spiking
        if self.config['use_dp']:
            self.attn_dropout = layer.Dropout(self.config['dropout_p'], step_mode='m')

        if spike_mode == "lif":
            self.attn_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])

            self.global_attn_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])
            self.local_attn_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])

        # Output projection 1
        self.mattn_conv = nn.Conv1d(self.config['n_hidden_neurons'], self.config['n_hidden_neurons'], kernel_size=1, bias=False)
        self.mattn_bn = MultiStepBatchNorm1d(self.config['n_hidden_neurons'])
        if self.use_dp:
            self.mattn_dropout = layer.Dropout(self.config['dropout_p'], step_mode='m')
        if spike_mode == "lif":
            self.mattn_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])
        else:
            self.mattn_lif = ParametricLIFNode(
                init_tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])

        # Output projection 2
        self.proj_conv = nn.Conv1d(self.config['n_hidden_neurons'], self.config['n_hidden_neurons'], kernel_size=1, bias=False)
        self.proj_bn = MultiStepBatchNorm1d(self.config['n_hidden_neurons'])
        if self.use_dp:
            self.proj_dropout = layer.Dropout(self.config['dropout_p'], step_mode='m')
        if spike_mode == "lif":
            self.proj_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])
        else:
            self.proj_lif = ParametricLIFNode(
                init_tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])

        if self.dataset == 'shd':
            self.head_lif = LIFNode(
                tau=init_tau, 
                v_threshold=self.config['v_threshold'], 
                v_reset=self.config['v_reset'],
                surrogate_function=self.config['surrogate_function'], 
                detach_reset=self.config['detach_reset'],
                step_mode='m', 
                decay_input=False, store_v_seq=False, 
                backend=self.config['backend'])

        # V—branch
        if self.dataset == 'shd':
            self.dw_kernel = 9  #
        elif self.dataset == 'ssc':
            self.dw_kernel = 7  #
        else:
            self.dw_kernel = 9  #
        if self.dataset == 'gsc':
            self.v_dw = DepthwiseConv2d(self.num_heads, self.num_heads, kernel_size=(self.dw_kernel, 3),
                                        stride=1,
                                        padding=((self.dw_kernel - 1) // 2, 1),
                                        bias=False)  # config.use_dw_bias
        else:
            self.v_dw = DepthwiseConv2d(self.num_heads, self.num_heads, kernel_size=(self.dw_kernel, 1),
                                        stride=1,
                                        padding=((self.dw_kernel - 1) // 2, 0),
                                        bias=False)  # config.use_dw_bias
        self.v_dw_lif = LIFNode(
            tau=init_tau, 
            v_threshold=self.config['v_threshold'], 
            v_reset=self.config['v_reset'],
            surrogate_function=self.config['surrogate_function'], 
            detach_reset=self.config['detach_reset'],
            step_mode='m', 
            decay_input=False, store_v_seq=False, 
            backend=self.config['backend'])

        self.v_pw = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=(1, 1), stride=1, padding=0,
                              bias=False)

    def forward(self, x, attention_mask=None):
        """
        x: [T, B, N], attention_mask: [B, T] (bool) where True = attend, False = mask
        """

        T, B, N = x.shape
        if self.config['dataset'] == 'shd':
            x = self.head_lif(x)

        # prepare Q/K/V inputs
        x_qkv = self.trans1(x)  # [B, N, T]

        # Q
        q_out = self.q_conv(x_qkv)
        q_out = self.trans2(q_out)  # [T, B, N]
        q_out = self.q_bn(q_out)
        q = self.q_lif(q_out)  # [T, B, N]

        q = q.reshape(T, B, self.num_heads, N // self.num_heads)
        q = q.permute(1, 2, 0, 3).contiguous()  # [B, H, T, D_h]

        # K
        k_out = self.k_conv(x_qkv)
        k_out = self.trans2(k_out)  # [T, B, N]
        k_out = self.k_bn(k_out)
        k_out = self.k_lif(k_out)
        k = k_out.reshape(T, B, self.num_heads, N // self.num_heads)
        k = k.permute(1, 2, 0, 3).contiguous()  # [B,H,T,D_h]

        # V
        v_out = self.v_conv(x_qkv)
        v_out = self.trans2(v_out)  # [T, B, N]
        v_out = self.v_bn(v_out)
        v_out = self.v_lif(v_out)
        v = v_out.reshape(T, B, self.num_heads, N // self.num_heads)
        # v = v.permute(1, 2, 0, 3).contiguous()  # [B,H,T,D_h]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # (B,1,T,1)
            q = q.masked_fill(~mask, 0.0)  # 把 q 在 mask==False 的位置置 0
            k = k.masked_fill(~mask, 0.0)
            # v = v.masked_fill(~mask, 0.0)

        B, H, T, D = q.shape
        w = self.attention_window

        q_pad = F.pad(q, (0, 0, w, w))  # [B,H,T+2w,D]
        k_pad = F.pad(k, (0, 0, w, w))  # 同上

        #    -> [B, H, num_windows=T, D, window=2w+1]
        q_win = q_pad.unfold(2, 2 * w + 1, 1)
        # [B, H, num_windows=T, D, window=2w+1] => [B, H, num_windows=T, window=2w+1, D
        q_win = q_win.permute(0, 1, 2, 4, 3).contiguous()  # [B, H, T, 2w+1, D]
        k_win = k_pad.unfold(2, 2 * w + 1, 1)
        # [B, H, num_windows=T, D, window=2w+1] => [B, H, num_windows=T, window=2w+1, D]
        k_win = k_win.permute(0, 1, 2, 4, 3).contiguous()  # [B, H, T, 2w+1, D]

        q_sum = q_win.sum(dim=3)
        # permute
        q_sum = q_sum.permute(2, 0, 1, 3).contiguous()  # [T,B,H,D]

        k_sum = k_win.sum(dim=3)
        # permute
        k_sum = k_sum.permute(2, 0, 1, 3).contiguous()  # [T,B,H,D]

        gate = self.local_attn_lif((q_sum + k_sum) * self.local_scale)

        out_local = gate * v  # [B,H,T,D]

        q_sum_all = q.sum(dim=2, keepdim=True)  # (B.H,1,D)
        # permute
        q_sum_all = q_sum_all.permute(2, 0, 1, 3).contiguous()  # [1,B,H,D]

        k_sum_all = k.sum(dim=2, keepdim=True)
        # permute
        k_sum_all = k_sum_all.permute(2, 0, 1, 3).contiguous()
        gate_all = self.global_attn_lif((q_sum_all + k_sum_all) * self.global_scale)  # [B,H,1,D]
        out_global = gate_all * v  # [B,H,T,D]

        v = v.permute(1, 2, 0, 3).contiguous()  # [B,H,T,D_h]
        v = self.v_pw(v)
        v_mask = self.v_dw(v)  # (B,H,T,D) => (T,B,H,D) (2,0,1,3)
        v_mask = v_mask.permute(2, 0, 1, 3).contiguous()  # [T,B,H,D]
        v_mask = self.v_dw_lif(v_mask)

        v_mask = v_mask.reshape(T, B, self.num_heads * N // self.num_heads).contiguous()

        attn = out_local + out_global  # [T,B,H,D]

        attn = attn.reshape(T, B, H * D).contiguous()

        x = self.attn_dropout(attn)

        x = self.trans1(x)
        x = self.mattn_conv(x)
        x = self.trans2(x)
        x = self.mattn_bn(x)
        x = self.mattn_lif(x)
        x = self.mattn_dropout(x)

        x = x + v_mask

        #  projection
        x = self.trans1(x)
        x = self.proj_conv(x)
        x = self.trans2(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.proj_dropout(x)

        return x


class SCRMLP(nn.Module):
    """
    Spiking Contextual Refinement MLP (SCRMLP) module used for selective channel and temporal refinement.
    NOTE: This is a partial implementation intended for peer review.
    Complete implementation will be released upon paper acceptance.
    """
    def __init__(
        self,
        config,
        in_features,
        hidden_features=None,
        out_features=None,
        kernel_size = 31,
        spike_mode="lif",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.config = config
        self.use_dp: bool = self.config.get('use_dp', False)
        self.init_tau = self.config['init_tau']
        # time, batch, dim => batch, dim, time
        self.trans1 = Transpose(1, 2, 0)
        # batch, dim, time => time, batch, dim
        self.trans2 = Transpose(2, 0, 1)

        """  MLP """
        self.fc1_linear = layer.Linear(in_features, hidden_features, bias=False, step_mode='m')
        self.fc1_bn = MultiStepBatchNorm1d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = LIFNode(tau=self.init_tau, 
                                   v_threshold=self.config['v_threshold'], 
                                   v_reset=self.config['v_reset'],
                                   surrogate_function=self.config['surrogate_function'], 
                                   detach_reset=self.config['detach_reset'],
                                   step_mode='m', 
                                   decay_input=False, store_v_seq=False, 
                                   backend=self.config['backend'])

        elif spike_mode == "plif":
            self.fc1_lif = ParametricLIFNode(init_tau=self.init_tau, 
                                             v_threshold=self.config['v_threshold'],
                                             surrogate_function=self.config['surrogate_function'],
                                             detach_reset=self.config['detach_reset'],
                                             step_mode='m', 
                                             decay_input=False, store_v_seq=False,
                                             backend=self.config['backend'])
        if self.use_dp:
            self.fc1_dropout = layer.Dropout(self.config['dropout_p'], step_mode='m')

        self.fc2_linear = layer.Linear(hidden_features, out_features, bias=False, step_mode='m')  # hidden_features //2
        self.fc2_bn = MultiStepBatchNorm1d(out_features)

        if spike_mode == "lif":
            self.fc2_lif = LIFNode(tau=self.init_tau, 
                                   v_threshold=self.config['v_threshold'], 
                                   v_reset=self.config['v_reset'],
                                   surrogate_function=self.config['surrogate_function'], 
                                   detach_reset=self.config['detach_reset'],
                                   step_mode='m', 
                                   decay_input=False, store_v_seq=False, 
                                   backend=self.config['backend'])

        elif spike_mode == "plif":
            self.fc2_lif = ParametricLIFNode(init_tau=self.init_tau, v_threshold=self.config['v_threshold'],
                                             surrogate_function=self.config['surrogate_function'],
                                             detach_reset=self.config['detach_reset'],
                                             step_mode='m', 
                                             decay_input=False, store_v_seq=False,
                                             backend=self.config['backend'])
        if self.use_dp:
            self.fc2_dropout = layer.Dropout(self.config['dropout_p'], step_mode='m')

        """ Convolution Module """
        self.pw1 = PointwiseConv1d(in_features, in_features, stride=1, padding=0, bias=True)
        self.pw2 = PointwiseConv1d(in_features, in_features, stride=1, padding=0, bias=True)
        self.dw = DepthwiseConv1d(hidden_features // 2, hidden_features // 2, kernel_size, stride=1,
                                  padding=(kernel_size - 1) // 2,
                                  bias=self.config['use_dw_bias'])

        self.bn1 = MultiStepBatchNorm1d(in_features)
        self.bn2 = MultiStepBatchNorm1d(hidden_features // 2)
        self.bn3 = MultiStepBatchNorm1d(in_features)

        if spike_mode == "lif":
            self.lif1 = LIFNode(tau=self.init_tau, 
                                v_threshold=self.config['v_threshold'], 
                                v_reset=self.config['v_reset'],
                                surrogate_function=self.config['surrogate_function'], 
                                detach_reset=self.config['detach_reset'],
                                step_mode='m', 
                                decay_input=False, store_v_seq=False, 
                                backend=self.config['backend'])
            self.lif2 = LIFNode(tau=self.init_tau, 
                                v_threshold=self.config['v_threshold'], 
                                v_reset=self.config['v_reset'],
                                surrogate_function=self.config['surrogate_function'], 
                                detach_reset=self.config['detach_reset'],
                                step_mode='m', 
                                decay_input=False, store_v_seq=False, 
                                backend=self.config['backend'])
            self.lif3 = LIFNode(tau=self.init_tau, 
                                v_threshold=self.config['v_threshold'], 
                                v_reset=self.config['v_reset'],
                                surrogate_function=self.config['surrogate_function'], 
                                detach_reset=self.config['detach_reset'],
                                step_mode='m', 
                                decay_input=False, store_v_seq=False, 
                                backend=self.config['backend'])
        elif spike_mode == "plif":
            self.lif1 = ParametricLIFNode(init_tau=self.init_tau, 
                                          v_threshold=self.config['v_threshold'],
                                          v_reset=self.config['v_reset'],
                                          surrogate_function=self.config['surrogate_function'],
                                          detach_reset=self.config['detach_reset'],
                                          step_mode='m', 
                                          decay_input=False, store_v_seq=False, 
                                          backend=self.config['backend'])
            self.lif2 = ParametricLIFNode(init_tau=self.init_tau, 
                                          v_threshold=self.config['v_threshold'],
                                          v_reset=self.config['v_reset'],
                                          surrogate_function=self.config['surrogate_function'],
                                          detach_reset=self.config['detach_reset'],
                                          step_mode='m', 
                                          decay_input=False, store_v_seq=False, 
                                          backend=self.config['backend'])
            self.lif3 = ParametricLIFNode(init_tau=self.init_tau, 
                                          v_threshold=self.config['v_threshold'],
                                          v_reset=self.config['v_reset'],
                                          surrogate_function=self.config['surrogate_function'],
                                          detach_reset=self.config['detach_reset'],
                                          step_mode='m', 
                                          decay_input=False, store_v_seq=False, 
                                          backend=self.config['backend'])

        if self.use_dp:
            self.dropout1 = layer.Dropout(self.config['dropout_p'], step_mode='m')
            self.dropout2 = layer.Dropout(self.config['dropout_p'], step_mode='m')
            self.dropout3 = layer.Dropout(self.config['dropout_p'], step_mode='m')

    def forward(self, x):
        x = self.trans1(x)
        x = self.pw1(x)
        x = self.trans2(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.dropout1(x)

        x = self.fc1_linear(x)
        x = self.fc1_bn(x).contiguous()
        x = self.fc1_lif(x)
        x = self.fc1_dropout(x)

        # time, batch, dim
        outputs, res = x.chunk(2, dim=-1)

        outputs = self.trans1(outputs)
        outputs = self.dw(outputs)
        outputs = self.trans2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.lif2(outputs)
        outputs = self.dropout2(outputs)

        # concat output and res
        x = torch.cat((res, outputs), dim=-1)

        x = self.fc2_linear(x)
        x = self.fc2_bn(x).contiguous()
        x = self.fc2_lif(x)
        x = self.fc2_dropout(x)

        x = self.trans1(x)
        x = self.pw2(x)
        x = self.trans2(x)
        x = self.bn3(x)
        x = self.lif3(x)
        x = self.dropout3(x)

        return x


class Backbone(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        init_tau=2.0,
        spike_mode="lif",
        layers=0,
    ):
        super().__init__()
        self.config = config

        # Attention
        self.attn = MSTASA_v_branch(
            dim,
            config,
            init_tau=init_tau,
            num_heads=num_heads,
            spike_mode=spike_mode,
            attention_window=self.config['attention_window'],  # 16
            layers=layers,
        )

        # MLP
        mlp_hidden_dim = self.config['hidden_dims']
        self.scrmlp = SCRMLP(
            self.config,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            kernel_size=self.config['kernel_size'],
            spike_mode=spike_mode,
        )


    def forward(self, x,attention_mask=None):

        # Attention with residual
        attn_output = self.attn(x, attention_mask=attention_mask)
        x = x + attn_output

        # Second MLP with residual
        mlp2_output = self.scrmlp(x)
        x = x + mlp2_output

        return x
