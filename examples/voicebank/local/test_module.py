import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from alpha.enh import model as enh_model
import modules.model.arch as model_arch
from modules.utils.common import merge_dicts
from modules.utils.logging import setup_root_logger, logger
import modules.blocks.gtcrn as GBlock

# Test and Profile------------------------------------------------------
from thop import profile, clever_format
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
from ptflops import get_model_complexity_info

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num

def init_test(conf):
    torch.manual_seed(conf['system']['seed'])
    torch.autograd.set_detect_anomaly(True)
    setup_root_logger(debug=True)

def show_model_info(model, *inputs):
    macs, params = profile(model, inputs=inputs)
    macs, params = clever_format([macs, params], "%.3f")
    print("=" * 20, f"MACs: {macs}, #Param: {params}", "=" * 20)
    # flops = FlopCountAnalysis(model, in_data)
    # print("=" * 20, "GFLOPs:", flops.total()/1e9, "=" * 20)
    # flops, params = get_model_complexity_info(model, tuple(in_data.shape[1:]), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print(flops, params)

def resolve_model_param(conf):
    model_param = conf['model'].get('param', None)
    if model_param:
        conf['model']['param'] = getattr(model_arch, model_param, None)
    return conf

def init_deepfilter(conf):
    deepfilter = conf.get('deepfilter', None)
    if deepfilter:
        deepfilter.update({'n_freq': conf['stft']['n_fft']//2 + 1})
    return deepfilter
    
def test_DNet(conf):
    conf = resolve_model_param(conf)
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    default_conf = {
        'complex_mask': True,
        'mag_mask': True,
        'norm': 'instance',
        'deepfilter': init_deepfilter(conf),
        'merge': 0.5,
        'debug': True
    }
    model_conf = merge_dicts(default_conf, model_conf) # merge
    model = enh_model.DNet(model_conf)
    in_data = torch.rand(1, 3, conf['n_freqs'], 157) # n_batch, 3, n_freqs, n_frames
    spec_out, _ = model(in_data)
    print(f"#Param: {numParams(model)}, input: {in_data.shape}, output: {spec_out.shape}")
    show_model_info(model, in_data)
    
def test_block(conf):
    # model = GBlock.GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False, time_last=True) #Param: 2162
    # model = GBlock.GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True, time_last=True) #Param: 2162
    # model = GBlock.DPGRNN(16, 33, 16, time_last=True) #Param: 4192
    # model = GBlock.SFE(5, 1) #Param: 0
    # in_data = torch.rand(1, 16, conf['n_freqs'], 157) # n_batch, n_channel, n_freqs, n_frames
    # in_data = torch.rand(1, 1, 157, conf['n_freqs']) # n_batch, n_channel, n_frames, n_freqs
    model = enh_model.InvMelSpec(nn_linear=conf.get('nn_linear', False))    
    in_data = torch.rand(1, conf['n_freqs'], conf['n_frames'], ) # n_batch, n_freqs, n_frames    
    data_out = model(in_data)
    print(f"#Param: {numParams(model)}, input: {in_data.shape}, output: {data_out.shape}")
    
def test_FNet(conf):
    conf = resolve_model_param(conf)
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    default_conf = {}
    model_conf = merge_dicts(default_conf, model_conf) # merge
    model = enh_model.FNet(model_conf)
    data_in = torch.rand(1, 1, conf['n_freqs'], conf['n_frames']) # n_batch, 3, n_freqs, n_frames
    data_out = model(data_in)
    print(f"#Param: {numParams(model)}, input: {data_in.shape}, output: {data_out.shape}")
    show_model_info(model, *(data_in, ))
    
def test_MIMONet(conf):
    conf = resolve_model_param(conf)
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    print(model_conf)
    default_conf = {'in_channels': 2}
    model_conf = merge_dicts(default_conf, model_conf) # merge
    
    model = enh_model.MIMONet(model_conf)
    data_in = torch.rand(1, model_conf['in_channels'], conf['n_freqs'], conf['n_frames']) # n_batch, in_channels, n_freqs, n_frames
    data_out = model(data_in)
    print(f"#Param: {numParams(model)}, input: {data_in.shape}, output: {data_out[0].shape}")
    show_model_info(model, *(data_in, ))

def test_LifterNet(conf):
    conf = resolve_model_param(conf)
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    print(model_conf)
    deepfilter = conf['deepfilter']
    deepfilter.update({'n_freq': conf['stft']['n_fft']//2 + 1})
    default_conf = {'in_channels': 1, 'deepfilter': deepfilter}
    model_conf = merge_dicts(default_conf, model_conf) # merge
        
    model = enh_model.LifterNet(model_conf)
    data_in = torch.rand(1, model_conf['in_channels'], conf['n_freqs'], conf['n_frames']) # n_batch, in_channels, n_freqs, n_frames
    data_out = model(data_in)
    print(f"#Param: {numParams(model)}, input: {data_in.shape}, output: {data_out[0].shape}")
    show_model_info(model, *(data_in, ))

def test_MIMOLiteNet(conf):
    conf = resolve_model_param(conf)
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    print(model_conf)
    default_conf = {'in_channels': 2}
    model_conf = merge_dicts(default_conf, model_conf) # merge
    
    model = enh_model.MIMOLiteNet(model_conf)
    data_in = torch.rand(1, model_conf['in_channels'], conf['n_freqs'], conf['n_frames']) # n_batch, in_channels, n_freqs, n_frames
    data_out = model(data_in)
    print(f"#Param: {numParams(model)}, input: {data_in.shape}, output: {data_out[0].shape}")
    show_model_info(model, *(data_in, ))

def test_MIMONetTimeEmbed(conf):
    conf = resolve_model_param(conf)
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    print(model_conf)
    default_conf = {'in_channels': 2}
    model_conf = merge_dicts(default_conf, model_conf) # merge
    
    model = enh_model.MIMONetTimeEmbed(model_conf)
    data_in = torch.rand(2, model_conf['in_channels'], conf['n_freqs'], conf['n_frames']) # n_batch, in_channels, n_freqs, n_frames
    data_out = model(data_in, torch.rand(1))
    print(f"#Param: {numParams(model)}, input: {data_in.shape}, output: {data_out[0].shape}")
    show_model_info(model, *(data_in, torch.rand(1)))

if __name__ == '__main__':
    conf = OmegaConf.create({
         "conf": "conf/config.yaml",
         "cmd": "test_DNet",
         "n_freqs": 257,
         "n_frames": 157})
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    init_test(conf)
    eval(conf.cmd)(conf)