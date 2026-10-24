import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from alpha.enh import models
import modules.model.arch as model_arch
from modules.utils.common import merge_dicts
from modules.utils.logging import setup_root_logger, logger

# Test and Profile------------------------------------------------------
from thop import profile, clever_format
# from fvcore.nn import FlopCountAnalysis, parameter_count_table
# from ptflops import get_model_complexity_info

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

def show_model_info(model, in_data):
    macs, params = profile(model, inputs=(in_data, ))
    macs, params = clever_format([macs, params], "%.3f")
    print("=" * 20, f"MACs: {macs}, #Param: {params}", "=" * 20)
    # flops = FlopCountAnalysis(model, in_data)
    # print("=" * 20, "GFLOPs:", flops.total()/1e9, "=" * 20)
    # flops, params = get_model_complexity_info(model, tuple(in_data.shape[1:]), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print(flops, params)

def resolve_model_param(conf):
    model_param = conf['model'].get('param', None)
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
    model = models.DNet(model_conf)
    in_data = torch.rand(1, 3, conf['n_freqs'], 157) # n_batch, 3, n_freqs, n_frames
    spec_out, _ = model(in_data)
    print(f"#Param: {numParams(model)}, input: {in_data.shape}, output: {spec_out.shape}")    
    show_model_info(model, in_data)

if __name__ == '__main__':
    conf = OmegaConf.create({
         "conf": "conf/config.yaml",
         "cmd": "test_DNet",
         "n_freqs": 257})
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    init_test(conf)
    eval(conf.cmd)(conf)