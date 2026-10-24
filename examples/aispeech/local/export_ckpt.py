import torch
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy


def export_ckpt(conf):
    asr_ckpt = Path(conf.get('asr_ckpt', '/public/home/wenbin_jiang/workspace/atec/proj/cn_fan_Midea-XiDun__YJGGZC-27436/exp/cfg_cn_Mins-pinlei_fengshanpro/train/ckpt_bak/checkpoint-12'))
    asr_model = torch.load(asr_ckpt)

    # only key list are exported, for validation
    key_list = ['model', 'hparams']
    print('asr_model:', asr_model.keys())
    export_model = {}
    for key in key_list:
        export_model[key] = asr_model[key]
    torch.save(export_model, asr_ckpt.with_name('export_v0.ckpt'))

    # export pytorch lightning ckpt
    ckpt = Path(conf.get('ckpt', 'exp/logs/LiteASR_QFSMN_R/5a0/checkpoints/last.ckpt'))
    pl_model = torch.load(ckpt)
    print('pl_model:', pl_model.keys())
    export_model = {
        'model': {},
        'hparams': deepcopy(asr_model['hparams'])}
    for key in pl_model['state_dict']:
        _, new_key = key.split('.', 1)
        export_model['model'][new_key] = pl_model['state_dict'][key]
    torch.save(export_model, asr_ckpt.with_name('export.ckpt'))


if __name__ == '__main__':
    conf = OmegaConf.create({
        'conf': 'conf/config_asr.yaml',
        'cmd': 'export_ckpt'
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    eval(conf.cmd)(conf)
