import torch
from omegaconf import OmegaConf
from matcha.models.components.flow_matching import CFM
from thop import profile, clever_format

def show_model_info(model, inputs):
    assert type(inputs) is tuple
    macs, params = profile(model, inputs=inputs)
    macs, params = clever_format([macs, params], "%.3f")
    print("=" * 20, f"MACs: {macs}, #Param: {params}", "=" * 20)
    
def test(conf):
    cfm_params = OmegaConf.create({"name": "CFM", "solver": "euler", "sigma_min": 1e-4})
    decoder_params = {
        "channels": [256, 256],
        "dropout": 0.05,
        "attention_head_dim": 64,
        "n_blocks": 1,
        "num_mid_blocks": 2,
        "num_heads": 2,
        "act_fn": "snakebeta",
    }

    model = CFM(
        in_channels=2*conf.n_freqs,
        out_channel=conf.n_freqs,
        cfm_params=cfm_params,
        decoder_params=decoder_params, 
        n_spks=0, spk_emb_dim=None)
    data_in = torch.rand(1, conf['n_freqs'], conf['n_frames'])
    mask = torch.ones((1, 1, conf['n_frames']))
    data_out = model(data_in, mask, 10)
    print(data_out.shape)
    show_model_info(model, (data_in, mask, 10))


if __name__ == '__main__':
    conf = OmegaConf.create({
         "conf": "conf/config.yaml",
         "cmd": "test",
         "n_freqs": 258,
         "n_frames": 158})
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    eval(conf.cmd)(conf)
