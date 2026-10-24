from pathlib import Path
from omegaconf import OmegaConf

def testset_1mic(conf):
    wav_path = Path(conf.data_dir).expanduser()
    for item in wav_path.rglob("*.wav"):
        for key in ["_Noise", "_Noisy", "_Clean_", "exp_16k256_gru_deep_filter_fsmn_SPECMRloss"]:
            if key in item.name:
                target_name = item.name.replace(key, "")
                item.rename(item.parent.joinpath(target_name))
                break

def br28(conf):
    wav_path = Path(conf.data_dir).expanduser()
    for item in wav_path.rglob("*.wav"):
        for key in ["_Clean_", "_Enhance_2mic_post_eq"]:
            if key in item.name:
                target_name = item.name.replace(key, "")
                item.rename(item.parent.joinpath(target_name))
                break

if __name__ == '__main__':
    conf = OmegaConf.create({
         "cmd": "br28",
         "data_dir": "~/data/aispeech/br28"
         })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)