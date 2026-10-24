from omegaconf import OmegaConf
from wvmos import get_wvmos

def run_eval(conf):
    model = get_wvmos(cuda=True)
    mos = model.calculate_dir(conf.wav_dir, mean=True) # infer average MOS score across .wav files in directory
    print(f"wvmos: {mos}")

if __name__ == '__main__':
    conf = OmegaConf.create({
         "cmd": "run_eval",
         "wav_dir": "exp/logs/UnSE/3a0e/clean",
         })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)
    