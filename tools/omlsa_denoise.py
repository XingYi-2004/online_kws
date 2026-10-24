# denoise with omlsa
import os
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import matlab.engine

eng = matlab.engine.start_matlab()
eng.addpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "./"))

def omlsa(noisy_file, est_file):
    _, _ = eng.omlsa(str(noisy_file), str(est_file), nargout=2)

def denoise_wavlist(pairlist):
    """denoise wavlist
    Args:
        pairlist (list): pairlist of wav, [(noisy, est), ...]
    Return:
        None
    """
    for (noisy_file, est_file) in tqdm(pairlist):
        omlsa(noisy_file, est_file)

def denoise_wav_scp(conf):
    with open(conf.wav_scp, "r") as f:
        all_lines = f.readlines()
        pairlist = [item.strip().split(" ") for item in all_lines]
    denoise_wavlist(pairlist)    

def prepare(conf):
    pairlist = []
    noisy_path = Path(conf.noisy_dir).expanduser()
    est_path = Path(conf.est_dir).expanduser()
    est_path.mkdir(parents=True, exist_ok=True)
    for noisy_wav in sorted(noisy_path.glob("*.wav")):
        pairlist.append([str(noisy_wav), str(est_path.joinpath(noisy_wav.name))])
    with open(conf.wav_scp, "w") as f:
        f.writelines("\n".join([" ".join(item) for item in pairlist]))
        f.write("\n")

if __name__ == '__main__':
    conf = OmegaConf.create({
        "cmd": "denoise_wav_scp",
        "noisy_dir": "~/data/DS_10283_2791/noisy_testset",
        "est_dir": "exp/denoise/OMLSA_voicebank_testset",
        "exp_dir": "exp/omlsa",
        "wav_scp": "exp/omlsa/wav.scp",
    })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)