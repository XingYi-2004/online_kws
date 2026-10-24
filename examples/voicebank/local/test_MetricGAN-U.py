import librosa
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
import functools
import torch
import soundfile as sf
from speechbrain.lobes.models.MetricGAN_U import EnhancementGenerator
from speechbrain.processing.signal_processing import resynthesize
from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.features import spectral_magnitude
from modules.utils.common import torch_float32, find_files


def eval_wav(conf):
    model_dict = torch.load(conf.ckpt, map_location="cpu")
    model = EnhancementGenerator()
    model.load_state_dict(model_dict)
    model.eval()
    stft = STFT(sample_rate=16000, win_length=32, hop_length=16, n_fft=512, window_fn=torch.hamming_window)
    istft = ISTFT(sample_rate=16000, win_length=32, hop_length=16, window_fn=torch.hamming_window)
    resynth = functools.partial(resynthesize, stft=stft, istft=istft, normalize_wavs=False)
    # wav_list = list(Path(conf.wav_dir).expanduser().rglob('*.wav'))
    wav_list = find_files(conf.wav_dir)
    dest = Path(conf.dest)
    for _wav in tqdm(wav_list):
        noisy_wav, fs = librosa.load(_wav, sr=conf.get('sr', None))
        noisy_wav = torch_float32(noisy_wav).unsqueeze(0)
        feats = stft(noisy_wav)
        noisy_spec = spectral_magnitude(feats, power=0.5)
        mask = model(noisy_spec, lengths=torch.ones([1]))
        mask = mask.clamp(min=0.2).squeeze(2)
        predict_spec = torch.mul(mask, noisy_spec)
        predict_wav = resynth(predict_spec, noisy_wav)
        sf.write(dest.joinpath(_wav.name), predict_wav.squeeze().detach().cpu().numpy(), fs)


if __name__ == '__main__':
    conf = OmegaConf.create({
        'cmd': 'eval_wav',
        'ckpt': 'save/CKPT+2021-10-27+14-22-39+00/generator.ckpt',
        'wav_dir': '~/shared/data/DS_10283_2791/noisy-vctk-16k/noisy_testset_wav_16k',
        'dest': 'est'})
    conf.merge_with_cli()
    eval(conf.cmd)(conf)
