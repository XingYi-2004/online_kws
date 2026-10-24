from tqdm import tqdm
import librosa
from pathlib import Path
from .utils import audio
from .utils.init_conf import init_conf
from .utils.common import get_pool_executor
from alpha.enh import system


def _save_wav(wav_path, wav, volume=None, volume_type='rms'):
    if volume:
        if volume_type == 'rms':
            audio.audiowrite(wav_path, wav, norm=True, target_level=volume)
        else:
            wav = audio.normalize_with_peak(wav, peak_level=volume)
            audio.audiowrite(wav_path, wav)
    else:
        audio.audiowrite(wav_path, wav)


def denoise(conf):
    device = conf.get('device', 'cuda')
    assert 'denoise' in conf
    w_oa = conf.denoise.get('w_oa', 0.0)  # observation adding
    if w_oa > 0:
        print(f'w_oa: {w_oa}')
    volume_type = conf.get('volume_type', 'rms')  # rms, peak
    volume = conf.get('volume', None)  # volume in dB, e.g., -25, -2.02
    if volume:
        print(f'volume_type:{volume_type}, volume: {volume}')
    pool = get_pool_executor(**conf['pool'])
    future_tasks = []
    out_dir = Path(conf.denoise.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    if conf.denoise.get('wavlist', None):
        with open(conf.denoise.wavlist) as f:
            content = f.readlines()
            wav_list = [Path(wav.strip()) for wav in content]
    else:
        wav_list = list(Path(conf.denoise.wav_dir).expanduser().rglob('*.wav'))
    print('wav_list: ', wav_list[:3], '...')
    conf = init_conf(conf)
    print(conf.ckpt)
    if conf.ckpt is None:
        return
    model = getattr(system, conf['system']['name']).load_from_checkpoint(conf.ckpt, conf=conf).to(device)
    model.eval()  # evaluation mode
    for _wav in tqdm(wav_list):
        y, _ = librosa.load(_wav, sr=conf.get('sr', None))
        est_wav = model.denoise(y, conf.denoise.get('chunk', -1)).cpu().numpy()
        if w_oa > 0:
            est_wav = (1 - w_oa) * est_wav + w_oa * y
        future_tasks.append(pool.submit(_save_wav, out_dir.joinpath(
            _wav.name), est_wav, volume, volume_type))
    for _ in tqdm(range(len(future_tasks))):
        f = future_tasks.pop(0)
        _ = f.result()
