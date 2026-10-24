# ref: https://github.com/shanwangshan/asymmetric_window/blob/main/Asymmetric_window_on_oracle.ipynb
from modules.utils.common import get_samples, get_batch_samples, torch_float32
from modules.utils import metrics
import numpy as np
from scipy import signal
from numpy.lib.stride_tricks import as_strided
import librosa
import torch
from omegaconf import OmegaConf

def asymmetric_window(long_length=512, short_length=128, hop_length=64, window="hamming"):
    K = long_length
    M = short_length // 2
    d = hop_length

    pro_1 = K - M - d
    pro_2 = M
    win_pro_1 = signal.get_window(window, 2 * pro_1, fftbins=False)
    win_pro_2 = signal.get_window(window, 2 * pro_2, fftbins=False)

    # analysis_win
    win_an_1 = np.zeros((d))
    win_an_2 = np.sqrt(win_pro_1[: K - 2 * M - d])
    win_an_3 = np.sqrt(win_pro_1[K - 2 * M - d : K - 2 * M - d + M])
    win_an_4 = np.sqrt(win_pro_2[M : M + M])
    window_analysis = np.concatenate((win_an_1, win_an_2, win_an_3, win_an_4))

    # synthesis_win
    win_sy_3 = (win_pro_2[:M]) / np.sqrt(win_pro_1[K - 2 * M - d : K - 2 * M - d + M])
    win_sy_4 = np.sqrt(win_pro_2[M : M + M])
    window_synthesis = np.concatenate((win_sy_3, win_sy_4))

    return window_analysis, window_synthesis


def AsymSTFT(s, n_fft, long_length, short_length, hop_length, window_analysis):
    sig_padded = np.pad(s, (long_length - short_length, 0), "constant")
    n_frames = int((len(sig_padded) - long_length) / hop_length) + 1

    y_frames = as_strided(
        sig_padded,
        shape=(long_length, n_frames),
        strides=(sig_padded.itemsize, hop_length * sig_padded.itemsize),
    )
    spectrogram = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex_)
    for idx in np.arange(n_frames):
        a = y_frames[:, idx]
        b = np.pad(a, (0, (n_fft - long_length)), "constant")
        window_analysis_ = np.pad(window_analysis, (0, n_fft - long_length), "constant")
        window_frame = b * window_analysis_
        spectrogram[:, idx] = np.fft.rfft(window_frame, axis=0)

    return spectrogram


def AsymISTFT(spectrogram_long, long_length, short_length, hop_length, length, window_synthesis):
    n_frames = spectrogram_long.shape[-1]
    output = np.zeros((length))
    for idx in np.arange(n_frames):
        a = spectrogram_long[:, idx]
        out_long = np.fft.irfft(a, axis=0)
        out_long_wnd = out_long[long_length - short_length : long_length]
        out_long_wnd = window_synthesis * out_long_wnd
        output[idx * hop_length : idx * hop_length + short_length] = (
            output[idx * hop_length : idx * hop_length + short_length] + out_long_wnd
        )
    return output


def AsymSTFT_librosa(s, n_fft, long_length, short_length, hop_length, window):
    s_padded = np.pad(s, (long_length - short_length, 0), mode='constant')
    return librosa.stft(
        s_padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=long_length,
        window=window, 
        center=False,
        pad_mode='constant'
    )


def AsymISTFT_parallel(spectrogram, short_length, hop_length, length, window_synthesis, axis=-2):
    # 无法使用librosa中的istft
    n_frames = spectrogram.shape[-1]
    all_frames = np.fft.irfft(spectrogram, axis=axis)
    windowed_frames = all_frames[-short_length:, :] * window_synthesis[:, None]
    print(windowed_frames.shape)
    buffer_length = max(length, (n_frames - 1) * hop_length + short_length)
    output = np.zeros(buffer_length)    
    starts = np.arange(n_frames) * hop_length
    indices = starts[:, None] + np.arange(short_length)
    np.add.at(output, indices.ravel(), windowed_frames.T.ravel())    
    return output[:length]


def AsymSTFT_torch(input, n_fft, long_length, short_length, hop_length, window):
    """
    Args:
        input: (batch_size, n_samples)
        n_fft:
    Returns:
        shape of [B, F, T]
    """
    input_padded = torch.nn.functional.pad(input, (long_length - short_length, 0))
    return torch.stft(
        input_padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=long_length,
        window=window,
        center=False,
        return_complex=True,
    )


def AsymISTFT_torch(spectrogram, short_length, hop_length, window, **kwargs):
    """
    非对称窗ISTFT
    Args:
        spectrogram: (batch_size, freq_bins, n_frames)
        short_length: 短窗口长度
        hop_length: 帧移长度
        window: 合成窗，形状为 (short_length,)
    Returns:
        output: 重建的时域信号，形状为 (batch_size, length)
    """
    batch_size, freq_bins, n_frames = spectrogram.shape
    device = spectrogram.device
    
    if 'length' in kwargs:
        length = kwargs['length']
    else:
        assert 'long_length' in kwargs, "'length' or 'long_length' must be provided !"
        length = (n_frames - 1) * hop_length + kwargs['long_length']
    
    all_frames = torch.fft.irfft(spectrogram, dim=1) # (batch_size, freq_bins, n_frames)

    window = window.to(device).view(1, -1, 1)  # (1, short_length, 1)
    windowed_frames = (all_frames[:, -short_length:, :] * window)  # (batch_size, short_length, n_frames)
    
    buffer_length = max(length, (n_frames - 1) * hop_length + short_length)
    output = torch.zeros((batch_size, buffer_length), device=device)

    starts = torch.arange(n_frames, device=device) * hop_length  #  (n_frames,)
    indices = starts.view(1, 1, -1) + torch.arange(
        short_length, device=device).view(1, -1, 1)  # (1, short_length, n_frames)
    output.scatter_add_(1, indices.expand(batch_size, -1, -1).reshape(batch_size, -1), 
                        windowed_frames.reshape(batch_size, -1))

    return output[:, :length]


def test(conf):
    n_fft = conf.get('n_fft', 512)
    long_length = conf.get('long_length', 512) # 32ms
    short_length = conf.get('short_length', 128) # 8ms    
    hop_length = conf.get('hop_length', 64)  # 4ms
    print(f"n_fft: {n_fft}, long: {long_length}, short: {short_length}, hop: {hop_length}")

    y, sr = get_samples()
    window_analysis, window_synthesis = asymmetric_window(long_length, short_length, hop_length)
    
    print("========AsymSTFT========")    
    spec = AsymSTFT(y, n_fft, long_length, short_length, hop_length, window_analysis)
    y_est = AsymISTFT(spec, long_length, short_length, hop_length, y.shape[0], window_synthesis)
    print(y.shape, y_est.shape)
    print("MSE:", np.mean((y - y_est) ** 2))
    print("Metric:", metrics.eval(y, y_est, sr))

    # Librosa实现STFT
    print("========AsymSTFT_librosa========")
    librosa_spec = AsymSTFT_librosa(y, n_fft, long_length, short_length, hop_length, window_analysis)
    print(librosa_spec.shape)
    print("max abs: ", np.max(np.abs(spec - librosa_spec)))
    y_est = AsymISTFT(librosa_spec, long_length, short_length, hop_length, y.shape[0], window_synthesis)
    print(y.shape, y_est.shape)
    print("MSE:", np.mean((y - y_est) ** 2))
    print("Metric:", metrics.eval(y, y_est, sr))

    # 并行实现ISTFT，无法用Librosa的istft实现
    print("========AsymISTFT_parallel========")
    spec = AsymSTFT(y, n_fft, long_length, short_length, hop_length, window_analysis)
    y_est = AsymISTFT_parallel(spec, short_length, hop_length, y.shape[0], window_synthesis)
    print(y.shape, y_est.shape)
    print("MSE:", np.mean((y - y_est) ** 2))
    print("Metric:", metrics.eval(y, y_est, sr))

    # pytorch
    print("========AsymSTFT_pytorch========")
    x, sr = get_batch_samples()
    x_torch = torch_float32(x)
    window_analysis = torch_float32(window_analysis)
    window_synthesis = torch_float32(window_synthesis)
    spec = AsymSTFT_torch(x_torch, n_fft, long_length, short_length, hop_length, window_analysis)
    print(spec.shape)
    x_est = AsymISTFT_torch(spec, short_length, hop_length, x_torch.shape[-1], window_synthesis)
    print(x_torch.shape, x_est.shape)
    print("MSE:", torch.mean((x_torch - x_est) ** 2))
    for i in range(x.shape[0]):
        print("Metric:", metrics.eval(x_torch[i, :], x_est[i, :], sr))


if __name__ == '__main__':
    conf = OmegaConf.create({
         "conf": "conf/config.yaml",
         "cmd": "test"
         })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    eval(conf.cmd)(conf)