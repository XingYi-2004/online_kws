from modules.utils.common import get_batch_samples, torch_float32
from modules.utils import metrics
import torch
from omegaconf import OmegaConf
from modules.stft import STFT

def test(conf):
    x, sr = get_batch_samples()
    x_torch = torch_float32(x)
    stft = STFT(conf['stft'])
    x_stft = stft.apply_stft(x_torch)
    print(x_stft.shape)
    # x_est = stft.apply_istft(x_stft, length=x_torch.shape[-1])
    x_est = stft.apply_istft(x_stft)
    x_est = x_est[..., :x_torch.shape[-1]]
    print(x_torch.shape, x_est.shape)
    print("MSE:", torch.mean((x_torch - x_est) ** 2))
    for i in range(x.shape[0]):
        print("Metric:", metrics.eval(x_torch[i, :], x_est[i, :], sr))

if __name__ == '__main__':
    conf = OmegaConf.create({
         "conf": "conf/FBankSE_ull.yaml",
         "cmd": "test"
         })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    eval(conf.cmd)(conf)