import pytest
import torch
from torch import tensor
import librosa
import numpy as np
import modules.utils.signal as signal 
from modules.utils.common import get_pool_executor, torch_float32

rtol = 1e-03 # relative tolerance
atol = 1e-04 # absolute tolerance

audio_size = 1000
audio = torch.randn((1, audio_size), dtype=torch.float32)

@pytest.mark.parametrize("n_sampes", [200, 256, 400, 512, 1024])
def test_drugman_unwrapping(n_sampes):
    for _ in range(10):
        random_signal = np.random.uniform(-1, 1, n_sampes)
        signal.drugman_unwrapping(random_signal)
    # tensor
    batch_signal = torch.rand(8, n_sampes)
    signal_numpy = [item.numpy() for item in batch_signal]
    
    pool = get_pool_executor()
    phase_tensor = torch_float32(np.array(list(pool.map(signal.drugman_unwrapping, signal_numpy))))
    for i in range(batch_signal.shape[0]):
        phase = torch_float32(signal.drugman_unwrapping(batch_signal[i,:].numpy()))
        assert torch.allclose(phase, phase_tensor[i])
    


from omegaconf import OmegaConf    
if __name__ == '__main__':
    test_drugman_unwrapping(200)