import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from pathlib import Path
import soundfile as sf
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import torch

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeMOS:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    def __init__(self, conf) -> None:
        model_path = Path(os.environ['MAIN_ROOT']).joinpath(conf['DNSMOS']['model_path'])        
        self.onnx_sess = ort.InferenceSession(model_path, providers=self.providers)
        self.fs = conf['data']['sample_rate']
        self.hop_samples = self.fs
        self.n_samples = int(INPUT_LENGTH * self.fs)
        self.io_binding = self.onnx_sess.io_binding()
    
    def eval_numpy(self, audio):
        # audio: numpy array, shape of [n_samples]
        assert len(audio) == self.n_samples
        input_features = np.array(audio).astype('float32')[np.newaxis,:]
        oi = {'input_1': input_features}
        sig, bak, ovl = self.onnx_sess.run(None, oi)[0][0]
        return sig, bak, ovl
    
    def eval_torch(self, audio):
        # audio: torch.tensor, shape of [n_samples]
        assert audio.shape[-1] == self.n_samples        
        input = audio.contiguous()
        self.io_binding.bind_input(name='input_1', device_type='cuda', device_id=input.device.index,
                                   element_type=np.float32, shape=tuple(input.shape), buffer_ptr=input.data_ptr())
        # Prepare output buffers
        output_shape = (input.shape[0], 3)
        scores = torch.empty(output_shape, dtype=torch.float32, device='cuda').contiguous()
        self.io_binding.bind_output(name='Identity:0', device_type='cuda', device_id=input.device.index,
                                    element_type=np.float32, shape=output_shape, buffer_ptr=scores.data_ptr())
        # Run the session
        self.onnx_sess.run_with_iobinding(self.io_binding)
        return scores


def test_wav_dir(conf):
    worker = ComputeMOS(conf)
    wav_list = list(Path(conf.wav_dir).expanduser().glob("*.wav"))
    print("wav:", len(wav_list))
    score1_list = []
    score2_list = []
    for wav_path in tqdm(wav_list):
        audio, _ = sf.read(wav_path)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).cuda()
        audio_padded = F.pad(audio_tensor, (0, worker.n_samples-audio_tensor.shape[1]), mode='replicate')
        score1 = worker.eval_numpy(audio_padded.squeeze().cpu())
        score2 = worker.eval_torch(audio_padded).cpu().squeeze()
        assert np.allclose(np.array(score1), np.array(score2))
        score1_list.append(np.array(score1))
        score2_list.append(np.array(score2))
    print(np.stack(score1_list).mean(0), np.stack(score2_list).mean(0))


if __name__ == "__main__":
    conf = OmegaConf.create({
         "yaml": "conf/UnSE.yaml",
         "cmd": "test_wav_dir",
         "wav_dir": "~/shared/data/VCTK_DEMAND/wav_16kHz/clean_testset",
         "subset": "train",
         'pDataset': None,
         "metric": "MOS" # {"PESQ", "MOS"}
         })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
    