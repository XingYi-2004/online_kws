import torch
from modules.wamlm import WavLM, WavLMConfig
import soundfile as sf

audio_path = "/home/jiangwenbin/shared/workspace/AlphaCore/examples/voicebank/exp/logs/UnSE/3a0e/noisy/p232_102.wav"
audio, sample_rate = sf.read(audio_path, dtype='float32')

# load the pre-trained checkpoints
checkpoint = torch.load('./data/model/wavlm/WavLM-Base+.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the representation of last layer
# wav_input_16khz = torch.randn(1, 2*16000)
wav_input_16khz = torch.tensor(audio).unsqueeze(0)
if cfg.normalize:
    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
rep = model.extract_features(wav_input_16khz)[0]
print(rep.shape)

# extract the representation of each layer
# wav_input_16khz = torch.randn(1, 2*16000)
wav_input_16khz = torch.tensor(audio).unsqueeze(0)
if cfg.normalize:
    input = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
rep, layer_results = model.extract_features(input, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]