import whisper
import soundfile as sf

# model = whisper.load_model("base")
model = whisper.load_model("turbo", download_root="/home/jiangwenbin/shared/download/whisper")

audio_path = "/home/jiangwenbin/shared/workspace/AlphaCore/examples/voicebank/exp/logs/UnSE/3a0e/noisy/p232_102.wav"
audio, sample_rate = sf.read(audio_path, dtype='float32')

audio_padded = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio_padded, n_mels=model.dims.n_mels, device=model.device)

# get audio features
audio_features = model.encoder(mel.unsqueeze(0))
print("audio_features:", audio_features.shape)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)