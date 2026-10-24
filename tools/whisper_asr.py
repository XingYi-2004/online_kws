from pathlib import Path
from omegaconf import OmegaConf
import whisper
from platform import node
from modules.utils.common import find_available_gpu, read_scp
import torch.utils.data as tdata
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import torch
import jiwer
from whisper.normalizers import EnglishTextNormalizer
import os

def init():
    if "CUDA_VISIBLE_DEVICES" not in os.environ: # slrum will set it
        os.environ['CUDA_VISIBLE_DEVICES'] = find_available_gpu()    
    print("Hostname: {}, CUDA_VISIBLE_DEVICES: [{}], device_count: {}".format(
        node(), os.environ.get('CUDA_VISIBLE_DEVICES', None), torch.cuda.device_count()))

class CHiME4(tdata.Dataset):
    __key_map = {
        "valid": {"real": "dt05_real_isolated_1ch_track",
                  "simu": "dt05_simu_isolated_1ch_track"
                  },
        "test": {"real": "et05_real_isolated_1ch_track",
                 "simu": "et05_simu_isolated_1ch_track"}
    }
    
    def __init__(self, conf, subset, scene, n_mels):
        self.n_mels = n_mels
        data_dir = Path(conf['data_dir']).expanduser()        
        wav_dir = self.__key_map[subset][scene]
        tag = conf.get('tag', None)
        if tag:
            wav_dir = f"{tag}_{wav_dir}"
        data_dir = data_dir.joinpath(wav_dir)
        if data_dir.joinpath("wav.scp").exists():
            wav_dict = read_scp(data_dir.joinpath("wav.scp"))
        elif data_dir.joinpath("spk1.scp").exists():
            wav_dict = read_scp(data_dir.joinpath("spk1.scp"))
        else:
            raise FileNotFoundError
        if conf['clean'] is True:
            for wav_id in wav_dict:
                wav_dict[wav_id] = wav_dict[wav_id].replace(
                    "isolated", "isolated_ext").replace(".wav", ".Clean.wav")
        text_dict = read_scp(data_dir.joinpath("text"))
        self.dataset = [] # wav_id, wav_path, text
        for wav_id in wav_dict:
            self.dataset.append((wav_id, wav_dict[wav_id], text_dict[wav_id]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        wav_id, wav_path, text = self.dataset[idx]
        audio, sample_rate = sf.read(wav_path)
        audio = audio.astype(np.float32)
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        return (mel, text)
    

def ASR(conf):
    init()
    model = whisper.load_model(conf["model"])
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    if conf['clean'] is True:
        assert conf['scene'] == 'simu'
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    subset_list = conf.get("subset", "valid,test").split(",")
    scene_list = conf.get("scene", "simu,real").split(",")
    result_path = Path("exp/results/whisper/")
    enh = conf.get('enh', "noisy")
    result_path.mkdir(parents=True, exist_ok=True)
    for subset in subset_list:
        for scene in scene_list:
            dataset = CHiME4(conf, subset, scene, model.dims.n_mels)
            loader = torch.utils.data.DataLoader(dataset, batch_size=conf.batch_size)            
            hypotheses = []
            references = []
            for mels, texts in tqdm(loader):
                results = model.decode(mels.to("cuda"), options)
                hypotheses.extend([result.text for result in results])
                references.extend(texts)
            data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
            normalizer = EnglishTextNormalizer()
            data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
            data["reference_clean"] = [normalizer(text) for text in data["reference"]]
            data.to_csv(result_path.joinpath(f"CHiME4_{subset}_{scene}_{enh}.csv"))
            wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
            print(f"[{subset}_{scene}_{enh}] WER: {wer * 100:.2f} %")


def generate_wav_scp(conf):
    post_tag = conf.get('post_tag', True)
    scene = conf['scene'].upper()
    wav_path = Path(conf['wav_dir'])
    wav_path_list = list(wav_path.rglob("*.wav"))
    data_list = []
    for item in wav_path_list:
        if post_tag:
            data_list.append((f"{item.stem}_{scene}", str(item.resolve())))
        else:
            data_list.append((item.stem, str(item.resolve())))
    df = pd.DataFrame(data=data_list, columns=['id', 'wav_path'])
    df = df.set_index('id')
    df.to_csv(conf.wav_scp, header=None, sep=" ")
    

def test_CHiME4(conf):
    dataset = CHiME4(conf, "valid", "simu")
    num_workers = conf.get('num_workers', 4)
    dataloader = tdata.DataLoader(dataset, shuffle=True, batch_size=conf.batch_size, 
                                  num_workers=num_workers)
    item = iter(dataloader).next()
    print("item:", len(item), ", wavshape:", item[0].shape)
    # travel through the data iterator
    for _ in tqdm(dataloader):
        pass
    
    
if __name__ == '__main__':
    conf = OmegaConf.create({
         "cmd": "ASR",
         "model": "base.en",
         "batch_size": 30,
         "data_dir": "~/workspace/espnet/egs2/chime4/enh1/data",
         "clean": False
        #  "scene": "real"
    })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)