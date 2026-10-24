from omegaconf import OmegaConf
from pathlib import Path
from tools.prepare_data import get_wav_dur
from modules.utils.common import get_pool_executor

pool = get_pool_executor()


def prepare_DNS2020(conf):
    data_dir = conf.get("data_dir", "/public/DNS2020/datasets/test_set/real_recordings/")
    csv_file = conf.get("csv_file", "data/test/real_recordings.csv")
    wav_type = conf.get("wav_type", "wav")  # flac, wav
    wav_list = list(Path(data_dir).rglob(f"*.{wav_type}"))
    wav_df = get_wav_dur(wav_list, pool)
    wav_df = wav_df.set_index("wav_id")
    wav_df.to_csv(csv_file)


def prepare_LibriSpeech(conf):
    dB = conf.get("dB", 0)
    data_dir = conf.get("data_dir", f"/home/wbj68/workspace/AlphaCore/examples/DNS2021/exp/denoise/noisy_{dB}db/",)
    csv_file = conf.get("csv_file", f"data/test/LibriSpeech_{dB}dB.csv")
    wav_type = conf.get("wav_type", "wav")  # flac, wav
    wav_list = list(Path(data_dir).rglob(f"*.{wav_type}"))
    wav_df = get_wav_dur(wav_list, pool)
    wav_df = wav_df.set_index("wav_id")
    wav_df.to_csv(csv_file)


if __name__ == "__main__":
    conf = OmegaConf.create({
        "cmd": "prepare_DNS2020",
    })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)
