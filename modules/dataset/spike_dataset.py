from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchaudio.transforms import AmplitudeToDB, MelScale, Resample as TorchResample, Spectrogram

from .shd import SpikingHeidelbergDigits, SpikingSpeechCommands
from . import augmentations
from .common import read_csv
from .data_store import WavStore

from modules.utils.logging import logger
from modules.utils import torch_audio

DEFAULT_GSC_LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

SPIKE_DATASET_NAMES = ('shd', 'ssc', 'gsc')

DEFAULT_GSC_NOISE_AUG = {
    'enabled': False,
    'noise_type': 'wav',
    'p': 1.0,
    'noise_csv': None,
    'noise_store': None,
    'snr_lower': 0,
    'snr_upper': 0,
    'target_level_lower': -35,
    'target_level_upper': -15,
    'seed': 0,
}


def _merge_dict(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if base:
        merged.update(base)
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_dict(merged[key], value)
            else:
                merged[key] = value
    return merged


def _get_dataset_conf(conf, subset: str, key: str) -> Dict[str, Any]:
    data_conf = OmegaConf.to_container(conf['data'], resolve=True)
    if not isinstance(data_conf, dict):
        return {}
    base_conf = data_conf.get(key, {})
    subset_conf = data_conf.get(subset, {})
    subset_key_conf = subset_conf.get(key, {}) if isinstance(subset_conf, dict) else {}
    return _merge_dict(base_conf, subset_key_conf)


def _apply_defaults(conf_dict: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(conf_dict)
    return merged


def _resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd().joinpath(path)
    return path.resolve()


def _strip_log_dir(path: Path, log_dir: Any) -> Optional[Path]:
    log_path = Path(str(log_dir)).expanduser()
    if log_path.is_absolute():
        return None

    log_parts = log_path.parts
    path_parts = path.parts
    for idx in range(len(path_parts) - len(log_parts), -1, -1):
        if path_parts[idx:idx + len(log_parts)] == log_parts:
            return Path(*path_parts[:idx])
    return None


def _infer_root_from_log_dir(conf, run_dir: Optional[Path] = None) -> Optional[Path]:
    log_dir = conf.get('system', {}).get('log_dir')
    if not log_dir:
        return None

    if run_dir is not None:
        root = _strip_log_dir(run_dir, log_dir)
        if root is not None:
            return root

    root_dir = conf.get('root_dir')
    if root_dir:
        return _strip_log_dir(_resolve_path(root_dir), log_dir)
    return None


def _resolve_experiment_root(conf) -> Path:
    conf_path = conf.get('conf')
    if conf_path:
        conf_path = _resolve_path(conf_path)
        conf_dir = conf_path.parent
        if conf_dir.name == 'conf':
            return conf_dir.parent
        if conf_path.name == 'hparams.yaml':
            log_root = _infer_root_from_log_dir(conf, conf_dir)
            if log_root is not None:
                return log_root
        return conf_dir
    return Path.cwd()


def _resolve_root(conf, root: Optional[str], default_subdir: str) -> str:
    data_dir = Path(conf['data'].get('data_dir', '.')).expanduser()
    if not data_dir.is_absolute():
        experiment_root = _resolve_experiment_root(conf)
        data_dir = experiment_root.joinpath(data_dir)

    if root is None:
        root_path = data_dir.joinpath(default_subdir)
    else:
        root_path = Path(root).expanduser()
        if not root_path.is_absolute():
            root_path = data_dir.joinpath(root_path)
    return str(root_path.resolve())


def _resolve_data_dir(conf) -> Path:
    data_dir = Path(conf['data'].get('data_dir', '.')).expanduser()
    if not data_dir.is_absolute():
        data_dir = _resolve_experiment_root(conf).joinpath(data_dir)
    return data_dir.resolve()


def _resolve_data_path(conf, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _resolve_data_dir(conf).joinpath(path)
    return path.resolve()


def _has_path_value(value: Any) -> bool:
    return value is not None and str(value).lower() != 'none'


def _resolve_noise_wav_paths(data_info, data_dir: Path):
    if data_info is None or 'wav_path' not in data_info.columns:
        return data_info

    def _resolve_item(value):
        value = str(value)
        if 'ark:' in value or 'hdf5:' in value:
            return value
        if '|' in value:
            return '|'.join(_resolve_item(item) for item in value.split('|'))
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        return str(data_dir.joinpath(path).resolve())

    data_info = data_info.copy()
    data_info['wav_path'] = data_info['wav_path'].map(_resolve_item)
    return data_info


def _as_namespace(conf_dict: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**conf_dict)


def _validate_n_bins(value: Any, dataset_name: str) -> int:
    n_bins = int(value)
    if n_bins <= 0:
        raise ValueError(f'{dataset_name}.n_bins must be positive')
    return n_bins


def _bin_frames(frames: Any, n_bins: int, dataset_name: str) -> np.ndarray:
    n_bins = _validate_n_bins(n_bins, dataset_name)
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(
            f'{dataset_name.upper()} frames must be a 2D [time, channels] array, '
            f'but got shape={frames.shape}'
        )
    if frames.shape[1] % n_bins != 0:
        raise ValueError(
            f'{dataset_name.upper()} channel count {frames.shape[1]} '
            f'is not divisible by n_bins={n_bins}'
        )
    return frames.reshape(frames.shape[0], -1, n_bins).sum(axis=2, dtype=np.float32)


def _load_binned_sample(dataset, index: int, dataset_name: str):
    frame_path, label = dataset.samples[index]
    with np.load(frame_path, allow_pickle=True) as frame_data:
        frames = frame_data['frames']
    binned_frames = _bin_frames(frames, dataset.n_bins, dataset_name)
    transform = getattr(dataset, 'transform', None)
    if transform is not None:
        binned_frames, label = transform(binned_frames, label)
    target_transform = getattr(dataset, 'target_transform', None)
    if target_transform is not None:
        label = target_transform(label)
    return np.asarray(binned_frames, dtype=np.float32), int(label)


class TimeNeuronsMaskAug(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, x, y):
        # `TimeNeuronsMaskAug` operates on dense 2D frame arrays
        # (shape = [T, C]).  When the dataset returns raw events
        # (``{'t': ..., 'x': ...}``), there is nothing to mask spatially,
        # so we leave the event dict untouched and return it as-is.
        if isinstance(x, dict):
            return x, y

        if self.config.TN_mask_aug_proba > 0 and self.config.time_mask_proportion > 0:
            if np.random.uniform() < self.config.TN_mask_aug_proba:
                mask_size = int(self.config.time_mask_proportion * x.shape[0])
                if 0 < mask_size < x.shape[0]:
                    ind = np.random.randint(0, x.shape[0] - mask_size)
                    x[ind:ind + mask_size, :] = 0

        if self.config.TN_mask_aug_proba > 0 and self.config.neuron_mask_size > 0:
            if np.random.uniform() < self.config.TN_mask_aug_proba:
                max_size = min(self.config.neuron_mask_size, x.shape[1])
                if max_size > 0:
                    mask_size = np.random.randint(1, max_size + 1)
                    ind = np.random.randint(0, x.shape[1] - mask_size + 1)
                    x[:, ind:ind + mask_size] = 0

        return x, y


class Augs(object):
    def __init__(self, config):
        self.config = config
        self.augs = [TimeNeuronsMaskAug(config)]

    def __call__(self, x, y):
        for aug in self.augs:
            x, y = aug(x, y)
        return x, y

    def list_augs(self):
        return [aug.__class__.__name__ for aug in self.augs]


class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
        self,
        root: str,
        n_bins: int,
        train: Optional[bool] = None,
        data_type: str = 'event',
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Any] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        n_bins = _validate_n_bins(n_bins, 'shd')
        super().__init__(
            root,
            train,
            data_type,
            frames_number,
            split_by,
            duration,
            custom_integrate_function,
            custom_integrated_frames_dir_name,
            transform,
            target_transform,
        )
        self.data_type = data_type
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events, label = self.transform(events,label)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, int(label)

        return _load_binned_sample(self, i, 'shd')


class BinnedSpikingSpeechCommands(SpikingSpeechCommands):
    def __init__(
        self,
        root: str,
        n_bins: int,
        split: str = 'train',
        data_type: str = 'event',
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Any] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ) -> None:
        n_bins = _validate_n_bins(n_bins, 'ssc')
        super().__init__(
            root,
            split,
            data_type,
            frames_number,
            split_by,
            duration,
            custom_integrate_function,
            custom_integrated_frames_dir_name,
            transform,
            target_transform,
        )
        # Mirror BinnedSpikingHeidelbergDigits: expose `data_type` as a
        # plain instance attribute so `__getitem__` can read it directly.
        self.data_type = data_type
        self.n_bins = n_bins

    def __getitem__(self, index: int):
        if self.data_type == 'event':
            events = {
                't': self.h5_file['spikes']['times'][index],
                'x': self.h5_file['spikes']['units'][index],
            }
            label = self.h5_file['labels'][index]
            if self.transform is not None:
                events, label = self.transform(events, label)
            if self.target_transform is not None:
                label = self.target_transform(label)
            return events, int(label)

        return _load_binned_sample(self, index, 'ssc')


class _GSCWaveformNoiseAug:
    def __init__(self, conf, noise_conf: Dict[str, Any], deterministic: bool, sample_rate: int):
        self.p = float(noise_conf.get('p', 1.0))
        if not 0.0 <= self.p <= 1.0:
            raise ValueError('gsc.noise_aug.p must be between 0 and 1')

        self.sample_rate = int(sample_rate)
        self.seed = int(noise_conf.get('seed', 0))
        self.deterministic = deterministic
        self.snr_lower = float(noise_conf['snr_lower'])
        self.snr_upper = float(noise_conf['snr_upper'])
        self.target_level_lower = float(noise_conf['target_level_lower'])
        self.target_level_upper = float(noise_conf['target_level_upper'])
        self._validate_range('snr', self.snr_lower, self.snr_upper)
        self._validate_range('target_level', self.target_level_lower, self.target_level_upper)

        self.noise_type = str(noise_conf.get('noise_type', 'wav')).lower()
        if self.noise_type not in ('wav', 'gaussian'):
            raise ValueError('gsc.noise_aug.noise_type must be "wav" or "gaussian"')
        self.noise_store = None
        if self.noise_type == 'wav':
            noise_csv = noise_conf.get('noise_csv')
            noise_store = noise_conf.get('noise_store')
            if not _has_path_value(noise_csv) and not _has_path_value(noise_store):
                raise ValueError('gsc.noise_aug.enabled=True requires noise_csv or noise_store')

            data_dir = _resolve_data_dir(conf)
            noise_csv_path = _resolve_data_path(conf, noise_csv) if _has_path_value(noise_csv) else None
            noise_store_path = _resolve_data_path(conf, noise_store) if _has_path_value(noise_store) else None
            data_info = read_csv(noise_csv_path) if noise_csv_path else None
            data_info = _resolve_noise_wav_paths(data_info, data_dir)
            self.noise_store = WavStore(noise_store_path, data_info)
            if len(self.noise_store) == 0:
                raise ValueError('gsc.noise_aug noise source is empty')
            self._assert_noise_sample_rate(data_info)

    def _assert_noise_sample_rate(self, data_info) -> None:
        '''Reject on-disk noise wavs whose sample rate differs from the clean
        audio. Resampling is left to the user (offline) to keep the online path
        cheap and lossless. Only plain audio files can be probed; ark/hdf5-packed
        stores carry no readable header and are skipped.'''
        # TODO:add option to assert off-line
        if data_info is None or 'wav_path' not in data_info.columns:
            return
        import soundfile as sf
        for wav_path in data_info['wav_path']:
            wav_path = str(wav_path)
            if 'ark:' in wav_path or 'hdf5:' in wav_path or '|' in wav_path:
                continue
            noise_sr = sf.info(wav_path).samplerate
            if noise_sr != self.sample_rate:
                raise ValueError(
                    f'gsc.noise_aug noise sample rate ({noise_sr} Hz) does not match '
                    f'clean audio sample rate ({self.sample_rate} Hz): {wav_path}. '
                    'Resample the noise offline before running the ablation.'
                )

    @staticmethod
    def _validate_range(name: str, lower: float, upper: float) -> None:
        if lower > upper:
            raise ValueError(f'gsc.noise_aug.{name}_lower must be <= {name}_upper')

    def _make_generator(self, index: Optional[int]) -> Optional[torch.Generator]:
        if not self.deterministic:
            return None
        if index is None:
            raise ValueError('deterministic GSC noise augmentation requires sample index')
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        return generator

    @staticmethod
    def _randint_closed(lower: int, upper: int, generator: Optional[torch.Generator]) -> int:
        if lower == upper:
            return lower
        return int(torch.randint(lower, upper + 1, (1,), generator=generator).item())

    def _sample_range(
        self,
        lower: float,
        upper: float,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if lower == upper:
            value = lower
        elif lower.is_integer() and upper.is_integer():
            value = self._randint_closed(int(lower), int(upper), generator)
        else:
            value = torch.rand((), generator=generator).item() * (upper - lower) + lower
        return torch.tensor([[value]], dtype=dtype, device=device)

    @staticmethod
    def _flatten_noise(noise: torch.Tensor) -> torch.Tensor:
        if noise.ndim == 1:
            return noise
        if noise.ndim == 2 and noise.shape[0] == 1:
            return noise.squeeze(0)
        if noise.ndim == 2 and noise.shape[1] == 1:
            return noise.squeeze(1)
        raise ValueError(f'GSC noise waveform must be mono, got shape={tuple(noise.shape)}')

    def _sample_wav_noise(
        self,
        num_samples: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if self.noise_store is None:
            raise ValueError('gsc.noise_aug wav noise source is not initialized')
        noise_index = self._randint_closed(0, len(self.noise_store) - 1, generator)
        _, noise = self.noise_store.get_data_by_index(noise_index)
        noise = torch.as_tensor(noise, dtype=dtype, device=device)
        noise = self._flatten_noise(noise)
        if noise.numel() == 0:
            raise ValueError('GSC noise waveform is empty')

        if noise.numel() < num_samples:
            repeats = (num_samples + noise.numel() - 1) // noise.numel()
            noise = noise.repeat(repeats)
        if noise.numel() > num_samples:
            max_start = noise.numel() - num_samples
            start = self._randint_closed(0, max_start, generator)
            noise = noise[start:start + num_samples]
        return noise.reshape(1, num_samples)

    @staticmethod
    def _sample_gaussian_noise(
        num_samples: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.randn((1, num_samples), dtype=dtype, device=device, generator=generator)

    def _sample_noise(
        self,
        num_samples: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if self.noise_type == 'gaussian':
            return self._sample_gaussian_noise(num_samples, dtype, device, generator)
        return self._sample_wav_noise(num_samples, dtype, device, generator)

    def __call__(self, waveform: torch.Tensor, index: Optional[int]) -> torch.Tensor:
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(
                'GSC noise augmentation expects mono waveform with shape [1, samples], '
                f'got shape={tuple(waveform.shape)}'
            )

        generator = self._make_generator(index)
        if self.p <= 0.0:
            return waveform
        if self.p < 1.0 and torch.rand((), generator=generator).item() > self.p:
            return waveform

        num_samples = waveform.shape[-1]
        noise = self._sample_noise(num_samples, waveform.dtype, waveform.device, generator)
        snr = self._sample_range(self.snr_lower, self.snr_upper, waveform.dtype, waveform.device, generator)
        noisy_rms_level = self._sample_range(
            self.target_level_lower,
            self.target_level_upper,
            waveform.dtype,
            waveform.device,
            generator,
        )
        # snr and noisy_rms_level are supplied explicitly, so snr_mixer reads
        # nothing from its params dict.
        _, _, noisy, _ = torch_audio.snr_mixer(
            {},
            waveform,
            noise,
            snr=snr,
            noisy_rms_level=noisy_rms_level,
        )
        return noisy


class _GSCTransform:
    def __init__(
        self,
        conf: Dict[str, Any],
        is_train: bool,
        noise_aug: Optional[_GSCWaveformNoiseAug] = None,
    ):
        sample_rate = conf.get('sample_rate', 16000)
        window_size = conf.get('window_size', 400)
        hop_length = conf.get('hop_length', 160)
        n_mels = conf.get('n_mels', 80)
        f_min = conf.get('f_min', 50)
        f_max = conf.get('f_max', 14000)
        resample_rate = conf.get('resample_rate', sample_rate // 2)

        self.pad = augmentations.PadOrTruncate(sample_rate)
        self.noise_aug = noise_aug
        self.transforms = [
            TorchResample(sample_rate, resample_rate),
        ]

        use_aug = is_train and conf.get('use_aug', False)
        if use_aug:
            augmentation_keys = ('speed_rates', 'speed_p', 'mF', 'F', 'mT', 'pS')
            missing_keys = [key for key in augmentation_keys if key not in conf]
            if missing_keys:
                raise ValueError(
                    'GSC augmentation is enabled, but these explicit configuration '
                    f'fields are missing: {", ".join(missing_keys)}'
                )
            self.transforms.extend([
                augmentations.RandomRoll(dims=(1,)),
                augmentations.SpeedPerturbation(
                    rates=tuple(conf['speed_rates']),
                    p=conf['speed_p'],
                ),
            ])

        self.transforms.extend([
            Spectrogram(n_fft=window_size, hop_length=hop_length, power=2),
            MelScale(
                n_mels=n_mels,
                sample_rate=resample_rate,
                f_min=f_min,
                f_max=f_max,
                n_stft=window_size // 2 + 1,
            ),
            AmplitudeToDB(),
        ])
        if use_aug:
            self.transforms.append(
                augmentations.SpecAugment(
                    mF=conf['mF'],
                    F=conf['F'],
                    mT=conf['mT'],
                    pS=conf['pS'],
                )
            )

    def __call__(self, waveform: torch.Tensor, index: Optional[int] = None) -> torch.Tensor:
        waveform = self.pad(waveform)
        if self.noise_aug is not None:
            waveform = self.noise_aug(waveform, index)
        for transform in self.transforms:
            waveform = transform(waveform)
        return waveform


def _build_gsc_transform(
    conf: Dict[str, Any],
    is_train: bool,
    noise_aug: Optional[_GSCWaveformNoiseAug] = None,
) -> _GSCTransform:
    return _GSCTransform(conf, is_train, noise_aug)


class SHDDataset(BinnedSpikingHeidelbergDigits):
    def __init__(self, conf, subset: str = 'train'):
        dataset_conf = _get_dataset_conf(conf, subset, 'shd')
        dataset_conf = _apply_defaults(
            dataset_conf,
            {
                'n_bins': 1,
                'data_type': 'frame',
                'time_step': None,
                'use_aug': False,
                'TN_mask_aug_proba': 0.0,
                'time_mask_proportion': 0.0,
                'neuron_mask_size': 0,
            },
        )
        n_bins = _validate_n_bins(dataset_conf['n_bins'], 'shd')

        root_value = dataset_conf.get('root', dataset_conf.get('root_path'))
        root = _resolve_root(conf, root_value, 'shd')
        data_type = dataset_conf['data_type']
        transform = None
        if subset == 'train' and dataset_conf['use_aug'] and data_type == 'event':
            logger.warning(
                "shd.use_aug=True is incompatible with "
                "data_type='event'; ignoring augmentations for SHD."
            )
        elif subset == 'train' and dataset_conf['use_aug']:
            transform = Augs(_as_namespace(dataset_conf))

        super().__init__(
            root=root,
            n_bins=n_bins,
            train=subset == 'train',
            data_type=data_type,
            duration=dataset_conf['time_step'],
            transform=transform,
        )


class SSCDataset(BinnedSpikingSpeechCommands):
    def __init__(self, conf, subset: str = 'train'):
        dataset_conf = _get_dataset_conf(conf, subset, 'ssc')
        dataset_conf = _apply_defaults(
            dataset_conf,
            {
                'n_bins': 1,
                'data_type': 'frame',
                'time_step': None,
                'use_aug': False,
                'TN_mask_aug_proba': 0.0,
                'time_mask_proportion': 0.0,
                'neuron_mask_size': 0,
            },
        )
        n_bins = _validate_n_bins(dataset_conf['n_bins'], 'ssc')

        root = _resolve_root(conf, dataset_conf.get('root'), 'ssc')
        data_type = dataset_conf['data_type']
        transform = None
        if subset == 'train' and dataset_conf['use_aug'] and data_type == 'event':
            logger.warning(
                "ssc.use_aug=True is incompatible with "
                "data_type='event'; ignoring augmentations for SSC."
            )
        elif subset == 'train' and dataset_conf['use_aug']:
            transform = Augs(_as_namespace(dataset_conf))

        split_map = {'train': 'train', 'valid': 'valid', 'test': 'test'}
        split = split_map.get(subset, 'train')

        super().__init__(
            root=root,
            n_bins=n_bins,
            split=split,
            data_type=data_type,
            duration=dataset_conf['time_step'],
            transform=transform,
        )


class GSCDataset(Dataset):
    def __init__(self, conf, subset: str = 'train'):
        dataset_conf = _get_dataset_conf(conf, subset, 'gsc')
        noise_defaults = dict(DEFAULT_GSC_NOISE_AUG)
        noise_defaults['seed'] = conf.get('system', {}).get('seed', 0)
        dataset_conf['noise_aug'] = _merge_dict(
            noise_defaults,
            dataset_conf.get('noise_aug', {}),
        )
        dataset_conf = _apply_defaults(
            dataset_conf,
            {
                'download': True,
                'use_aug': False,
                'sample_rate': conf['data'].get('sample_rate', 16000),
            },
        )
        root = _resolve_root(conf, dataset_conf.get('root'), 'gsc')
        split_map = {'train': 'training', 'valid': 'validation', 'test': 'testing'}
        split = split_map.get(subset, 'training')
        noise_aug = None
        if dataset_conf['noise_aug']['enabled']:
            noise_aug = _GSCWaveformNoiseAug(
                conf,
                dataset_conf['noise_aug'],
                deterministic=subset != 'train',
                sample_rate=dataset_conf.get('sample_rate', 16000),
            )
        self.transform = _build_gsc_transform(dataset_conf, subset == 'train', noise_aug)
        self.labels = dataset_conf.get('labels', DEFAULT_GSC_LABELS)
        self.label_to_index = {name: idx for idx, name in enumerate(self.labels)}
        download = bool(dataset_conf['download'])
        if download:
            Path(root).mkdir(parents=True, exist_ok=True)
        self.dataset = SPEECHCOMMANDS(root, download=download, subset=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        waveform, _, label, _, _ = self.dataset[index]
        if self.transform is not None:
            waveform = self.transform(waveform, index).squeeze(0).transpose(0, 1)
        if label not in self.label_to_index:
            raise ValueError(f'Unknown label: {label}')
        target = self.label_to_index[label]
        frames = torch.as_tensor(waveform, dtype=torch.float32).contiguous()
        return frames, int(target)


SPIKE_DATASET_REGISTRY = {
    'shd': SHDDataset,
    'ssc': SSCDataset,
    'gsc': GSCDataset,
}


class SpikeDataset(Dataset):
    """Route a config-selected spiking dataset into the map-style pipeline."""

    def __init__(self, conf, subset: str = 'train'):
        spike_conf = _get_dataset_conf(conf, subset, 'spike')
        dataset_name = spike_conf.get('name')
        if not dataset_name:
            raise ValueError(
                'data.spike.name is required; '
                f'available datasets are {", ".join(SPIKE_DATASET_NAMES)}'
            )

        dataset_name = str(dataset_name).lower()
        dataset_class = SPIKE_DATASET_REGISTRY.get(dataset_name)
        if dataset_class is None:
            raise ValueError(
                f'Unknown spiking dataset: {dataset_name}; '
                f'available datasets are {", ".join(SPIKE_DATASET_NAMES)}'
            )

        dataset_conf = _get_dataset_conf(conf, subset, dataset_name)
        data_type = dataset_conf.get('data_type', 'frame')
        if data_type != 'frame':
            raise NotImplementedError(
                'SpikeDataset currently supports only data_type="frame"; '
                f'got {dataset_name}.data_type={data_type!r}'
            )

        self.dataset_name = dataset_name
        self.dataset = dataset_class(conf, subset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]
