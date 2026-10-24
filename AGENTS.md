# AGENT.md

## Project Vision
AlphaCore is a speech toolkit for training and running enhancement, ASR, and joint enhancement+ASR systems with a shared Lightning-based pipeline.

## Tech Stack
- Language: Python 3.10
- Frameworks: PyTorch 2.1.0, Lightning 2.2.5
- Key libraries: numpy 1.26.4, librosa 0.10.2, onnxruntime-gpu 1.15.0, h5py 3.11
- Supporting libs used across modules: omegaconf, loguru, rich, pandas, scipy, torchaudio, torch_ema

## Folder Map
- alpha/: Task-specific packages for enhancement (enh), ASR (asr), and joint enh+asr (enh_asr). Each defines system classes, models, and losses.
- modules/: Shared training/runtime infrastructure (Lightning BaseModule, data loaders, optimizers, STFT, utilities, and command entry points).
- modules/blocks/: Neural network building blocks (UNet, DPRNN, conformer variants, etc.).
- modules/dataset/: Dataset IO, chunking, caching, and feature extraction.
- modules/loss/, modules/model/, modules/utils/: Common losses, model helpers, logging, and signal processing utilities.
- examples/: Dataset-specific configs and helper scripts for common benchmarks (VoiceBank, DNS2021, CHiME, etc.).
- scripts/: Launch helpers and environment setup scripts.
- tools/: Standalone utilities for evaluation, data preparation, and inference helpers.
- test/: Sanity and smoke tests (PyTorch, dataset, ONNX, signal).

## Project Tree
```
AlphaSNN/
├── AGENT.md                # AI collaboration guide (this file)
├── readme.md               # Setup and dependency versions
├── alpha/                  # Task packages (enh, asr, enh_asr)
│   ├── enh/                 # Enhancement systems, models, losses
│   ├── asr/                 # ASR systems, models, losses
│   └── enh_asr/             # Joint enhancement + ASR systems
├── modules/                # Core runtime (Lightning, data, utils, STFT)
│   ├── blocks/              # NN building blocks
│   ├── dataset/             # Dataset IO, chunking, caching
│   ├── loss/                # Shared losses
│   ├── model/               # Shared model helpers
│   └── utils/               # Logging, metrics, signal utils
├── examples/               # Dataset-specific configs and scripts
├── scripts/                # Launch and environment helpers
├── tools/                  # Evaluation and data prep utilities
└── test/                   # Smoke and unit tests
```

## Core Logic
1) Dataset pipeline (modules/system.py, modules/dataset/dataset_base.py)
   - Config is loaded via OmegaConf and merged with CLI overrides.
   - DatasetBase resolves data paths, reads CSVs or HDF5 stores, and builds chunk lists based on STFT hop/window and chunk_size.
   - Optional mixing (noise/RIR), filtering by duration, and caching are applied before dataloaders are created.
   - BaseModule prepares data, sets augmentations, and constructs train/valid/test loaders depending on dataset type (iterable vs map).

2) Speech enhancement training/inference (alpha/enh/system/system.py)
   - BaseSE builds STFT, model, loss, and optional deepfilter configuration from the config.
   - Forward passes operate on STFT magnitudes/phases and apply masking or filtering, then ISTFT to waveform.
   - Training uses Lightning hooks for loss logging; inference uses `denoise()` with chunked processing for long audio.

3) ASR and joint Enh+ASR training (alpha/asr/system.py, alpha/enh_asr/system.py)
   - ASR: features are extracted (FeatExtract) then passed to the ASR model, loss computes token accuracy and loss stats.
   - Joint Enh+ASR: BaseEnhASR uses manual optimization with two optimizers. It first optimizes enhancement loss, then runs ASR on enhanced audio features and optimizes ASR loss.
   - Checkpoint loading differs by task: enhancement expects Lightning `state_dict`, ASR expects `model` keys.

## Coding Conventions
- Naming: classes use PascalCase, functions and variables use snake_case.
- Imports: standard library, third-party, then internal modules.
- Formatting: 4-space indentation, minimal type hints, single quotes for strings are common.
- Comments: sparse, used only to clarify non-obvious behavior.
- Logging: loguru `logger` with consistent key prefixes (train/, valid/).

## Interaction Rules
- Keep config schema stable: if you add model or loss options, update the corresponding config files in examples/ and any OmegaConf loaders.
- System name routing is config-driven: ensure new systems are registered in their package (alpha/*/system.py) and referenced by `system.name` in configs.
- For joint Enh+ASR, preserve manual optimization with two optimizers and keep loss logging keys consistent (`train/enh_loss`, `train/asr_loss`, `train/acc`).
- Respect checkpoint formats: enhancement expects Lightning `state_dict`, ASR expects `model`. If formats change, update `init_model()` and any inference scripts.
- Data pipeline relies on `noisy/clean/noise` keys and chunking logic. If you change dataset fields, update DatasetBase and its CSV/HDF5 readers.
- When adding new augmentation or feature extractors, ensure they are wired into BaseModule so training and inference remain consistent.
- If you modify STFT parameters (fft/hop/window), update both dataset chunking and any model assumptions that depend on feature dimensions.
