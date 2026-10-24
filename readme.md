# AlphaSNN: Alpha Spiking Neural Network

## 1. NVIDIA

### 1.1 Create conda environment

```bash
conda create -n speech python=3.10
conda activate speech
```

### 1.2 Install requirements

- numpy==1.26.4
- torch==2.1.0
- lightning==2.2.5
- librosa==0.10.2
- onnxruntime-gpu==1.15.0
- h5py==3.11

```bash
pip install numpy==1.26.4 #默认会安装2.0
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia # 在香农和图灵上
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.2.5 librosa==0.10.2 onnxruntime-gpu==1.15.0 h5py==3.11
pip install ipython pandas matplotlib omegaconf rich pytest seaborn asteroid_filterbanks
pip install pystoi nvitop thop tensorboard einops loguru editdistance joblib whisper
pip install pesq # 如果报错，则根据提示加上 --use-pep517
pip install kaldiio matcha-tts torchtyping torch_ema ninja
```

cpu 版本

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
```

### 1.3 Test pytorch

```bash
python test/test_pytorch.py
```

output:

```bash
PyTorch: 2.1.0
Tesla P100-PCIE-16GB
tensor([1.], device='cuda:0')
cudnn: True
```

## 2. Sugon slurm DCU

### 2.1 Create conda environment

```bash
conda create -n speech python=3.10 openssl=1.1.1
conda activate speech
```

### 2.2 Install requirements

- numpy==1.26.4
- torch==2.1
- pytorch-lightning==2.2.5
- librosa==0.10.2
- onnxruntime-gpu==1.15.0
- h5py==3.11

```bash
# login 节点
pip install numpy==1.26.4 #默认会安装2.0
pip install pip==24.0 #更高版本pip以下会报错
pip install /public/software/apps/DeepLearning/whl/dtk-24.04/pytorch/torch2.1/py310/torch*
pip install /public/software/apps/DeepLearning/whl/dtk-24.04/onnxruntime/onnxruntime-1.15.0+das1.0+gita9ca438.abi0.dtk2404-cp310-cp310-manylinux2014_x86_64.whl
pip install pytorch-lightning==2.2.5 librosa==0.10.2 h5py==3.11
pip install ipython pandas matplotlib omegaconf rich pytest seaborn asteroid_filterbanks
pip install pesq pystoi nvitop thop tensorboard einops loguru editdistance joblib whisper
```

### 2.3 Test pytorch

- 加载模块、激活 conda 环境

```bash
. scripts/path.sh
```

- debug 节点

```bash
python test/test_pytorch.py
```

output:

```bash
Hostname: d4-hpc-debuggpu-007
PyTorch: 2.1.0
Vega 20
tensor([1.], device='cuda:0')
cudnn: True
```

- login 节点

```bash
srun -p kshdnormal --gres=dcu:1 python test/test_pytorch.py
```

output:

```bash
srun: job 66008560 queued and waiting for resources
srun: job 66008560 has been allocated resources
srun: ROUTE: split_hostlist: hl=b01r4n06 tree_width 0
Hostname: b01r4n06
PyTorch: 2.1.0
Device 66a1
tensor([1.], device='cuda:0')
cudnn: True
```

## 3. Sugon slurm GPU

### 3.1 Load modules
module load nvidia/cuda/11.8

### 3.2 Create conda environment

```bash
conda create -n speech python=3.10
conda activate speech
```

### 3.3 Install requirements

- numpy==1.26.4
- torch==2.1.0
- pytorch-lightning==2.2.5
- librosa==0.10.2
- onnxruntime-gpu==1.15.0
- h5py==3.11

```bash
conda install numpy==1.26.4 # 默认会安装2.0
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning==2.2.5 librosa==0.10.2 onnxruntime-gpu==1.15.0 h5py==3.11
pip install ipython pandas matplotlib omegaconf rich pytest seaborn asteroid_filterbanks
pip install pystoi nvitop thop tensorboard einops loguru editdistance joblib whisper
pip install pesq # 如果报错，则根据提示加上 --use-pep517
```

### 3.3 Test pytorch

```bash
srun -p qdagnormal --gres=gpu:1 python test/test_pytorch.py
```

output:

```bash
Hostname: gpu1
PyTorch: 2.1.0
NVIDIA A800 80GB PCIe LC
tensor([1.], device='cuda:0')
cudnn: True
```

## 4. Dataset of SHD, SSC and GSC

当前版本使用 `data.name: SpikeDataset` 和 `data.spike.name: shd|ssc|gsc`
选择具体数据集。SHD 与 SSC 会复用相同的数据准备流程，自动下载、校验 MD5、
解压并在数据根目录下构建 `events_h5`。

### 4.1 手动下载 SHD / SSC

如果希望手动下载 SHD 或 SSC，可以使用如下命令：

```bash
wget https://zenkelab.org/datasets/shd_train.h5.zip
wget https://zenkelab.org/datasets/shd_test.h5.zip

wget https://zenkelab.org/datasets/ssc_train.h5.zip
wget https://zenkelab.org/datasets/ssc_valid.h5.zip
wget https://zenkelab.org/datasets/ssc_test.h5.zip
```

ZenkeLab 官方 MD5 清单：
https://zenkelab.org/datasets/md5sums.txt

### 4.2 手动下载 GSC

GSC 使用 Torchaudio 的 SpeechCommands v0.02 数据集。默认配置
`examples/GSC/conf/gsc_baseline.yaml` 会将 GSC 根目录解析为
`examples/GSC/data/gsc`。

如果计算节点无法访问外网，可以先在可联网节点手动下载归档：

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p examples/GSC/data/gsc
wget -c -O examples/GSC/data/gsc/speech_commands_v0.02.tar.gz \
  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

printf '%s  %s\n' \
  af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58 \
  examples/GSC/data/gsc/speech_commands_v0.02.tar.gz \
  | sha256sum -c -
```

以上命令应从仓库根目录执行；第一行会自动切换到仓库根目录。归档约为
2.26 GiB，`wget -c` 支持断点续传。这里使用 Torchaudio 2.1.0 内置的官方
HTTP 下载地址，并使用其内置 SHA-256 值校验文件完整性。

如果使用 HTTPS 地址时出现 `no alternative certificate subject name matches`
或“没有匹配的证书主体别名”，说明当前代理、网关或证书环境返回的证书与
`download.tensorflow.org` 不匹配。不要直接使用 `--no-check-certificate`；
优先使用上面的 HTTP 地址并确保 SHA-256 校验通过，或在证书环境正常的机器
下载并传输归档。

此时保持配置中的 `data.gsc.download: true`。Torchaudio 检测到本地归档后，
不会重复下载，只会将其解压到正确目录。

也可以提前手动解压，并关闭 Torchaudio 的下载与解压流程：

```bash
mkdir -p examples/GSC/data/gsc/SpeechCommands/speech_commands_v0.02
tar -xzf examples/GSC/data/gsc/speech_commands_v0.02.tar.gz \
  -C examples/GSC/data/gsc/SpeechCommands/speech_commands_v0.02
```

手动解压后，将 `examples/GSC/conf/gsc_baseline.yaml` 修改为：

```yaml
data:
  gsc:
    download: false
```

Torchaudio 最终要求以下目录存在：

```text
examples/GSC/data/gsc/
├── speech_commands_v0.02.tar.gz
└── SpeechCommands/
    └── speech_commands_v0.02/
        ├── validation_list.txt
        ├── testing_list.txt
        ├── yes/
        ├── no/
        └── ...
```

如果修改了 `data.data_dir` 或各数据集的 `root`，请将下载目录同步调整为
`<data_dir>/<root>`。

### 4.3 SHD / SSC 参考目录结构

```txt
data
└── ssc / shd
    ├── download
    ├── duration_10
    ├── events_h5
    └── extract
```

## 5. Dataset of car_command
离线学习训练集路径：/home/huxixiang20040727/AlphaSNN/examples/carKWS_offline/data
```txt
carKWS_offline/data： command_feats_test测试集FBANK特征
                      command_feats_train训练集FBANK特征
                      command_cuts_test_29.jsonl测试集.jsonl文件，类似于csv文件
                      command_cuts_train_29.jsonl训练集.jsonl文件，类似于csv文件
    
```
## Baseline Replication

基于论文，已经跑通了所有数据集的baseline，参考数值如下。
_shd没有测试集，所以以验证集数据作为替代_

|     | Baseline | Replication | Gap    |
| --- | -------- | ----------- | ------ |
| GSC | 96.92%   | 96.57%      | -0.35% |
| SSC | 83.26%   | 83.10%      | -0.16% |
| SHD | 96.41%   | 96.3%       | -0.11% |


