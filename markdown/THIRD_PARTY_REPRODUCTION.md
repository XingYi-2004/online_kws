# AlphaSNN-KWS（原 SNN_aol）第三方复现与部署说明

> **⚠️ 2026-10-24 目录迁移**：项目已整体迁入 AlphaSNN 框架（SNN 与 SNN_aol 撤销）。
> 文中旧路径按以下映射读取：
> - `SNN_aol/offline/` → `AlphaSNN/alpha/asr/kws/offline/`
> - `SNN_aol/online/` → `AlphaSNN/alpha/asr/kws/online/`
> - `online/config/da_aol.yaml` → `alpha/asr/kws/conf/online_adakws.yaml`（entropy_threshold 已改回冠军值 0.3）
> - `offline/conf/multitau_29class_kd.yaml` → `alpha/asr/kws/conf/offline_kd.yaml`
> - `SNN/data/` → `AlphaSNN/data_source/`；`SNN_aol/{data,data_train,kd_teacher,alphasnn_29class*,scripts}` → `AlphaSNN/` 下同名目录
>
> 正文中的目录树与历史描述保留迁移前原貌；启动命令与数据路径均已更新为迁移后位置。


> 本文是面向第三方的当前代码复现手册：说明任务、数据、ANN 教师、SNN 蒸馏、无标签在线 AdaKWS、运行命令、checkpoint 和部署接口。
>
> **本文与历史实验文档的关系**：根目录 `README.md`、`ONLINE_LEARNING_HANDOFF.md` 保留了 DA/CDAN/SPACE 等完整历史实验记录；其中部分目录树/历史命令对应清理前代码。本文描述当前 checkout 中仍存在的代码和当前推荐配置。
>
> **证据标记**：
> - **已验证**：本地当前代码/指定 checkpoint 已实际运行并得到记录值。
> - **日志还原**：由训练/解码日志或 checkpoint 元数据确认；原实现可能未完整保留。
> - **重建版**：原 SNN 离线训练脚本误删后，根据日志与 checkpoint 结构重建的脚本。
> - **未验证**：明确标出，不将推测描述为已完成结果。

---

## 1. 项目目标与最终推荐实验

任务：**29 类中文车载语音指令分类**，比较 ANN 教师、脉冲 SNN 学生，并用无标签目标域语音执行测试时适应。

```text
29 类有标签车载语音
       │
       ├── ANN 教师：TdnnLstm
       │
       └── ANN 软输出 + 监督标签
                    ↓ 知识蒸馏（KD）
             离线 MultiTauLIFCommander SNN
                    ↓ 无标签 AdaKWS 测试时适应
             目标域适应后的 SNN
```

| 阶段 | 模型/数据 | 已验证结果 | 结果文件/证据 |
|---|---|---:|---|
| ANN 教师 | TdnnLstm，29 类 | **99.72%** | 2026-07-09 decode 日志；epoch 9–17 平均后导出 `pretrained.pt` |
| 离线 SNN raw | MultiTauLIFCommander experiment4，干净测试 357 条 | **98.04%** | `best-acc.pt`，E224；固定评测脚本实测 350/357 |
| 离线 SNN EMA | experiment4 EMA，干净测试 357 条 | **98.04%** | `best-ema.pt`，E253；固定评测脚本实测 350/357 |
| 在线起点 | experiment4 `best-ema.pt`，无 RIR 目标域 1071 条 | **87.86%** | `online/train.log/2026-10-19_15-24-45/` |
| 在线 AdaKWS 后 | 同一目标域 | **90.20%** | 同日志，真实车载 100.00%、加噪 simu 84.14%；复跑逐位一致 |

### 历史结果与当前结果不可混为一谈

- **KD-MT-08（2026-08-04）**：历史 checkpoint/日志记录 98.04%，但原版离线 SNN 训练代码于 2026-10-08 清理事故中丢失。不能声称当前重建脚本逐行复现了原始训练过程。
- **experiment4（2026-10-16）**：由当前重建版 `offline/train_multitau.py` 训练。raw 和 EMA 均 98.04%，当前 `offline/eval_multitau.py` 能以固定口径逐位复现。这是当前推荐的离线 checkpoint。
- **旧在线冠军 90.10%**：KD-MT-08 raw checkpoint 起步，AdaKWS lr=3e-3。
- **当前在线冠军 90.20%**：experiment4 EMA checkpoint 起步，AdaKWS lr=2.2e-3。起点和超参不同，比较时应同时注明。

---

## 2. 仓库目录树与文件职责

以下目录树从 Git 工作区根目录 `online_kws/` 开始。`[外部数据/权重]` 表示文件存在于运行机器，但被 `.gitignore` 排除，不随 Git clone 交付。

```text
online_kws/
├── .gitignore                         # 排除数据集、缓存和大部分 checkpoint
├── SNN/                               # ANN 教师代码及旧数据管线
│   ├── model_ann.py                   # TdnnLstm ANN 教师结构
│   ├── train_ann.py                   # ANN 教师训练入口（CE/AdamW）
│   ├── decode_ann.py                  # ANN checkpoint 平均、评测及导出
│   ├── asr_datamodule.py              # Lhotse manifests、FBank、SpecAugment、Sampler
│   ├── run_ann.sh                     # ANN 快捷命令；GPU id/绝对路径需按机器修改
│   ├── test.sh                         # 调用 decode_ann.py 的旧 ANN 评测脚本
│   ├── data/fbank/                    # [外部数据] ANN manifests、FBank 特征和索引
│   ├── tdnn_lstm_6ann_768_new2/exp/
│   │   ├── pretrained.pt             # [外部权重] epoch 9–17 平均的 ANN 教师，约123MB
│   │   ├── epoch-*.pt                # [外部权重] ANN 逐 epoch 检查点
│   │   └── log/                      # ANN 训练与 decode 日志
│   ├── datalarge/fbank/              # [外部数据] 另一个特征目录；当前主流程未引用
│
└── SNN_aol/
    ├── AlphaSNN/                     # MultiTauLIFCommander 源码依赖
    │   ├── alpha/asr/model/multitau_lif_commander/ # MultiTau 主干/Block/LIF/mixer/pooling
    │   └── alpha/asr/loss/spiking_loss.py         # SpikingCELoss 等损失
    ├── data/fbank/                   # [外部数据] SNN KD 训练/干净评测 manifests 和 FBank
    ├── data_train/fbank/              # [外部数据] 在线目标域 manifests 和 FBank
    ├── kd_teacher/tdnn_ann_29class.pt # [外部权重] ANN 蒸馏教师副本，约123MB
    ├── alphasnn_29class_kd_rebuild_v2/experiment4/
    │   ├── best-acc.pt               # [外部权重] raw 98.04%，E224
    │   ├── best-ema.pt               # [外部权重] EMA 98.04%，E253
    │   └── log-train-.rebuild        # 运行时配置快照、300 epoch 指标
    ├── offline/
    │   ├── train_multitau.py         # 重建版离线 SNN KD + EMA 训练入口
    │   ├── kd_teacher.py             # TdnnLstm 教师结构定义
    │   ├── eval_multitau.py          # 固定评测 experiment4 raw/EMA，无参数
    │   ├── conf/multitau_29class_kd.yaml # SNN KD 配置（当前 exp_dir=experiment5）
    ├── online/
    │   ├── train_adakws.py           # 在线 AdaKWS 入口：加载→初测→适应→终测
    │   ├── model_kws.py              # 下采样 + MultiTau 主干封装 + lengths mask
    │   ├── config/da_aol.yaml        # 当前推荐配置：experiment4 EMA、lr=2.2e-3
    │   ├── trainers/
    │   │   ├── online_trainer.py     # 在线方法入口/分流器
    │   │   └── adakws_trainer.py     # 熵/PKC 筛选、损失、BN 更新、best 选择
    │   ├── utils/
    │   │   ├── config.py             # YAML、路径、时间戳实验目录
    │   │   ├── data.py               # Lhotse cuts、Dataset、AdaKWS loader
    │   │   ├── build_replay.py       # 旧回放库构造工具；当前纯 AdaKWS 流程不调用
    │   │   └── train_utils.py        # 权重键映射、标签、整体/子集精度
    │   └── train.log/                # 在线运行日志和 checkpoint
    │       └── 2026-10-19_15-24-45/  # [外部产物] 当前 90.20% run 与 best-adakws.pt
    └── scripts/measure_efficiency.py # 模型效率统计脚本（不属于训练主流程）
```

---

## 3. 软件环境与前置条件

当前验证环境：

| 组件 | 已验证版本/状态 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| Lhotse | 1.33.0 |
| SpikingJelly | 0.0.0.0.14（项目源码/环境安装） |
| OmegaConf | 2.3.1 |
| NumPy | 2.2.5；入口处理旧 `np.int` 兼容 |
| PyYAML | 6.0.3 |
| CuPy | 14.1.1 |
| icefall / k2 | 项目使用本地/源码依赖；没有完整 pip lock 文件 |

```bash
conda activate snn
python -c "import torch, torchaudio, lhotse, spikingjelly; print(torch.__version__, torchaudio.__version__, lhotse.__version__)"
```

**第三方环境交付注意**：当前仓库无完整 `environment.yml`/依赖锁定文件。需要另外确认 icefall/k2 的安装来源、CUDA 驱动、CuPy 与 CUDA runtime 配套。仅安装 `AlphaSNN/requirements.txt` 不足以覆盖 ANN/在线/离线所有依赖。

---

## 4. 数据、特征、标签

### 4.1 离线训练数据

目录：`AlphaSNN/data/fbank/`。

| Manifest/资源 | 规模 | 用途 |
|---|---:|---|
| `command_cuts_train_29.jsonl` | 128,076 条 | SNN KD 训练集 |
| `command_cuts_test_29.jsonl` | 357 条 | 干净测试/离线评测集 |
| `command_feats_*`、`.dat`、`.idx` | 预计算 FBank 和索引 | Lhotse 数据读取 |

项目记录的训练集为人工加噪语音（SNR 2–15 dB，平均约 7.3 dB），测试集 357 条为干净语音；是“噪声训练 → 干净评测”的跨域设置。

### 4.2 在线目标域

当前 online YAML 使用 `AlphaSNN/data_train/fbank/command_cuts_test_29_no_rir.jsonl`：

```text
1071 条 = 409 条真实车载 + 662 条加性噪声 simu（不含 RIR）
```

标准目标域有 1428 条（另含 357 条 RIR 远场混响）；当前 90.20% 结果只针对 **1071 条 no-RIR 子集**，不可与 1428 条标准域数字混用。

目标 manifest 确实包含 `speaker/text` 真值字段，但在线适应代码只使用 `inputs`、`num_frames/duration` 和模型伪标签；真值只进入初始/最终 `compute_test_accuracy` 统计。**不得将目标真标签用于筛样本、计算 loss 或挑 best iteration。**

### 4.3 特征和类别表

- FBank：80 bins，项目预计算特征 frame shift 为 10 ms；online 默认 `on_the_fly_feats=false`，从预计算数据读取。
- manifest 通过 `speaker` 存类别 id，通过 `text` 存中文类名。
- 类别 id 0–28：

| id | 类别 | id | 类别 | id | 类别 |
|---:|---|---:|---|---:|---|
| 0 | 上一台 | 1 | 关闭路况 | 2 | 打开内循环 |
| 3 | 切换AM | 4 | 切换FM | 5 | 切换到主路 |
| 6 | 切换到辅路 | 7 | 切换到音乐 | 8 | 加入收藏 |
| 9 | 打开外循环 | 10 | 少收费 | 11 | 屏幕亮一点 |
| 12 | 我要重拨 | 13 | 打开路况 | 14 | 打开广播 |
| 15 | 打开收藏 | 16 | 打开歌词 | 17 | 打开氛围灯 |
| 18 | 打开电台列表 | 19 | 查看联系人 | 20 | 扫描电台 |
| 21 | 拨打救援电话 | 22 | 接听电话 | 23 | 搜索电台 |
| 24 | 播放电台 | 25 | 显示歌词 | 26 | 暂停电台 |
| 27 | 第六个 | 28 | 黑夜模式 | | |

### 4.4 数据如何交付

大数据目录与 checkpoint 被 `.gitignore` 排除，不随普通 Git clone 下载。接收方需要单独获得：

1. `AlphaSNN/data/fbank/`、`AlphaSNN/data_train/fbank/` manifests 和特征文件/索引。
2. `AlphaSNN/kd_teacher/tdnn_ann_29class.pt`（约 123 MB）。
3. `AlphaSNN/alphasnn_29class_kd_rebuild_v2/experiment4/best-acc.pt`、`best-ema.pt`。

原始音频授权/下载地址与构造当前所有 29 类子集所需的完整原始数据流程没有在当前 checkout 中完整记录；不能假设第三方仅凭仓库脚本就能得到完全相同语料。

---

## 5. ANN 教师训练（TdnnLstm）

### 5.1 架构

源码：`SNN/model_ann.py` 的 `TdnnLstm`，参数量 32,096,285（约 32.1M）。输入 `[B,80,T]`：

```text
Conv1d 80→768, k=3, stride=1 → ReLU → BatchNorm1d(affine=False)
Conv1d 768→768, k=3, stride=1 → ReLU → BatchNorm1d(affine=False)
Conv1d 768→768, k=3, stride=3 → ReLU → BatchNorm1d(affine=False)
转为 [T,B,768]
6 × (单层 LSTM 768→768 → BatchNorm1d(affine=False) → Dropout(0.1) → residual add)
时间均值池化 → Linear 768→29 → logits [B,29]
```

这是整句 29 类分类。`train_ann.py:318–320` 实际使用 `CrossEntropyLoss`；代码注释中遗留的 CTC/ASR 通用内容不是本任务的训练损失。

### 5.2 ANN 训练参数及证据

| 参数 | 值 | 来源 |
|---|---:|---|
| 类别数 | 29 | `--num-class 29`，训练日志 |
| 输入 FBank | 80 维 | `feature_dim=80` |
| hidden / LSTM 层数 | 768 / 6 | `model_ann.py` |
| subsampling_factor | 3 | train/decode params |
| optimizer | AdamW | `train_ann.py` |
| lr / weight decay | 2e-4 / 5e-4 | `get_params()` 与训练日志 |
| epoch | 20 | `run_ann.sh` |
| LR scheduler | Linear warmup 3 epoch（0.1→1.0）+ cosine，eta_min=1e-6 | 训练代码 |
| batch sampler | DynamicBucketing，max_duration=200s，num_buckets=30 | CLI 与 datamodule |
| SpecAugment | enabled；time-warp factor=80；本次日志报告 time frame masks=10；feature masks=2、宽度 27、frames mask size=100 | `asr_datamodule.py` 与训练日志 |
| MUSAN | disabled | 训练日志 |
| seed | 42；每 epoch 使用 seed+epoch | 训练代码 |

#### 验证集使用限制

当前 `asr_datamodule.py` 的 `valid_cuts()` 和 `test_cuts()` 都读取 `command_cuts_test_29.jsonl.gz`。也就是说训练过程中反复验证的 357 条与 `decode_ann.py` 报告测试精度所用清单相同；99.72% 不是对一个完全独立、从未用于 epoch 观察的 holdout 的估计。第三方报告结果时必须保留这一限制说明。

#### 当前 ANN 教师的日志记录

- 训练日志：`AlphaSNN/data_source/tdnn_lstm_6ann_768_new2/exp/log/log-train-2026-06-26-16-23-48`
- 导出日志：`AlphaSNN/data_source/tdnn_lstm_6ann_768_new2/exp/log/log-decode-2026-07-09-18-47-56`
- decode 配置：`epoch=17, avg=9, num_class=29, subsampling_factor=3, max_duration=200, num_buckets=30`
- averaging 的 checkpoint：epoch 9–17
- 导出结果测试精度：0.9971988796（约 99.72%），文件 `tdnn_lstm_6ann_768_new2/exp/pretrained.pt`

### 5.3 ANN 数据准备状态

旧的 `SNN/prepare.sh` 已删除：它依赖当前 checkout 不存在的 `shared/parse_options.sh`，不能作为可直接运行的准备入口。`SNN/local/compute_fbank_command.py` 仍保留，但它读取 `data/manifests/` 中已有的 Command manifests 并计算 FBank；它本身不负责下载音频或生成这些 manifests。

因此第三方若要复现已验证的 ANN 教师训练，应取得现成的 `SNN/data/fbank/` 清单和特征；若要从原始语音重新生成，项目提供方还需另行提供原始音频、Command recipe/manifest 生成流程和授权信息。只拿当前 Git 仓库不能完整从原始音频重建数据。

### 5.4 训练和导出 ANN 教师

`SNN/run_ann.sh` 固定 GPU 4；第三方更建议显式选设备和 manifest：

```bash
cd online_kws/SNN
conda activate snn
export CUDA_VISIBLE_DEVICES=0

python train_ann.py \
  --world-size 1 \
  --num-epochs 20 \
  --num-class 29 \
  --max-duration 200 \
  --manifest-dir /path/to/SNN/data/fbank
```

导出 epoch 9–17 平均教师：

```bash
python decode_ann.py \
  --epoch 17 --avg 9 --num-class 29 \
  --manifest-dir /path/to/SNN/data/fbank \
  --export True
```

检查 `AlphaSNN/data_source/tdnn_lstm_6ann_768_new2/exp/pretrained.pt` 是否生成，并检查 decode 日志中的 `Test set accuracy`。SNN KD 默认读取的是 `AlphaSNN/kd_teacher/tdnn_ann_29class.pt`；如果自行重训 ANN，需要将新 teacher 指向/复制到此配置路径。

---

## 6. 离线 SNN 知识蒸馏

### 6.1 MultiTauLIFCommander 学生架构

主干在 `AlphaSNN/alpha/asr/model/multitau_lif_commander/multitau_lif_commander.py`，训练 wrapper `AlphaSNNForKWS` 在 `offline/train_multitau.py`：

```text
FBank [B,80,T]
  → AvgPool1d(k=stride=2)
  → SEELiteFrontend
  → 5 × MultiTauBlock
       context norm
       → 3 个时间常数的 LIF memory（taus 初始化 [2,4,8]，learnable_alpha=true）
       → channel_adaptive 分支融合
       → bottleneck temporal conv mixer（conv）+ residual
  → masked temporal mean pool（忽略 padding）
  → Linear(384→29)
  → logits [B,29]
```

experiment4 学生参数量 660,111（约 0.660M）。每个 batch 前后通过 `functional.reset_net(model)` 清 SNN 膜电位；不同 utterance/batch 不能串用膜电位。

### 6.2 知识蒸馏方法

teacher 是冻结的 ANN `TdnnLstm`；student 学习标签 CE 和 teacher 的 soft logits。当前重建训练代码：

```text
L_total = (1 - α) · L_SpikingCE + α · T² · KL(p_teacher || p_student)
α=0.7，T=10.0
```

代码实现使用 `F.kl_div(log_softmax(student/T), softmax(teacher/T)) * T²`，即以 teacher probability 为 target、student log-probability 为 input。`SpikingCELoss` 实际封装 CrossEntropyLoss，experiment4 的 label smoothing=0.1。Teacher eval/frozen，不反向更新。

### 6.3 experiment4 的实际训练配置（日志确认）

来源：`alphasnn_29class_kd_rebuild_v2/experiment4/log-train-.rebuild` 的启动配置快照。

| 配置项 | experiment4 值 |
|---|---:|
| train/test manifests | `command_cuts_train_29.jsonl` / `command_cuts_test_29.jsonl` |
| 训练条数 / clean test 条数 | 128,076 / 357 |
| model | input=80, hidden=384, output=29, depths=5, params=0.660M |
| taus / learnable alpha | `[2,4,8]` / `true` |
| memory / fusion | soft / channel_adaptive |
| frontend / mixer | see_lite / conv |
| mixer | bottleneck ratio=4, kernel=5, dilation=1, norm groups=8 |
| subsampling / dropout | 2 / 0.3 |
| LIF | threshold=1.0, ATan surrogate alpha=2.0, detach_reset=true, decay_input=true |
| teacher | `kd_teacher/tdnn_ann_29class.pt`，32,096,285 参数 |
| KD alpha / T | 0.7 / 10.0 |
| student loss | SpikingCE + label smoothing 0.1 |
| SpecAugment | time width≤15 ×1；frequency width≤5 ×1；只在训练启用 |
| optimizer | AdamW，lr=2e-3，weight_decay=5e-3 |
| LR | linear warmup 30 epoch + cosine `T_max=270` |
| epochs / seed | 300 / 312 |
| gradient clipping / branch balance | 5.0 / 0.01 |
| EMA | decay=0.9995 |
| train 和 test loader | DynamicBucketing，max_duration=1500s，num_buckets=30，drop_last=false |
| num_workers | 8 |

**原版和重建版必须区分**：原始 offline SNN 训练代码于 2026-10-08 清理中误删；当前 `train_multitau.py` 是依据日志和 checkpoint 结构重建。上表是 experiment4 实际记录，不代表已证明与 KD-MT-08 原版的每个细节完全相同。当前 `offline/conf/multitau_29class_kd.yaml` 的输出目录指向 experiment5；新训练前应改到新的实验目录，不覆盖 experiment4。

### 6.4 离线训练命令和产物

先确认 `offline/conf/multitau_29class_kd.yaml` 中 GPU、manifests、teacher 权重和新的 `exp_dir`：

```bash
cd online_kws/AlphaSNN
conda activate snn
python alpha/asr/kws/offline/train_multitau.py --conf alpha/asr/kws/conf/offline_kd.yaml
```

每轮对指定 clean test manifest 评测 raw 与 EMA；保存：

| 文件 | 内容 |
|---|---|
| `best-acc.pt` | raw student 最佳权重 + `best_test_acc/config/epoch` |
| `best-ema.pt` | EMA student 最佳权重 + `best_test_acc/is_ema/config/epoch` |
| `last.pt` | 最后一 epoch raw 权重与记录值 |
| `log-train-.rebuild` | 启动完整配置和逐 epoch 指标 |

### 6.5 固定评测命令

```bash
cd online_kws/AlphaSNN
conda activate snn
python alpha/asr/kws/offline/eval_multitau.py
```

脚本无 CLI 参数，固定评测 experiment4 raw+EMA，结果：

```text
[raw] 350/357 = 98.04%（E224）
[EMA] 350/357 = 98.04%（E253）
```

**评测 loader 必须与训练 test loader 一致**：seed=312，Lhotse DynamicBucketing `max_duration=1500s`、`num_buckets=30`、`drop_last=False`。本机 Lhotse 1.33.0 默认 `num_buckets=10`；少传 30 会改变 batch 组成/补零，并实际得到 raw 96.08%、EMA 97.48%。脚本中的 30 不是随意可改的性能参数。

训练每个 epoch 都检查同一 357 条集并保存历史最佳，故 98.04% 是反复观察这份评测集后选择出的 best，不是完全独立 untouched holdout；第三方报告应注明。

---

## 7. 在线 AdaKWS 测试时适应

### 7.1 目标域和当前完整配置

当前配置 `online/config/da_aol.yaml` 使用 experiment4 **EMA** 权重和 no-RIR 1071 条目标集：

| 项 | 当前值 |
|---|---|
| pretrained | `experiment4/best-ema.pt`（clean test 98.04%，E253） |
| target manifest | `command_cuts_test_29_no_rir.jsonl`（409 real-car + 662 addsimu） |
| seed / iterations | 42 / 10 |
| batch / steps | 128 条 / 每 batch 更新 1 步 |
| optimizer / lr / wd | Adam / 2.2e-3 / 0 |
| entropy threshold | 0.3 |
| PKC/PLPD threshold | 0.05 |
| sigma / temperature | 0.5 / 1.0 |
| BN mode / update scope | `train_running` / `bn_affine` |
| 可训练 affine 参数 | 4,608 个（日志确认） |
| mask | time ≤20 帧 ×2；freq ≤5 bins ×2 |
| grad clip | 5.0 |

在线模型配置块中的一些构造初值与 experiment4 训练时 YAML 有历史差异；当前运行日志确认该 checkpoint 通过 `model.* → backbone.*` 成功加载 99/99 个参数。模型形状/前向结构必须保持一致；不要凭本文随意改 `model:` 块。

### 7.2 AdaKWS 原理与单批流程

论文：Yang Xiao et al., *AdaKWS: Towards Robust Keyword Spotting with Test-Time Adaptation*, arXiv:2505.14600 / Interspeech 2025。论文针对小型关键词识别模型在噪声、远场等测试分布变化下精度下降的问题，提出测试时适应（TTA）：适应阶段不访问源域训练数据和目标真标签，只根据目标输入调整预训练模型。

本文所称“无标签”是指适应算法不读取目标标签。manifest 虽包含 `speaker/text`，这些字段只用于适应前后的 accuracy 报告；真实目标标签不进入伪标签、样本筛选、损失、梯度或最佳轮次选择。

AdaKWS 每批数据包括三个关键步骤：先用预测熵筛出较可信的样本，再用 PKC 检查模型预测是否依赖输入语音内容，最后对通过筛选的样本做加权熵最小化。

#### 第一步：熵筛选——排除模型自己也拿不准的样本

对目标语音 `x`，模型输出 29 类概率：

```text
p = [p(类别0|x), p(类别1|x), ..., p(类别28|x)]
伪标签 c = 概率最大的类别 argmax(p)
```

预测熵用普通求和公式表示为：

```text
entropy(x) = - Σ_j [ p_j(x) × log(p_j(x)) ]
```

概率集中在某一类时，entropy 较低，模型比较确定；概率分散时，entropy 较高，模型比较犹豫。AdaKWS 只保留：

```text
entropy(x) < entropy_threshold
```

本项目阈值是 0.3。低熵只说明模型对自己的预测确定，并不保证预测一定正确，所以还需要第二道筛选。

#### 第二步：PKC（Pseudo-Keyword Consistency）——检查预测是否依赖语音内容

对熵筛选留下的特征做时间/频率遮罩，得到 `x_masked`。随后比较模型对同一个原伪标签 `c` 的概率：

```text
原始概率 = p(c | x)
遮罩概率 = p(c | x_masked)
probability_drop = 原始概率 - 遮罩概率
```

本项目保留条件为：

```text
probability_drop > pkc_threshold
```

例子：原始输入对类别 c 的概率是 0.90，遮罩后是 0.72，那么 probability_drop = 0.90 - 0.72 = 0.18；阈值为 0.05 时该样本通过 PKC。若遮罩后概率几乎不变，样本就被排除。

论文用 PKC 作为第二种关键/可靠样本筛选，并解释时间遮罩可模拟语音中断、频率遮罩可模拟频谱变化。论文正文有一句关于“小概率下降表示稳定”的文字与公式和筛选不等式方向不完全一致；本项目遵循公式及实际代码：`drop = 原始概率 - 遮罩概率`，并要求 `drop > threshold`。这里的 PKC 是筛样本，不是对两个视图直接施加一致性损失。

#### 第三步：按样本权重做熵最小化

对通过两道筛选的样本，代码使用：

```text
weight(x) = exp(sigma - entropy(x)) + exp(probability_drop(x))
```

熵越低，第一项越大；原伪标签概率被遮罩拉低得越多，第二项越大。最终损失为：

```text
loss = 所有通过筛选的样本的 [weight(x) × entropy(x)] 的平均值
```

最小化该损失会让被选中样本的预测分布更集中；样本权重让低熵、且 PKC 认为与语音内容相关的样本对更新影响更大。遮罩输入的前向只用于计算筛选值，不单独反向；梯度来自原始输入上的加权熵损失。

#### 单批流程总结

```text
目标 batch FBank（适应时不读取真标签）
  → 原始输入前向 → 概率 + 模型伪标签
  → 熵筛选（本配置 < 0.3）
  → 通过者做时频遮罩，再前向计算 probability_drop
  → PKC 筛选（本配置 > 0.05）
  → 计算 weight × entropy 并平均得到 loss
  → 反向传播，只更新 BN affine（本配置 4,608 个参数）
```

#### 论文方法与本项目实现的差别

论文实验使用 BC-ResNet-3 和 40 维 MFCC，在 GSC 及多种噪声上测试；论文设置包括 batch size 128、熵阈值 0.4、sigma 0.5、PKC 阈值 0.05，以及两个最大宽度为 20 的时间遮罩和两个最大宽度为 5 的频率遮罩。

本项目将方法迁移到 MultiTauLIFCommander SNN + 80-bin FBank。当前项目熵阈值是 0.3；PKC/mask 参数主要沿用论文设置。本项目 BN 更新模式为 `train_running`，由当前 SNN 上的消融实验选定，不是把论文原 backbone 和训练设置原样复现。

模型适应和 best iteration 选择不读取目标真值；但实验人员曾通过目标域评测结果选择超参数。因此应区分“适应算法无标签”与“超参数开发时看过目标域评测集”。

### 7.3 BN mode 及其选择

BatchNorm 有可学习的 `weight/bias` affine 参数和 `running_mean/var` 运行统计量。三个模式：

| `adakws_bn_mode` | 行为 | 已记录对照 |
|---|---|---:|
| `no_running` | 扔掉 running stats，每个 batch 现场算统计量，不积累 | DA-MT-09：71.78%（失效） |
| `checkpoint_running` | 固定使用 checkpoint 的 running stats，只适应 affine | DA-MT-10：83.47% |
| **`train_running`** | 用 batch stats 归一化，同时滑动更新 running stats | 当前推荐；当前 experiment4 EMA 起步 90.20% |

实现位置：`adakws_trainer.py:_set_adaptation_parameters()` 和 `_set_adaptation_mode()`。整体模型保持 train 以支持 SNN surrogate gradient；Dropout 单独设 eval；默认只有 4,608 个 BN affine 参数参与梯度更新。

### 7.4 启动和已验证结果

```bash
cd online_kws/AlphaSNN
conda activate snn
python alpha/asr/kws/online/train_adakws.py
```

`2026-10-19_15-24-45` 日志：

| 子集 | 起点 | 适应后 |
|---|---:|---:|
| 总体 1071 | 87.86% | **90.20%** |
| 真实车载 409 | 98.78% | **100.00%** |
| 加噪 simu 662 | 81.12% | **84.14%** |

提升 +2.33 percentage points；无监督评分选择 iteration 5；同 seed/配置复跑 90.20%。完整配置快照和结果在 `online/train.log/2026-10-19_15-24-45/log-train-2026-10-19-15-24-45`。

---

## 8. Checkpoint 格式、路径和加载检查

离线 AlphaSNN wrapper 的 state dict 前缀主要是 `subsample.*` 和 `model.*`；在线 wrapper 主干注册名为 `backbone.*`。`online/utils/train_utils.py:load_pretrained_model()` 将 `model.xxx → backbone.xxx`，`subsample.xxx` 同名加载。

当前 experiment4 raw/EMA checkpoint metadata：

| 文件 | `best_test_acc` | epoch | EMA |
|---|---:|---:|---|
| experiment4 `best-acc.pt` | 0.980392… | 224 | 否 |
| experiment4 `best-ema.pt` | 0.980392… | 253 | 是 |
| 在线 `best-adakws.pt` | 仅 model state dict | — | — |

**启动时检查**：日志必须显示 `文件是否存在: True` 和 `成功加载 99 个预训练参数（共 99 个）`。当前 loader 在加载失败时会 warning 并继续随机初始化；若见“将使用随机初始化的模型”，应停止运行，不要把输出当实验结果。

当前推荐重建版 checkpoint 以 `weights_only=True` 安全加载。KD-MT-08 原版 checkpoint 内含历史 `PosixPath` 对象，不是当前推荐的 online pretrained；若切回旧 checkpoint，需另行处理 `weights_only` allowlist。

---

## 9. 硬件部署接口与限制

模型前向接口（输入 FBank 已提取完成）：

```python
import numpy as np
if not hasattr(np, "int"):
    np.int = int

import torch
from spikingjelly.activation_based import functional
from model_kws import MultiTauLIFCommanderKWS
from utils.config import load_config
from utils.train_utils import load_pretrained_model

params = load_config("online/config/da_aol.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTauLIFCommanderKWS(dict(params.model)).to(device)
load_pretrained_model(params, model)
model.eval()

# features: [B, 80, T]；lengths: 每条 utterance 的有效 FBank 帧数 [B]
functional.reset_net(model)       # 每批/独立 utterance 前必须 reset 膜电位
with torch.no_grad():
    logits = model(features.to(device), lengths.to(device))
pred_ids = logits.argmax(dim=1)  # 0–28，中文标签映射见第 4.3 节
```

部署注意：

1. 当前仓库验证的是 Python/PyTorch/CUDA；**没有**已验证的 MCU/NPU C/C++ 移植、量化包、实时 VAD/端点检测或功耗/时延数据。
2. 训练和部署必须一致地实现 80-bin FBank、采样率/帧移、有效帧长度、subsampling 和 padding mask。
3. SNN 每个独立句子/批次前必须 `functional.reset_net(model)`，否则膜电位状态会跨句泄漏。
4. 普通固定权重推理与 AdaKWS 在线适应不同：AdaKWS 会更新 BN affine 和 `train_running` running stats；硬件若不允许模型状态更新，应部署离线 checkpoint 的固定模型，而不是声称部署了在线适应。

---

## 10. 当前限制与对外复现检查表

### 已知限制

- 原版 offline SNN 训练代码丢失；当前 `train_multitau.py` 是重建版。experiment4 的 98.04% 是当前可复现结果，但不能据此声称复现了原版全部隐藏实现细节。
- SNN KD 每个 epoch 都反复评估同一 357 条 clean test 并挑 best；该数值不是未触碰的独立 holdout。ANN 的 `valid_cuts()` 和 `test_cuts()` 也都指向同一 357 条 manifest。
- 所有大数据和权重均不在 Git 中；第三方必须通过项目提供方获得数据、teacher 和 SNN checkpoints。
- YAML/ANN datamodule 仍含开发环境绝对路径；接收方必须替换为本机数据/权重路径。
- `SNN/run_ann.sh` 固定 GPU 4；运行前应检查并修改设备号。
- 不能仅靠仓库代码重建原始语料：原始音频、数据许可、recipe 依赖和个别旧数据筛选过程未完整打包。

### 交付检查表

- [ ] 提供 `AlphaSNN/data/fbank/`、`AlphaSNN/data_train/fbank/` manifests、features 和索引。
- [ ] 提供 `AlphaSNN/kd_teacher/tdnn_ann_29class.pt`。
- [ ] 提供 experiment4 `best-acc.pt`、`best-ema.pt`。
- [ ] 把 ANN 数据路径、online/offline YAML 路径改成接收方路径。
- [ ] 核对 Python/PyTorch/torchaudio/Lhotse/SpikingJelly/CuPy/CUDA/icefall 环境。
- [ ] 离线固定评测 raw/EMA 均得到 98.04%。
- [ ] 在线日志确认所有 99 个预训练参数加载成功、目标真标签没有参与适应。
- [ ] 交付运行日志、完整 config dump、checkpoint SHA256、GPU 和驱动信息。
- [ ] 若声称硬件已部署，另外提供硬件平台、特征提取实现、量化方案、延迟/内存/能耗/准确率验证材料。

---

## 附录 A：命令速查

```bash
# 环境
conda activate snn

# ANN 教师训练（在 online_kws/SNN；先设置 manifest-dir/GPU）
python train_ann.py --world-size 1 --num-epochs 20 --num-class 29 \
  --max-duration 200 --manifest-dir /path/to/SNN/data/fbank

# ANN epoch 9–17 checkpoint 平均并导出教师
python decode_ann.py --epoch 17 --avg 9 --num-class 29 \
  --manifest-dir /path/to/SNN/data/fbank --export True

# 离线 SNN KD 重建训练（先换新 exp_dir）
python alpha/asr/kws/offline/train_multitau.py --conf alpha/asr/kws/conf/offline_kd.yaml

# 离线 experiment4 raw + EMA 固定口径评测
python alpha/asr/kws/offline/eval_multitau.py

# 当前推荐在线 AdaKWS（experiment4 EMA 起步，lr=2.2e-3）
python alpha/asr/kws/online/train_adakws.py
```
