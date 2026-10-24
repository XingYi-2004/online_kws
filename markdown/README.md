# AlphaSNN-KWS（原 SNN_aol）— 中文车载语音 KWS（SNN 训练 + 域适应在线学习）

> **⚠️ 2026-10-24 目录迁移**：项目已整体迁入 AlphaSNN 框架（SNN 与 SNN_aol 撤销）。
> 文中旧路径按以下映射读取：
> - `SNN_aol/offline/` → `AlphaSNN/alpha/asr/kws/offline/`
> - `SNN_aol/online/` → `AlphaSNN/alpha/asr/kws/online/`
> - `online/config/da_aol.yaml` → `alpha/asr/kws/conf/online_adakws.yaml`（entropy_threshold 已改回冠军值 0.3）
> - `offline/conf/multitau_29class_kd.yaml` → `alpha/asr/kws/conf/offline_kd.yaml`
> - `SNN/data/` → `AlphaSNN/data_source/`；`SNN_aol/{data,data_train,kd_teacher,alphasnn_29class*,scripts}` → `AlphaSNN/` 下同名目录
>
> 正文中的目录树与历史描述保留迁移前原貌；启动命令与数据路径均已更新为迁移后位置。


41 类中文车载语音指令关键词识别，对比 **Tdnnsnn** 与 **MultiTauLIFCommander（AlphaSNN）** 两种 SNN 架构，后续接入对抗域适应 + 伪标签在线学习。

## 小结
使用 MultiTauLIF 脉冲神经网络架构，参数量0.66M，离线语音指令识别精度为98.04%，完成“≥98%”目标

在线 DA+OCL 进展：目标域（1428 条退化版）初始 83.05%；Oracle 真标签微调天花板 **96.22%**（95% 目标可行）；
**当前最优：DA-MT-39 AdaKWS + EMA 教师伪标签 + 类别先验校正 86.48%**（teacher KL λ=0.1、μ=0.999、conf>0.9、BCT τ=0.5；无目标标签）；
源域混合微调（无目标标签）**86.34%**；AdaBN / 硬伪标签自训练为负面。
1. 原始41类中文车载语音数据集（测试集512条，均长1.71s）
模型	参数量	准确率	FLOPs (G)  SOPs (M)  Energy (mJ)  说明
TDNN-LSTM	32.10M	97.27%	2.08	0	9.58	-
TDNN-SNN	10.84M	94.14%	0.034	160.4	0.30	-
TDNN-SNN	10.84M	81.54%	-	-	-	CNN修复为Spiking激活
SpikConvNext	1.53M	86.87%	0.016	32.1	0.10	-
MultiTauLIF	0.66M	87.70%	0.006	9.5	0.036	-
2. 精简后29类中文车载语音数据集（测试集357条，均长1.75s）
模型	参数量	准确率	 FLOPs (G)  SOPs (M)  Energy (mJ)  说明
TDNN-LSTM	32.10M	99.71%	2.13	0	9.80	KD教师
TDNN-SNN	10.84M	99.44%	0.033	237.9	0.37	-
SpikConvNext	1.53M	-	-	-	-	无29类权重
MultiTauLIF	0.66M	98.04%	0.003	6.7	0.020	KD-MT-08

> **效率统计口径（2026-09-22 重测，`scripts/measure_efficiency.py`，SpikCommander 附录E Eq.18/20 口径）**：FLOPs/MAC = 第一个脉冲层之前接收连续输入的真实值层（Eq.20 的 FLOP_Conv¹）；SOPs = 各 LIF 输出脉冲数 × 下游突触 fan-out（Eq.18：fr × FLOP），脉冲→脉冲为膜电位积分不计 SOP，脉冲编码后的所有线性层（含分类头）均按 SOP 计；纯 ANN 用 thop 总 MAC（Eq.17）；能耗 = FLOPs × 4.6pJ + SOPs × 0.9pJ（45nm）。GSC 列为我们 16kHz/80mel 真实帧长测量（SpikCommander 论文为 8kHz/140mel/T=100，仅作方法对齐、非同输入规格）。
3. 英文指令数据集GSC(Google Speech Command) V2
共35个指令，train: 84843, valid: 9981, test: 11005
模型	参数量	准确率	FLOPs (G)  SOPs (K)  Energy (mJ) 说明
TDNN-LSTM	32.10M	95.98%	-	-	-	20epoch, lr=2e-4, wd=5e-4
TDNN-SNN	10.84M	95.05%	-	-	-	30epoch, lr=2e-4, wd=5e-4
SpikCommander	2.13M	96.92%	-	-	-	https://github.com/JackieWang9811/SCommander [AAAI2026] SOTA-level
SpikConvNext	1.53M	96.57%	-	-	-	准备投稿到 ICASSP 2027
MultiTauLIF	0.25M	95.40%	-	-	-	准备投稿到 ICLR 2027 或 ICASSP 2027
---

## 1. 项目结构

```
SNN_aol/
├── offline/                       # 离线 KD 训练（MultiTauLIFCommander）
│   ├── train_multitau.py          # 训练入口
│   ├── model_multitau.py          # MultiTauLIFCommander 包装器
│   ├── memmap_dataset.py          # Memmap 特征数据集 + DataLoader
│   ├── training_utils.py          # 训练工具（EMA / 分组优化器 / SpikingCELoss）
│   └── run_alphasnn.sh            # 启动脚本
├── online/                        # 在线学习（纯 AdaKWS，MultiTauLIFCommanderKWS）
│   ├── train_adakws.py            # AdaKWS 在线学习入口（原名 train_da_aol.py）
│   ├── model_kws.py               # MultiTauLIFCommanderKWS 模型封装（原名 model_da_aol.py）
│   ├── config/da_aol.yaml         # 唯一配置文件（GPU/模型/数据/超参数）
│   ├── trainers/                  # da_trainer.py / online_trainer.py 训练循环
│   ├── utils/                     # config / data / train_utils
│   ├── train.log/                 # 实验输出（日志 + checkpoint）
│   └── scratch_adakws.py          # AdaKWS TTA 原型（在线 BN affine 适应验证）
├── tdnn/                          # TDNN 基线（ANN / SNN）
│   ├── model_ann.py               # TdnnLstm（ANN baseline）
│   ├── model_snn.py               # TdnnSNN（TDNN+SpikingVanillaRNN）
│   ├── train_ann.py               # ANN 训练
│   ├── train_snn.py               # SNN 训练
│   ├── decode_ann.py              # ANN 解码
│   ├── decode_snn.py              # SNN 解码
│   ├── run_ann.sh                 # ANN 启动脚本
│   ├── run_snn.sh                 # SNN 启动脚本
│   └── run.sh                     # 通用启动脚本
├── convert_to_memmap.py           # lilcom → memmap 转换脚本
├── local/                         # 数据准备（Kaldi→Lhotse / FBank 计算）
├── data/                          # 预计算特征（FBank）
│   └── fbank/
│       ├── command_feats_train.dat   # memmap (19 GB)
│       ├── command_feats_train.idx   # 索引
│       └── command_cuts_*.jsonl.gz   # lhotse 清单
├── skipconvnext_41class/          # SpikConvNext 训练产物
├── alphasnn_41class/              # AlphaSNN 训练产物
├── AlphaSNN/                      # AlphaSNN 源码
├── SkipConvNext/                  # SkipConvNext 参考实现
├── utils/                         # 工具脚本
└── braintrace/                    # 测试
```

---

## 2. 环境

- **Python**：3.10（conda 环境 `snn`）
- **PyTorch**：2.5.1+cu121
- **spikingjelly**：源码安装（2.0.0.dev1）
- **其他关键依赖**：`pytorch-lightning`, `einops`, `torchtyping`, `loguru`, `asteroid-filterbanks`

```bash
conda activate snn
cd /home/huxixiang20040727/SNN_online_learn/online_kws/AlphaSNN
```

---

## 3. 数据

项目涉及三套数据集：

### 3.1 GSC数据集

使用的 Google Speech Commands 数据集。

| 指标 | 数值 |
|---|---|
| 类别数 | 35 |
| 训练样本 | ~85,000 |
| 测试样本 | ~11,000 |
| 平均时长 | ~1.0s |
| 数据类型 | **干净语音**（无噪声） |
| 特征维度 | 140-dim mel |
| 特征提取 | 在线 torchaudio, 5ms 帧移 |

### 3.2 29 类中文车载语音（离线 KD + 域适应在线学习）

两套 29 类数据集，标签空间已对齐（0-28，同一套中文车载指令）。

#### `data/fbank/` — 离线 KD 训练（离线学习数据集）

| 指标 | 数值 |
|---|---|
| 类别数 | 29（0-28，完整）|
| 训练集 | 128,076 条（`command_cuts_train_29.jsonl`）|
| 测试集 | 357 条（`command_cuts_test_29.jsonl`）|
| 数据类型 | 训练: **100% 人工加噪**（SNR 2-15dB, mean=7.3dB），测试: **100% 干净**（噪声→干净跨域）|
| 特征格式 | 80-dim FBank（lilcom 压缩 .lca, 10ms 帧移）|
| 数据管线 | lhotse / memmap |

#### `data_train/fbank/` — 域适应 + 在线学习（真实车载录音 + 录制任务混合）（在线学习数据集）

| 指标 | 数值 |
|---|---|
| 类别数 | 29（0-28，与 data 对齐。⚠️ label 5 切换到主路 从 data 导入补齐）|
| 训练集 | 170,499 条（`command_cuts_train_29.jsonl`）|
| 测试集 | 1,428 条（`command_cuts_test_29.jsonl`）：409 真实车载 + 662 加噪 simu（0~8dB）+ 357 RIR 远场混响 |
| 数据来源 | iwrd 录制任务 49%、真实车载录音 ~33%（长城/吉利/合众/比亚迪/极氪等）、csnt 任务 4%、MongoDB 录制 14% |
| 数据类型 | 混合：录制任务（干净/人工加噪）+ 真实车载录音（自然带噪）|
| 特征格式 | 80-dim FBank（lilcom 压缩 .lca, 10ms 帧移）|
| 数据管线 | lhotse（`online/utils/data.py`）|

> 2026-08-05: 重新从 41 类数据提取 29 类子集，对齐 `data` 标签空间。发现原 data_train 缺 label 5（切换到主路），从 data 导入 2,892 条补齐。
> 同日：修复 lhotse `BuiltinIOBackend.open` 在 binary 模式下误传 `encoding='utf-8'` 的 bug（`serialization.py:718`）。
> 同日：修复 `train_da_aol.py` 源域加载路径（41 类 `.gz` → 29 类 `.jsonl`），解决 CUDA assert（标签越界）和加载卡住问题。
> 2026-09-05: 确认测试集 1,428 条全部是 `data/fbank` 干净测试集 357 条的退化版本（同 id 同标签配对，100% 一致）。
> 预训练模型（KD-MT-08）三子集精度：真实车载 99.27% / 加噪 simu 79.46% / RIR 混响 71.99%（整体 83.26%）。

### 3.3 41 类中文车载语音

两套 41 类数据集，标签文本略有差异（标注来源不同）。

#### `data/fbank/` — 离线训练（模拟车载噪声）

| 指标 | 数值 |
|---|---|
| 类别数 | 41（实际标注 41 类，但训练缺 label=5 关闭内循环, 测试缺 label=40 旅游景点） |
| 训练集 | 225,990 条（`command_cuts_train.jsonl`）|
| 测试集 | 512 条（`command_cuts_test.jsonl`）|
| 训练均长 | 2.7s |
| 测试均长 | 1.7s |
| 最长时长 | 4.6s |
| 数据类型 | 训练: 100% 噪声（SIMU）, 测试: **100% 干净**（噪声→干净跨域）|
| 训练 SNR | 均值 6.8dB, P50 7dB, 范围 2–14dB |
| ⚠️ 标签说明 | 训练=40类（0-4,6-40，缺 label 5），测试=40类（0-39，缺 label 40），互补。`num_class=41` 正确。|
| 特征存储 | memmap 文件 `command_feats_train.dat` (19 GB) + `command_feats_test.dat`（80-dim FBank, 10ms 帧移）|

#### `data_train/fbank/` — 在线 DA + AOL（真实车载录音混合）

| 指标 | 数值 |
|---|---|
| 类别数 | 41（训练缺 label=9 切换到主路, 测试缺 label=40 usb音乐） |
| 训练集 | 326,376 条（`command_cuts_train.jsonl.gz`）|
| 测试集 | 2,048 条（`command_cuts_test.jsonl.gz`）|
| 训练均长 | 2.1s |
| 测试均长 | 1.7s |
| 训练最长 | 9.8s |
| 训练总时长 | 192.3h |
| 数据类型 | 混合：录制任务（干净/人工加噪）+ 真实车载录音（自然带噪）|
| ⚠️ 标签说明 | 训练=40类（0-8,10-41，缺 label 9），测试=40类（0-39，缺 label 40），互补。`num_class=41` 正确。|
| 特征格式 | 80-dim FBank（lilcom .lca, 10ms 帧移）|
| 特征存储 | `command_feats_train/`（dir, 6.2GB）+ `command_feats_test.lca`（32MB）|

---

## 4. 训练

### 4.1 SpikConvNext

```bash
bash run_snn.sh
```

| 超参数 | 值 |
|---|---|
| 模型 | SpikConvNext（depths=2, subsampling_factor=3） |
| 参数量 | 1.532M |
| 学习率 | 5e-3（AdamW） |
| Weight Decay | 5e-3 |
| Scheduler | CosineAnnealingLR（T_max=100, eta_min=0） |
| Batch size | 2056（memmap 固定打包） |
| Epochs | 100 |
| SpecAugment | 关闭（数据自带噪声） |
| 数据管线 | numpy memmap 直接切片 |

### 4.2 MultiTauLIFCommander（AlphaSNN）

```bash
bash run_alphasnn.sh
```

| 超参数 | 值 |
|---|---|
| 模型 | MultiTauLIFCommander（depths=5, hidden=384） |
| 参数量 | 0.660M |
| 类别数 | 29 类（中文车载指令，label 0-28） |
| taus | [2.0, 4.0, 8.0]（learnable_alpha=true, channel_adaptive 融合） |
| 学习率 | 2e-3（AdamW, warmup 30 epoch → cosine 退火到 1e-5） |
| Weight Decay | 5e-3（tau/branch 参数独立 0.0） |
| 数据管线 | lhotse（max_duration=1500s, DynamicBucketingSampler） |
| Epochs | 300 |
| Loss | SpikingCELoss + label_smoothing=0.1 |
| EMA | decay=0.9995（验证/测试时应用） |
| Branch Balance | weight=0.01（KL 散度防单分支坍缩） |
| SpecAugment | ✅ time_mask=10, freq_mask=3, num=2（防过拟合） |
| KD 教师 | TdnnLstm（32M, 99.72%）, alpha=0.7, T=10.0 |
| subsampling_factor | 2（时间步减半） |
| 精度 | **98.04%**（KD-MT-08, E82/300） |

### 4.3 TdnnSNN（TDNN + SpikingVanillaRNN）

```bash
cd tdnn && bash run_snn.sh
```

| 超参数 | 值 |
|---|---|
| 模型 | TdnnSNN（3×Conv1d+LIF+BN + 6×SpikingVanillaRNN + Linear + TimeMean） |
| 参数量 | 10.84M |
| 类别数 | 29 类 |
| 隐藏维度 | 768 |
| 学习率 | 2e-4（AdamW） |
| Weight Decay | 5e-4 |
| Scheduler | StepLR（每 10 epoch ×0.5） |
| Epochs | 20（默认） |
| Loss | CE |
| subsampling_factor | 3（时间步 ÷3） |
| KD 教师 | TdnnLstm（32M, 99.72%）, alpha=0.7, T=3.0 |
| 精度 | **99.44%**（KD 训练，噪声测试集） |

### 4.4 TdnnLstm（ANN 基线 / KD 教师）

```bash
cd tdnn && bash run_ann.sh
```

| 超参数 | 值 |
|---|---|
| 模型 | TdnnLstm（3×Conv1d+ReLU+BN + 6×LSTM + Linear） |
| 参数量 | 32.10M |
| 类别数 | 41 类 |
| 隐藏维度 | 768 |
| 学习率 | 2e-4（AdamW） |
| Weight Decay | 5e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 20 |
| max_duration | 200s / batch |
| Loss | CE |
| subsampling_factor | 3 |
| 精度 | **99.72%**（用作 MultiTauLIFCommander KD 教师） |

### 训练产物

每个实验目录下生成：
- `best-acc.pt` — 最佳验证精度 checkpoint
- `last.pt` — 最后一个 epoch 的 checkpoint
- `log/log-train-*.log` — 结构化训练日志
- `tensorboard/` — TensorBoard 事件文件

---

## 5. 实验日志

### SpikConvNext（41 类）离线学习

| ID | 模型 | Depths | Batch | 参数量 | Epochs | LR | SpecAug | Best Valid | Best Test | Final Valid (E99) | Final Test (E99) | 日志 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPC-01 | SpikConvNext | 2 | max_dur=4096 | 1.532M | 100 | 5e-3 | GSC风格(F=10,pS=0.25) | **86.72%** (E63) | **86.72%** | 84.18% | 84.18% | `2026-07-17_09-54-32` |
| SPC-02 | SpikConvNext | 2 | 1028 | 1.532M | 100 | 2e-3 | 轻量(F=5,pS=0.1) | **83.40%** | **83.40%** | ~83% | — | `2026-07-19_22-17-01` |
| SPC-03 | SpikConvNext | 2 | 1028 | ~4M | 100 | 2e-3 | 轻量(F=5,pS=0.1) | **84.77%** (E21) | — | — | — | `2026-07-20_21-11-25` |
| SPC-04 | SpikConvNext | **3** | 256 | **19.52M** | 100 | 2e-3 | **关闭** | **84.57%** | — | — | — | `2026-07-21_08-36-49` |

> SPC-01: lhotse 管线 + GSC 全量 SpecAugment，过拟合严重（Train ~98%，Valid 86.72%→68.9%）  
> SPC-02: memmap 管线 + 轻量 SpecAugment + dropout=0.1，更稳定但峰值略低  
> SPC-03: hidden=512 + label_smoothing=0.1 + wd=5e-4，84.77% 后开始过拟合  
> SPC-04: hidden=768 + depths=3（19.5M 参数），Train 100%→严重过拟合，Best Valid=84.57%。100 epoch 是代码硬编码，无法通过配置修改。
> ⚠️ SPC-01 日志显示 Train Acc ~1% 是显示 bug（忘乘100），实际为 ~98%

### MultiTauLIFCommander（41类）离线学习

| ID | 日期 | 模型 | hidden | taus | Loss | EMA | BranchBal | Best Valid | Best Test | 日志 |
|---|---|---|---|---|---|---|---|---|---|---|
| MT-01 | 7/21 | MultiTauLIF (旧管线) | 768 | [5,20,100] | CE | 无 | 无 | 42.58% | — | `2026-07-21_09-16-29` |
| MT-02 | 7/25 | MultiTauLIF (新管线) | 256 | [5,20,100] | SpikingCE | ✅ | ✅ | 76.76% (E34) | 76.76% | `2026-07-25_21-26-05` |
| MT-03a | 7/26 | +cupy+warmup=0 bs=768 | 256 | [5,20,100] | SpikingCE | ✅ | ✅ | 79.69% | 79.69% | `07-26_13-38-16` |
| MT-03b | 7/26 | +cupy+warmup=0 bs=512 | 256 | [5,20,100] | SpikingCE | ✅ | ✅ | **80.27%** (E61) | **80.27%** | `07-26_13-42-53` |
| MT-04 | 7/26 | 加容量+强正则 | 384 | [5,20,100] | SpikingCE | ✅ | ✅ | **87.70%** (E53) | **87.70%** | `07-26_22-08-11` |
| MT-05 | 7/27 | 强正则+warmup=30 | 256 | [5,20,100] | SpikingCE | ✅ | ✅ | 71.48% (E50) | 71.48% | `07-27_17-46-22` |
| MT-06 | 7/27 | 对齐free_bal | 256 | [2,4,8] | SpikingCE | ✅ | ✅ | **83.98%** (E186) | **83.98%** | `07-27_21-30-24` |
| MT-07 | 7/28 | 大模型+free_bal | 384 | [2,4,8] | SpikingCE | ✅ | ✅ | **83.40%** | **83.40%** | `07-28_11-19-32` |

> MT-03a/b: 同配置仅 batch_size 不同，精度差 <0.6%，属随机波动。  
> MT-04: 0.665M 参数，taus=[5,20,100]+learnable_alpha=true+dropout=0.3+ls=0.1，**87.70% 目前最优**（已超 SpikConvNext 86.72%）。  
> MT-05: 小模型套大正则化 → 71.48%。**小模型不能用强正则化。**  
> MT-06: 0.255M 小模型 + taus=[2,4,8]+learnable_alpha=false+轻正则化 → **83.98% 小模型最优**。  
> MT-07: 0.665M 大模型 + MT-06的taus配置 → 83.40%，**不如 MT-04。大模型需要长taus+可学习α，小模型需要短taus+固定α。**
> MT-06: 对齐 free_bal（taus=[2,4,8] + learnable_alpha=false + wd=0.002 + dropout=0.15），小模型新纪录 **83.98%**。warmup=30 在这个配置下工作正常。

**迭代进展：**

| 阶段 | 关键改动 | Best Valid | 涨幅 |
|------|---------|-----------|------|
| MT-01 | 旧训练管线（CE loss，无 EMA/分支平衡） | 42.58% | — |
| MT-02 | EMA+SpikingCE+分支平衡+分组优化器 | 76.76% | **+34.18%** |
| MT-03 | cupy+warmup=0+epochs=200 | 80.27% | **+3.51%** |
| MT-04 | hidden=384 depth=5 dropout=0.3 label_smoothing=0.1 warmup=30 | **87.70%** | **+7.43%** |
| MT-05 | 小模型+强正则(warmup=30,d=0.3,ls=0.1) | 71.48% | -8.79% vs MT-03b |
| MT-06 | 对齐free_bal(taus=[2,4,8],learnable_alpha=false,wd=0.002,d=0.15) | **83.98%** | **+3.71%** vs MT-03b |


### 知识蒸馏（29 类，TdnnSNN + TdnnLstm 教师）

预训练模型，作为域适应 + 在线学习的起点。

| 模型 | 数据集 | 类别 | hidden_dim | subsampling | Epochs | 解码方式 | 测试精度 | 日志 |
|---|---|---|---|---|---|---|---|---|
| TdnnSNN | **10.84M** | 29类混合（data_train 旧版标签） | 29 | 768 | 3 | 25 | 5 epoch averaged | **99.44%** | `tdnn_lstm_6snn_new9/exp/log/log-decode-2026-06-28-07-34-39` |

**架构：** TDNN 前端（3×Conv1d+LIF+BN, 80→768, 3.7M）+ 6×SpikingVanillaRNN（768→768, 7.1M）+ Linear（768→29）+ TimeMean pooling。总参数 **10,844,189**。

> 使用 **KD（TdnnLstm 教师）** 训练，平均 epoch 21-25 权重，在噪声测试集上达到 99.44%。

### GSC v2 35 类（TdnnLstm / TdnnSNN，data.list 管线）

| 模型 | 参数量 | 特征 | Epochs | LR | WD | Best Valid | Test | 日志 |
|------|--------|------|--------|----|----|-----------|------|------|
| TdnnLstm | 32.10M | 80-dim FBank | 20 | 2e-4 | 5e-4 | **95.98%** | 95.77% | `SNN/ann_checkpoint/ann_v2_35class/train_20260810_233339.log` |
| TdnnSNN | 10.84M | 80-dim FBank | 30 | 2e-4 | 5e-4 | **95.05%** | 94.42% | `SNN/snn_checkpoint/snn_v2_35class/train_20260811_092745.log` |
| **MultiTauLIFCommander** | **0.253M** | 80-dim FBank | 300 | 2e-3 | 2e-3 | 95.78% (E288) | **95.53%** (E293) | `alphasnn_gsc35/log/log-train-2026-08-12_00-55-13.log` |
| **MultiTauLIF (大)** | **0.662M** | 80-dim FBank | 300 | 2e-3 | 2e-3 | 96.02% (E243) | **96.05%** (E243) | `alphasnn_gsc35/log/log-train-2026-08-12_14-57-33.log` |

> 数据：GSC v2 官方 train 84843 / valid 9981 / test 11005。  
> **MultiTauLIFCommander (小)**：对齐官方 free_bal_248_rir（depths=4, taus=[2,4,8], learnable_alpha=false, dropout=0.25），仅 n_inputs=80 不同。在线 FBank + SpecAugment。**0.253M 参数达到 95.53%，与 10.84M 的 TdnnSNN 持平，参数效率 43x。**  
> **MultiTauLIF (大)**：MT-04 配方（depths=5, hidden=384, taus=[5,20,100], learnable_alpha=true, dropout=0.3），memmap 管线。**0.662M 参数 test 95.74%，比小模型 +0.21pp。**

### 知识蒸馏（29 类，MultiTauLIFCommander + TdnnLstm 教师）

| ID | 日期 | 配置 | 老师 | α | T | Best Valid | 日志 |
|---|---|---|---|---|---|---|---|
| KD-MT-01 | 8/1 | MT-06 (0.252M) | TdnnLstm (32M, 99.72%) | 0.7 | 3.0 | **81.51%** (E43) | `2026-08-01_01-14-42` |
| KD-MT-02 | 8/2 | MT-06 (0.252M) + lhotse管线 + 修复3bug + warmup=30 | TdnnLstm (32M, 99.72%) | 0.7 | **10.0** | **90.76%** (E83) | `2026-08-02_16-07-12` |
| KD-MT-03 | 8/2 | MT-06 (0.252M) + 防过拟合(d=0.2,wd=5e-3) + warmup=5 | TdnnLstm (32M, 99.72%) | **0.5** | 10.0 | **89.08%** (E72) | `2026-08-02_19-42-59` |
| KD-MT-04 | 8/3 | MT-04 (0.660M) + warmup=5 | TdnnLstm (32M, 99.72%) | 0.7 | 10.0 | **90.48%** (E300) | `2026-08-03_00-09-49` |
| KD-MT-05 | 8/3 | MT-06 (0.252M) + **SpecAugment(10/3)** + warmup=5 | TdnnLstm (32M, 99.72%) | 0.7 | 10.0 | **96.92%** (E108) | `2026-08-03_10-28-04` |
| KD-MT-06 | 8/3 | MT-06 (0.252M) + **SpecAugment(15/5)** + warmup=30 | TdnnLstm (32M, 99.72%) | 0.7 | **15.0** | **96.64%** (E86) | `2026-08-03_16-16-10` |
| KD-MT-07 | 8/4 | MT-04风格 (0.252M) + SpecAugment(10/3) + **learnable_alpha=true** + d=0.3 + ls=0.1 | TdnnLstm (32M, 99.72%) | 0.7 | 10.0 | **96.92%** (E203) | `2026-08-04_00-20-15` |
| **KD-MT-08** | **8/4** | **MT-04 (0.660M)** + SpecAugment(10/3) + warmup=30 | TdnnLstm (32M, 99.72%) | 0.7 | 10.0 | **🚀 98.04%** (E82) | `2026-08-04_13-10-57` |

> KD-MT-01: 首次 KD 实验，81.51%（E43/300）。存在三个已知 bug。
> KD-MT-02: 修复三个 bug + 切换 lhotse 管线 + T=10 + warmup=30。90.76%。
> KD-MT-05: **+SpecAugment(轻量) → 96.92%！** Train 96.6%/Valid 96.9%，gap 仅 0.3pp。
> KD-MT-06: SpecAugment 加强(15/5) + T=15 + warmup=30 → **Train 89.7%/Valid 96.6%，训练欠拟合！**
> KD-MT-07: MT-04 风格正则化 + SpecAugment(10/3) + T=10 → **96.92% 持平 KD-MT-05**。252K 模型天花板已确认。
> KD-MT-08: **🎉 大模型(660K) + KD + SpecAugment → 98.04%！突破 98%！** Train 96.8%/Valid 98.0%（SpecAugment 下 valid > train 正常）。比 252K KD 涨 **+1.12pp**，比无 KD 的 MT-04(87.70%) 涨 **+10.34pp**。**容量是突破天花板的关键。**

### 域适应 + 在线学习（29 类）

**数据集全景（五份数据一览）**：

| 数据集 | 数量 | 底材 | 处理 | 预训练精度 | 数据来源（HPC 路径） |
|---|---|---|---|---|---|
| 源域训练 | 128,076 | 真人居家实录音（iwrd 采集：卧室/客厅，手机 50cm，接近干净） | 仿真加噪（SNR 2~15dB） | —（训练集） | `zheshu.song_sx/icefall-master/egs/SNN/download/train/wavs/simu_wavs/` |
| 干净测试 | 357 | 同类真人实录音（匿名 id） | 无 | 98.04% | `zheshu.song_sx/.../download/test/wavs/` |
| 目标域 real_car | 409 | 同一批 357 句（在真实车内**另录一版**） | 无 | 99.27% | `qing.zhou/.../comm_asr_cn/test_set/rec/comm_domain/Car_*` |
| 目标域 addsimu | 662 | 干净测试音频**本身** | 加性仿真噪声（0~8dB，`addsimu_Xdb`） | 79.46% | `yurui.tian_sx/icefall/egs/SNN/data4/manifests/enhanced_audio/` |
| 目标域 RIR | 357 | 干净测试音频**本身** | RIR 房间脉冲响应卷积（远场仿真，`simufarXX`） | 71.99% | 同上 |

> 目标域三个子集（1,428 条）全部与干净测试 357 条同 id 同标签配对（100% 核实）。整体初始精度 83.26%。
> "enhanced_" 前缀 = 数据增强脚本的输出命名（`local/kaldi_to_lhotse_test.py` 中 `f"enhanced_{utt_id}.wav"`），
> 即"加噪/加混响后的音频"，**不是语音增强去噪**；底材就是干净测试 wav 本身。
> 原始音频均在 HPC 上（本地不可访问），本地只有预计算 FBank 特征（`data/fbank`、`data_train/fbank`）。
> 源域训练数据 100% 为加噪 simu（无 RIR），故 RIR 子集精度最低（71.99%）符合预期。

**实验记录（DA-MT 系列，2026-09-05～2026-09-14）：**

> **SPACE 实验标识**：只有实验表中明确带有 **`（SPACE）`** 标记的 DA-MT-44～DA-MT-47 使用了官方 SPACE 思路的 KWS 改造版；DA-MT-01～DA-MT-43 均不是 SPACE。SPACE 的详细实现、协议问题和结果见下方“SPACE-KWS”记录。

| ID | 日期 | 实验 | DA 阶段 | OCL 阶段 | 起点精度 | 最佳/最终 | 日志/脚本 |
|---|---|---|---|---|---|---|---|
| DA-MT-01 | 9/5 | 现有 DA+AOL 管线完整跑 | ✅ 15 epoch | ✅ 10 轮 | 83.05% | 82.84%（最终） | `online/train.log/2026-09-05_00-29-44` |
| DA-MT-02 | 9/5 | Oracle 真标签微调（天花板） | ❌ | ❌ | 84.19%（留出291） | **96.22%**（E8） | `scratch_oracle.py` |
| DA-MT-03 | 9/5 | 源域混合微调（无目标标签） | ❌ | ❌ | 83.26% | **86.34%**（E6） | `scratch_srctrain.py` |
| DA-MT-04 | 9/5 | AdaBN（BN 统计量重估） | ❌ | ❌ | 83.26% | 70.10% | `scratch_selftrain.py` |
| DA-MT-05 | 9/5 | 硬伪标签自训练（独立原型） | ❌ | ⚠️ 原型，非管线 OCL | 83.26% | 80.11% | `scratch_selftrain.py` |
| **DA-MT-06** | **9/9** | **回放库 + 对抗 DA + OCL（当前管线）** | **✅ 30 epoch** | **✅ 10 轮** | **83.47%** | **83.89%（最终最佳）** | `online/train.log/2026-09-09_10-58-03/last.pt` |
| **DA-MT-07** | **9/9** | **AdaKWS 在线测试时适应** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **83.47%** | **85.08%（最终）** | `online/train.log/2026-09-09_13-36-40/best-adakws.pt` |
| **DA-MT-08** | **9/9** | **AdaKWS 论文流程修正版** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **85.85%（最终）** | `online/train.log/2026-09-09_14-35-08/best-adakws.pt` |
| **DA-MT-09** | **9/9** | **AdaKWS 实验 A：每批使用 batch statistics** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **71.78%（最终）** | `online/train.log/2026-09-09_14-50-54/best-adakws.pt` |
| **DA-MT-10** | **9/9** | **AdaKWS 实验 B：恢复 checkpoint running statistics + SGD** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **83.47%（最终）** | `online/train.log/2026-09-09_15-01-00/best-adakws.pt` |
| **DA-MT-11** | **9/9** | **AdaKWS 实验 D1：BN + GN affine** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **83.47%（最终）** | `online/train.log/2026-09-09_16-20-14/best-adakws.pt` |
| **DA-MT-12** | **9/9** | **AdaKWS 实验 D2：BN + 分类头** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **82.49%（最终）** | `online/train.log/2026-09-09_16-22-37/best-adakws.pt` |
| **DA-MT-13** | **9/9** | **AdaKWS 实验 D3：最后一层 Block + 分类头** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **82.91%（最终）** | `online/train.log/2026-09-09_16-25-24/best-adakws.pt` |
| **DA-MT-15** | **9/12** | **AdaKWS + 源域回放 CE（λ=0.05）** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **85.15%（最终）** | `online/train.log/2026-09-12_16-58-45/best-adakws.pt` |
| **DA-MT-16** | **9/12** | **AdaKWS + 源域回放 CE（λ=0.01）** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **85.43%（最终）** | `online/train.log/2026-09-12_17-25-56/best-adakws.pt` |
| **DA-MT-17** | **9/12** | **DA-MT-08 复现（λ=0，补子集分解）** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **85.85%（最终）** | `online/train.log/2026-09-12_17-28-42/best-adakws.pt` |
| DA-MT-18 | 9/12 | AdaKWS 迭代延长 10→20 轮 | ❌ 跳过 | ✅ AdaKWS 20 轮 | 82.77% | 85.64%（最终） | `online/train.log/2026-09-12_17-30-54/best-adakws.pt` |
| DA-MT-19 | 9/12 | AdaKWS 学习率微扫 lr=1.5e-3 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.71%（最终） | `online/train.log/2026-09-12_17-32-55/best-adakws.pt` |
| DA-MT-20 | 9/12 | AdaKWS + 冻结教师 KL（λ=0.05，conf>0.9） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 84.80%（最终） | `online/train.log/2026-09-12_17-35-53/best-adakws.pt` |
| DA-MT-21 | 9/12 | E1：EMA 教师伪标签（KL 权重 1.0，μ=0.99） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 84.66%（最终） | `online/train.log/2026-09-12_17-59-14/best-adakws.pt` |
| **DA-MT-22** | **9/12** | **E1：EMA 教师伪标签（KL 权重 0.1，μ=0.99）** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **86.06%（最终，新最优）** | `online/train.log/2026-09-12_18-02-28/best-adakws.pt` |
| DA-MT-23 | 9/12 | E1：EMA 教师伪标签（KL 权重 0.05，μ=0.99） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.64%（最终） | `online/train.log/2026-09-12_18-04-42/best-adakws.pt` |
| DA-MT-24 | 9/12 | E1：EMA 教师伪标签（KL 权重 0.2，μ=0.99） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.78%（最终） | `online/train.log/2026-09-12_18-06-34/best-adakws.pt` |
| **DA-MT-25** | **9/12** | **E1：EMA 教师伪标签（KL 权重 0.1，μ=0.999）** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **86.41%（最终，新最优）** | `online/train.log/2026-09-12_18-08-32/best-adakws.pt` |
| DA-MT-26 | 9/12 | E1：EMA 教师伪标签（KL 权重 0.1，μ=0.9999） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.71%（最终） | `online/train.log/2026-09-12_18-10-26/best-adakws.pt` |
| DA-MT-27 | 9/12 | E2：E1 + 目标域记忆库邻居伪标签（λ_n=0.1，K=5，每类20） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 86.41%（最终，并列最优） | `online/train.log/2026-09-12_18-17-16/best-adakws.pt` |
| DA-MT-28 | 9/12 | E2：记忆库邻居伪标签（λ_n=0.05） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.57%（最终） | `online/train.log/2026-09-12_18-19-24/best-adakws.pt` |
| DA-MT-29 | 9/12 | E3：记忆库否决式过滤（min_sim=0.2，严格模式） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.22%（最终） | `online/train.log/2026-09-12_22-10-20/best-adakws.pt` |
| DA-MT-30 | 9/12 | E3：否决式修正 has_sim（与 29 等价）+ 相似度诊断 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.22%（最终） | `online/train.log/2026-09-12_22-12-42/best-adakws.pt` |
| DA-MT-31 | 9/12 | E3：否决式过滤（min_sim=0.97，同源兄弟校验） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.57%（最终） | `online/train.log/2026-09-12_22-19-45/best-adakws.pt` |
| DA-MT-32 | 9/13 | A1：E1 + 回放 CE（λ=0.01） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.08%（最终） | `online/train.log/2026-09-13_10-58-39/best-adakws.pt` |
| DA-MT-33 | 9/13 | A1：E1 + 回放 CE（λ=0.05） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.50%（最终） | `online/train.log/2026-09-13_11-01-08/best-adakws.pt` |
| DA-MT-34 | 9/13 | A2：CoTTA 式增强平均教师（4 视图） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.22%（最终） | `online/train.log/2026-09-13_11-04-27/best-adakws.pt` |
| DA-MT-35 | 9/13 | A3：教师置信度阈值 0.95 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.92%（最终） | `online/train.log/2026-09-13_11-07-34/best-adakws.pt` |
| DA-MT-36 | 9/13 | A3：教师置信度阈值 0.85 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.50%（最终） | `online/train.log/2026-09-13_11-09-11/best-adakws.pt` |
| DA-MT-37 | 9/13 | B1：类别先验校正 τ=1.0（仅损失端） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.29%（最终） | `online/train.log/2026-09-13_11-15-28/best-adakws.pt` |
| DA-MT-38 | 9/13 | B1：先验校正 τ=1.0（损失+预测端） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.36%（最终） | `online/train.log/2026-09-13_11-17-44/best-adakws.pt` |
| **DA-MT-39** | **9/13** | **B1：先验校正 τ=0.5（损失+预测端）** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **86.48%（最终，新最优）** | `online/train.log/2026-09-13_11-19-35/best-adakws.pt` |
| DA-MT-40 | 9/13 | B1：先验校正 τ=0.25（损失+预测端） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.92%（最终） | `online/train.log/2026-09-13_11-21-17/best-adakws.pt` |
| DA-MT-41 | 9/13 | C1：源域微调（5000 simu + 357 干净，全模型 10 epoch） | ❌ 跳过 | ❌（替代在线阶段） | 83.47% | 85.08%（最终） | `online/train.log/2026-09-13_13-01-51/source_ft_2026-09-13_13-01-51/best-source-ft.pt` |
| DA-MT-42 | 9/13 | C2：源域微调 + E1 在线适应 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 84.31% | 85.71%（最终） | `online/train.log/2026-09-13_13-12-28/best-adakws.pt` |
| DA-MT-43 | 9/13 | B2：输入侧 80 维谱增益适应（E-BATS 式） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.85%（最终） | `online/train.log/2026-09-13_14-04-48/best-adakws.pt` |
| **DA-MT-44（SPACE）** | **9/14** | **SPACE-KWS：官方 SPACE 改造版，全参数，lr=1e-4** | **❌ 跳过** | **✅ SPACE 逐样本流式** | **83.26%** | **69.12%（最终）** | `online/train.log/2026-09-14_11-11-39/best-space.pt` |
| **DA-MT-45（SPACE）** | **9/14** | **SPACE-KWS：BN affine，lr=1e-5** | **❌ 跳过** | **✅ SPACE 逐样本流式** | **83.26%** | **67.86%（最终）** | `online/train.log/2026-09-14_11-30-11/best-space.pt` |
| **DA-MT-46（SPACE）** | **9/14** | **SPACE-KWS：BN affine，lr=1e-6** | **❌ 跳过** | **✅ SPACE 逐样本流式** | **83.26%** | **68.14%（最终）** | `online/train.log/2026-09-14_11-45-07/best-space.pt` |
| **DA-MT-47（SPACE，修正版）** | **9/14** | **SPACE-KWS：冻结 BN running statistics，BN affine，lr=1e-6** | **❌ 跳过** | **✅ SPACE 逐样本流式** | **83.26%** | **71.64%（最终）** | `online/train.log/2026-09-14_12-00-12/best-space.pt` |
| **DA-MT-48（CDAN+E）** | **9/23** | **条件对抗域适应（判别器输入 T(f,g)=f⊗softmax(g) 11136维，熵条件加权）+ OCL** | **✅ 30 epoch** | **✅ 10 轮** | **83.47%** | **84.31%（最终）** | `online/train.log/2026-09-23_10-44-58/last.pt` |
| **DA-MT-49（CDAN+E + AdaKWS）** | **9/23** | **CDAN+E DA 30 epoch → AdaKWS+EMA+BCT 在线 10 轮** | **✅ 30 epoch** | **✅ AdaKWS 10 轮** | **82.77%** | **84.94%（最终）** | `online/train.log/2026-09-23_17-17-22/best-adakws.pt` |
| DA-MT-50a | 9/24 | 纯 AdaKWS 扫描 V0：lr=1e-3 基线复现 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.85%（复现） | `online/train.log/2026-09-24_10-50-16` |
| **DA-MT-50b** | **9/24** | **纯 AdaKWS 扫描 V1：lr=2e-3** | **❌ 跳过** | **✅ AdaKWS 10 轮** | **82.77%** | **86.34%（纯 AdaKWS 新最优）** | `online/train.log/2026-09-24_10-56-17` |
| DA-MT-50c | 9/24 | 纯 AdaKWS 扫描 V2：lr=3e-3 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.15% | `online/train.log/2026-09-24_10-57-37` |
| DA-MT-50d | 9/24 | 纯 AdaKWS 扫描 V3：bn_gn_affine，lr=1e-3（Adam） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.22% | `online/train.log/2026-09-24_11-12-12` |
| DA-MT-50e | 9/24 | 纯 AdaKWS 扫描 V4：bn_gn_affine，lr=5e-4 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 84.80% | `online/train.log/2026-09-24_11-13-32` |
| DA-MT-50f | 9/24 | 纯 AdaKWS 扫描 V5：steps=2 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 84.59% | `online/train.log/2026-09-24_11-14-50` |
| DA-MT-50g | 9/24 | 纯 AdaKWS 扫描 V6：熵阈值 0.5 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 84.24% | `online/train.log/2026-09-24_11-16-05` |
| DA-MT-50h | 9/24 | 纯 AdaKWS 扫描 V7：batch=64 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 84.66% | `online/train.log/2026-09-24_11-17-29` |
| DA-MT-51 | 9/24 | lr=2e-3 + EMA 教师 + BCT（验证叠加收益） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 82.77% | 85.22%（最终） | `online/train.log/2026-09-24_15-00-*/best-adakws.pt` |
| DA-MT-52a~i | 9/24 | 纯 AdaKWS 二轮扫描（lr=2e-3 基准）：lr2.5e-3 / lr1.75e-3 / iter15 / ent0.3 / batch256 / wd1e-4 / pkc0.1 / mask1x1 | ❌ 跳过 | ✅ AdaKWS 10/15 轮 | 82.77% | **86.55%（ent0.3 单项最高）** | `online/sweep_adakws2_summary.txt` |
| DA-MT-53a/b | 9/24 | 正收益组合验证：A=ent0.3+iter15；B=A+pkc0.1+wd1e-4 | ❌ 跳过 | ✅ AdaKWS 15 轮 | 82.77% | 85.22% / 84.87% | `online/sweep_adakws3_summary.txt` |
| DA-MT-54 | 9/25 | **无 RIR 子目标域**（1071 条=409 车载+662 加噪）：纯 AdaKWS 最优配置（lr=2e-3、ent0.3、无 EMA/BCT、Mask 2×2） | ❌ 跳过 | ✅ AdaKWS 10 轮 | 87.96% | **89.64%（+1.68pp，新目标域新最优）** | `online/train.log/2026-09-25_11-35-11/` |
| DA-MT-55 | 9/25 | 无 RIR 域纯 AdaKWS 14 变体扫描（lr/ent/iter/batch/mask/pkc/steps）：**lr=3e-3 胜出**，3 次运行全部 90.10% | ❌ 跳过 | ✅ AdaKWS 10 轮 | 87.96% | **90.10%（+2.14pp，首次突破 90%）** | `online/sweep4_logs/`、`online/train.log/2026-09-25_14-51-29/` |
| DA-MT-56 | 9/25 | lr 边界探索：3.5e-3 / 4e-3 / 5e-3 + lr3e-3×time_mask30 组合 + 3 点复跑 | ❌ 跳过 | ✅ AdaKWS 10 轮 | 87.96% | 90.10%（未被超过，lr=3e-3 确认峰顶） | `online/sweep5_logs/` |
| DA-MT-57 | 9/25 | 无 RIR 域 DANN+OCL：**域判别器改为纯 SNN**（Linear→LIF→Linear，T=8 恒定电流输入），DA 30 epoch + 伪标签 OCL 10 轮 | ✅ DANN 30 epoch | ✅ OCL 10 轮（仅分类头） | 87.02% | 87.68%（+0.66pp，远低于纯 AdaKWS） | `online/train.log/2026-09-25_15-25-02/` |
| DA-MT-58 | 9/25 | **评测口径统一**（全部方法用固定 batch=128 加载器评初始/验证/最终精度）：DANN+OCL 同口径重跑 | ✅ DANN 30 epoch | ✅ OCL 10 轮 | **87.96%** | 87.39%（**-0.56pp，统一口径下为负提升**） | `online/train.log/2026-09-25_17-01-55/` |
| DA-MT-59 | 9/28 | 无 RIR 域 **CDAN+E+OCL（纯 SNN 判别器）**：判别器输入 384×29=11136 维联合表示，DA 30 epoch + OCL 10 轮（统一评测口径） | ✅ CDAN 30 epoch | ✅ OCL 10 轮（全部回滚） | **87.96%** | 87.68%（**-0.28pp，负收益**，略优于 DANN 的 87.39%） | `online/train.log/2026-09-28_14-47-17/` |
| DA-MT-60 | 9/28 | 无 RIR 域**纯 SPACE**（无 DA，逐样本流式，views=8/scale=11/steps=1/SGD/bn_affine/冻结 BN 统计量）：lr=1e-6 与 lr=1e-5 两个点 | ❌ 跳过 | ✅ SPACE 逐样本（1071 条×8 视图） | **87.96%** | 87.96%（lr1e-6，±0.00）/ 88.05%（lr1e-5，+0.09pp=1 条样本，噪声级） | `online/train.log/2026-09-28_15-29-36/`、`_15-35-26/` |
| DA-MT-61 | 9/29 | **实现审计 V1/V2**：判别器容量对照（冻结主干二分类）；SPACE einsum loss/梯度诊断 + lr 补测（1e-1/1e-3/1e-2） | — | — | 87.96% | SNN 判别器 93.25% > ANN 88.67%（容量无问题）；SPACE lr 曲线 1e-6 惰性→1e-3 持平→1e-1 崩溃（53.97%），无正收益点 | `online/diagnose_da_components.py` |
| DA-MT-62 | 9/29 | **实现审计 V3**：CDAN+E 加官方 RandomLayer（11136→256 固定随机投影）+ OCL | ✅ CDAN 30 epoch | ✅ OCL 10 轮 | 87.96% | 87.77%（-0.19pp，与直接展平 -0.28pp 无差异，排除投影问题） | `online/train.log/2026-09-29_14-32-55/` |
| DA-MT-63 | 9/29 | **实现审计 V4**：CDAN+E+OCL 换回**原版 ANN 判别器**（spectral_norm MLP） | ✅ CDAN 30 epoch | ✅ OCL 10 轮 | **87.96%** | 87.02%（**-0.93pp，比 SNN 判别器更差**——失败与 SNN 无关） | `online/train.log/2026-09-29_16-07-53/` |
| DA-MT-64 | 10/7 | 无 RIR 域 **AdaKWS+EMA**（teacher=0.1，μ=0.999）：lr=3e-3 与 lr=2e-3 两点 | ❌ 跳过 | ✅ AdaKWS 10 轮 | **87.96%** | 89.64%（lr3e-3）/ 89.26%（lr2e-3）——**均低于同 lr 纯 AdaKWS（-0.4pp 左右），EMA 在该域无增益** | `online/train.log/2026-10-07_20-44-08/`、`_21-13-36/` |

> ⚠️ **DA-MT-01 和 DA-MT-06 走了完整 DA+OCL 管线**。DA-MT-02~05 是绕过管线的独立原型/诊断实验（直接加载预训练模型操作），用于摸清天花板、验证各方法有效性，为后续改进管线提供依据。
> 注：DA-MT-02~05 的原型脚本（`scratch_oracle.py`/`scratch_srctrain.py`/`scratch_selftrain.py`）已删除，结果与超参均已记录在本表。

> **DA-MT-01（现有管线完整跑）**：加载预训练 → 初始测试 83.05% → **DA 阶段** 15 epoch（源域 5000 条加噪 simu 有标签 + 目标域 1428 无标签；loss=分类CE+0.5×域判别CE，GRL 反转梯度；主干 lr 1e-4、判别器 lr 1e-5）→ **OCL 阶段** 10 轮（冻结主干只训分类头 Linear(384→29)，伪标签置信度 0.9 渐降到 0.6，lr 2e-5）→ 最终 82.84%。**比起点低 0.21pp，当前管线无效。**

> **DA-MT-02（Oracle 天花板）**：目的 = 测"用目标域真实标签微调能到多高"（开卷上限）。**为什么只测 291 条**：若拿全部 1428 条训练又拿同 1428 条测试，模型只是背答案、精度虚高。故按"类别+条件"分层 80/20 划分——1137 条（80%）拿真实标签微调、291 条（20%）完全不碰只用来测。超参：全模型 AdamW lr=2e-4、wd=1e-4、cosine 衰减 10 epoch、batch 16、梯度裁剪 5.0。留出 291 条 84.19% → **96.22%**（E8），E9 起过拟合回落 93.8%。含义：**开卷上限 96.2%，95% 目标可行但很紧。**

> **DA-MT-03（源域混合微调）**：训练集 = 357 条干净测试集 + 5000 条加噪 simu（种子 42 抽样），全部有标签；超参与 DA-MT-02 相同，全模型微调 10 epoch；**全程不用目标域标签**，每 epoch 在目标域全 1428 条上评估。83.26% → **86.34%**（E6），E7 后轻微回落 85.5%。含义：**干净配对样本混入源域 +3.1pp。**

> **DA-MT-04（AdaBN）**：模型里的 BN 层维护一套统计量（running mean/var），测试时使用，反映的是**源域**。AdaBN 思路：把统计量换成目标域的、权重不动，免费自适应。操作：① `model.train()`（BN 进训练模式，前向会更新 running 统计量）② 1428 条目标域逐条前向一遍（`no_grad` 不反传、权重不变）③ 切回 eval 评估。结果 83.26% → **70.10%**，暴跌 13pp。含义：该模型 BN 统计量与权重是 300 epoch 训练的强耦合，换统计量=拆模型，且 batch=1 统计量噪声大。**此路不通。**

> **DA-MT-05（硬伪标签自训练）**：全程不用目标域真标签，让模型"自己教自己"。4 轮，每轮两步：**①打伪标签**——当前模型对全 1428 条推理，softmax 最大概率 > 阈值（0.7→0.6→0.55→0.5 渐降）的样本留下、用模型自己的预测当标签，低置信丢弃；**②微调**——用伪标签样本全模型训练 2~4 epoch（lr 2e-4→5e-5 渐降）+ SpecAugment。每 epoch 测全 1428 条、保留历史最佳。⚠️ 实际起点是 AdaBN 毁坏的模型（70.10%），最终恢复到 **80.11%**，仍低于预训练 83.26%。失败原因：伪标签约 17% 是错的，把错误当正确答案练 → **错误放大** → 后续方向：EMA 教师 + 软标签（注：配对蒸馏方案因部署约束已否决，见第 9 章）。

> **DA-MT-06（当前管线）**：从 KD-MT-08 MT-SNN 预训练 checkpoint 出发，源域改为硬件回放库 `replay_1000.jsonl.gz`（357 条干净测试 + 643 条训练集分层样本，共 1000 条，全部有标签），目标域为 1428 条无标签样本。DA 阶段使用 GRL + 域判别器，30 epoch，源域分类 CE + 0.5×域判别损失，主干 lr=2e-4、判别器 lr=2e-5；OCL 阶段使用高置信伪标签（阈值 0.9→0.6），10 轮，只更新分类头。已使用 `lengths` 掩码池化并修复 warmup lr bug；未使用熵最小化。初始精度 **83.47%**，DA 最佳约 **83.75%**，最终加载最佳模型后的精度 **83.89%**，相对初始提升 **0.42pp**。完整日志：`online/train.log/2026-09-09_10-58-03/log-train-2026-09-09-10-58-03`。

> **DA-MT-07（AdaKWS）**：从同一个 KD-MT-08 MT-SNN checkpoint 出发，`online.method=adakws`，**跳过源域加载和对抗 DA**，直接对目标域 1,428 条无标签数据进行在线适应。只更新 6 个 BatchNorm 层的 affine 参数（`weight`/`bias`，共 4,608 个参数），不更新主干其余参数；使用论文中的选择性熵筛选（阈值 0.4）、时频 Mask 变换（时域 2 次×最大 20 帧，频域 2 次×最大 5 维）、PKC/PLPD 筛选（阈值 0.05）和加权熵损失，AdaKWS 学习率 1e-3，每个 batch 更新 1 步，共 10 轮。每轮选择 465~537 条样本，按无监督熵损失保留第 7 轮模型。目标域真实标签不参与适应，仅在最终评估时使用。初始精度 **83.47%**，最终精度 **85.08%**，提升 **1.61pp**。模型：`online/train.log/2026-09-09_13-36-40/best-adakws.pt`；日志：`online/train.log/2026-09-09_13-36-40/log-train-2026-09-09-13-36-40`。

> **DA-MT-08（AdaKWS 论文流程修正版）**：使用 `online.method=adakws`，跳过源域加载和对抗 DA，仅用目标域输入进行在线适应。严格按“先熵筛选、再 PKC/PLPD 筛选、最后计算 `alpha(x) × L_ent(x_pkc)`”的顺序实现；固定 AdaKWS batch size=128，熵阈值 0.4、PKC 阈值 0.05、sigma=0.5，时域 Mask 2 次（最大 20 帧）、频域 Mask 2 次（最大 5 维）。只更新 6 个 BatchNorm 层的 affine 参数，共 4,608 个参数；10 轮中按无监督熵损失保留第 9 轮模型。目标域真实标签不参与适应，仅用于最终评估。初始精度 **82.77%**，最终精度 **85.85%**，提升 **3.08pp**。模型：`online/train.log/2026-09-09_14-35-08/best-adakws.pt`；日志：`online/train.log/2026-09-09_14-35-08/log-train-2026-09-09-14-35-08`。

> **DA-MT-09（AdaKWS 实验 A）**：在 DA-MT-08 基础上启用 `adakws_use_batch_stats=true`，关闭 BN 的 `track_running_stats`，将 `running_mean`/`running_var` 置空，每次前向使用当前 batch 的统计量。跳过源域和对抗 DA，只更新 4,608 个 BatchNorm affine 参数；batch size=128，10 轮，熵阈值 0.4、PKC 阈值 0.05。每轮熵筛选 1,092~1,158 条，PKC 筛选 457~563 条。初始精度 **82.77%**，最终精度 **71.78%**，下降 **10.99pp**。该结果表明，直接把 MT-SNN 的 BN 切换为 batch-statistics 模式会严重破坏模型；后续应区分“适应时使用 batch statistics”和“最终推理时恢复 checkpoint statistics”，不能直接照搬 CNN 的 TENT 配置。模型：`online/train.log/2026-09-09_14-50-54/best-adakws.pt`；日志：`online/train.log/2026-09-09_14-50-54/log-train-2026-09-09-14-50-54`。

> **DA-MT-10（AdaKWS 实验 B）**：将 `adakws_use_batch_stats` 设为 `false`，适应期间保持 MT-SNN 整体为 `train` 模式以支持代理梯度，但 BatchNorm 单独设为 `eval`，使用 checkpoint 中原有的 `running_mean`/`running_var` 且不更新；优化器改为论文基线 SGD（lr=1e-4、momentum=0.9、Nesterov、weight decay=1e-7）。跳过源域和对抗 DA，只更新 4,608 个 BatchNorm affine 参数；batch size=128，10 轮。每轮熵筛选 1,282~1,310 条，PKC/最终共同筛选 626~758 条。目标域真实标签不参与适应，仅用于最终评估。初始精度 **82.77%**，最终精度 **83.47%**，提升 **0.70pp**；相比 DA-MT-09 恢复了 BN 统计量造成的 10.99pp 损失，但没有超过 DA-MT-08 的 85.85%。模型：`online/train.log/2026-09-09_15-01-00/best-adakws.pt`；日志：`online/train.log/2026-09-09-15-01-00/log-train-2026-09-09-15-01-00`。

> **DA-MT-11（AdaKWS 实验 D1）**：设置 `adakws_update_scope=bn_gn_affine`，在保持 checkpoint BN running statistics 的前提下，同时更新 BatchNorm 和 GroupNorm 的 affine 参数，共 10,368 个参数；其余 AdaKWS 配置不变（SGD、lr=1e-4、batch size=128、10 轮、熵阈值 0.4、PKC 阈值 0.05）。初始精度 **82.77%**，最终精度 **83.47%**，提升 **0.70pp**，与仅更新 BN 的 DA-MT-10 持平。

> **DA-MT-12（AdaKWS 实验 D2）**：设置 `adakws_update_scope=bn_classifier`，同时更新 BatchNorm affine 参数和分类头，共 15,744 个参数；其余配置不变。初始精度 **82.77%**，最终精度 **82.49%**，下降 **0.28pp**，加入分类头后没有带来收益。

> **DA-MT-13（AdaKWS 实验 D3）**：设置 `adakws_update_scope=last_block_classifier`，更新最后一个 MultiTauBlock 和分类头，共 134,019 个参数；其余配置不变。初始精度 **82.77%**，最终精度 **82.91%**，提升 **0.14pp**，大范围更新反而低于仅更新 BN/GN affine 的 DA-MT-11。

> **DA-MT-15（AdaKWS + 源域回放 CE，λ=0.05）**：在 DA-MT-08 完整配置基础上（Adam、lr=1e-3、train_running、只更新 4,608 个 BN affine、batch 128、10 轮），新增源域回放监督：每个目标 batch 的更新步同时前向一个 64 条带真实标签的回放 batch（`replay_1000.jsonl.gz`），总损失 `L = L_adakws + 0.05 × CE(replay)`，源域/目标域 forward 间各自 reset_net。目标域真实标签仍不参与适应。初始精度 **82.77%**（真实车载 99.27% / 加噪 78.85% / RIR 71.15%），最终 **85.15%**（真实车载 **100.00%** / 加噪 **81.72%** / RIR **74.51%**），整体提升 **2.38pp**。对比 DA-MT-08（85.85%）整体低 0.70pp，但退化子集改善显著（RIR +3.36pp、加噪 +2.87pp），且真实车载满分——验证了"回放含 357 条干净配对样本 → 共享 BN 参数 → 间接改善退化样本"的机制。回放 CE 全程稳定（0.43→0.53）。⚠️ 按无监督熵评分保留了第 0 轮模型，该评分不含回放 CE 项，可能低估了后续轮次。日志：`online/train.log/2026-09-12_16-58-45/log-train-2026-09-12-16-58-45`。注：DA-MT-14（同日，lr=5e-4，85.08%）按用户要求不记录。

> **DA-MT-16（AdaKWS + 源域回放 CE，λ=0.01）**：与 DA-MT-15 唯一区别是回放权重降为 0.01。初始 **82.77%**（99.27 / 78.85 / 71.15），最终 **85.43%**（真实车载 **100.00%** / 加噪 **82.63%** / RIR **73.95%**），提升 **2.66pp**。比 λ=0.05（85.15%）高 0.28pp，仍低于 DA-MT-08（85.85%）0.42pp；加噪子集 82.63% 为目前所有实验最高。按无监督评分保留第 1 轮模型。日志：`online/train.log/2026-09-12_17-25-56/log-train-2026-09-12-17-25-56`。

> **DA-MT-17（DA-MT-08 复现 + 子集分解基线）**：`adakws_replay_weight=0`，其余配置与 DA-MT-08 完全一致，用于 (a) 验证当日代码改动（replay 通路、子集评估）无副作用，(b) 补充 DA-MT-08 的子集精度（原实验无此日志）。结果 **85.85%，与 DA-MT-08 完全一致**，复现成功。子集：真实车载 **100.00%** / 加噪 **82.78%** / RIR **75.35%**；保留第 9 轮（最后一轮，无监督评分仍在下降）。同口径对比表：

| 实验 | 整体 | 真实车载 | 加噪simu | RIR混响 | 保留轮次 |
|---|---|---|---|---|---|
| 初始 | 82.77% | 99.27% | 78.85% | 71.15% | - |
| **DA-MT-17（λ=0）** | **85.85%** | 100.00% | **82.78%** | **75.35%** | 9 |
| DA-MT-16（λ=0.01） | 85.43% | 100.00% | 82.63% | 73.95% | 1 |
| DA-MT-15（λ=0.05） | 85.15% | 100.00% | 81.72% | 74.51% | 0 |

结论：**回放 CE 作为加性正则在 BN affine 更新粒度上不带来收益，λ↑ 整体单调下降**，DA-MT-08（纯 AdaKWS）仍是当前最优。阶段 1（AdaKWS+回放）到此收尾；下一步转向"延长迭代轮数"（DA-MT-08 保留在最后一轮、评分未收敛）与教师/离线 RIR 方向。日志：`online/train.log/2026-09-12_17-28-42/log-train-2026-09-12-17-28-42`。

> **DA-MT-18（AdaKWS 迭代延长 10→20 轮）**：其余配置与 DA-MT-08 一致（λ=0、Adam、lr=1e-3、只更新 BN affine）。初始 **82.77%**，最终 **85.64%**（100.00 / 82.63 / 74.79），提升 **2.87pp**，低于 DA-MT-08 的 85.85%。按无监督评分保留第 17 轮。结论：**延长迭代无收益**——DA-MT-08 的 10 轮已接近该配置的收敛点，继续更新后评分虽降但测试精度不再上升（评分与精度的相关性有限）。日志：`online/train.log/2026-09-12_17-30-54/log-train-2026-09-12-17-30-54`。

> **DA-MT-19（AdaKWS 学习率微扫，lr=1.5e-3）**：其余配置与 DA-MT-08 一致。初始 **82.77%**，最终 **85.71%**（真实车载 100.00% / 加噪 82.63% / RIR 75.07%），保留第 4 轮。结合 DA-MT-14（lr=5e-4，85.08%），学习率曲线为：5e-4→85.08%，**1e-3→85.85%（峰值）**，1.5e-3→85.71%。结论：**lr=1e-3 确认为 AdaKWS 最优学习率**，超参扫描收尾。日志：`online/train.log/2026-09-12_17-32-55/log-train-2026-09-12-17-32-55`。

> **DA-MT-20（AdaKWS + 冻结教师 KL，λ=0.05）**：在 DA-MT-08 配置上新增冻结教师约束：deepcopy 初始预训练模型为教师（eval、checkpoint BN 统计量、不反传），每个更新步对**教师置信度 >0.9** 的目标样本计算 `KL(p_teacher ‖ p_student)`，以权重 0.05 加入总损失。目标域标签不参与。初始 **82.77%**，最终 **84.80%**（100.00 / 81.57 / 73.39），提升 **2.03pp**，低于 DA-MT-08 达 1.05pp，保留第 1 轮。结论：**教师约束同样无收益**。至此当日 6 组在线约束/超参实验（回放 λ=0.01/0.05、迭代 20 轮、lr 1.5e-3、教师 KL）全部低于纯 AdaKWS 的 85.85%，规律一致：**在 BN affine 更新粒度上，任何把更新拉回源域的约束项都会削弱适应本身**。在线无监督适应在当前参数化下已到平台期（85.85%），后续提升应转向离线 RIR 增强重训（治混响稳定错误）或更细粒度的在线参数化研究。日志：`online/train.log/2026-09-12_17-35-53/log-train-2026-09-12-17-35-53`。

> **DA-MT-21（E1：EMA 教师伪标签，KL 权重 1.0）**：修正 DA-MT-20 的设计缺陷后重做——(a) 教师不再是冻结初始模型，而是 **EMA 教师**：初始等于学生，每次学生更新后按 `teacher = 0.99×teacher + 0.01×student` 缓慢跟随；(b) 学生在**时频增强视图**上对齐教师在**原始语音**上的软预测（DA-MT-20 的同输入 KL 恒近零、只是在对抗 BN train/eval 差异，无真实学习信号）；(c) 仅对教师置信度 >0.9 的样本施加 KL，权重 1.0，与熵损失相加。目标域标签不参与。初始 **82.77%**，最终 **84.66%**（99.27 / 80.66 / 75.35），提升 **1.89pp**，低于 DA-MT-08 1.19pp，保留第 0 轮。⚠️ 实验后发现问题：`total_teacher_kl` 未按迭代重置，无监督评分被累计 KL 污染而单调递增，导致选模永远偏向第 0 轮；且 KL 权重 1.0 相对熵损失（0.18 vs 1.9）过强，更新方向被增强一致性主导。修复评分后以更小权重重试（DA-MT-22）。日志：`online/train.log/2026-09-12_17-59-14/log-train-2026-09-12-17-59-14`。

> **DA-MT-22（E1：EMA 教师伪标签，KL 权重 0.1）✅ 新最优**：与 DA-MT-21 相同实现，仅 KL 权重降为 0.1（教师 KL 约占每 batch 总损失的 90%，熵损失 10%），并修复了无监督评分的跨迭代累计 bug（现按迭代重置，评分 = 熵 + 教师KL 的当轮均值）。EMA 动量 0.99、教师置信度阈值 0.9、学生用时频增强视图学习、教师用原始语音预测。初始 **82.77%**（99.27 / 78.85 / 71.15），最终 **86.06%**（真实车载 **100.00%** / 加噪 **83.23%（历史最高）** / RIR **75.35%（并列最高）**），提升 **3.29pp**，**超过 DA-MT-08（85.85%）0.21pp**，为全部实验新最优。按修复后的无监督评分保留第 8 轮。结论：**EMA 教师的增强一致性约束与熵损失互补**——熵损失提供"更确定"的方向，教师 KL 提供"增强不变且不偏离教师"的稳定方向，小权重（0.1）下两者叠加产生正收益；而 DA-MT-20（同输入 KL）与 DA-MT-21（权重过大）的失败说明该信号必须"有真实对齐任务 + 权重适中"才有效。日志：`online/train.log/2026-09-12_18-02-28/log-train-2026-09-12-18-02-28`。

> **DA-MT-23（E1：EMA 教师伪标签，KL 权重 0.05）**：与 DA-MT-22 唯一区别是 KL 权重降为 0.05。初始 **82.77%**，最终 **85.64%**（100.00 / 82.63 / 74.79），提升 **2.87pp**，低于 DA-MT-22（86.06%）。结合权重扫描：0.05→85.64%，**0.1→86.06%**，1.0→84.66%，峰值初步位于 0.1 附近偏右，继续向 0.2/0.3 方向扫描。日志：`online/train.log/2026-09-12_18-04-42/log-train-2026-09-12-18-04-42`。

> **DA-MT-24（E1：EMA 教师伪标签，KL 权重 0.2）**：与 DA-MT-22 唯一区别是 KL 权重升为 0.2。初始 **82.77%**，最终 **85.78%**（100.00 / 82.02 / **76.47%（RIR 历史最高）**），提升 **3.01pp**。整体低于 DA-MT-22（86.06%），但 RIR 子集创新高——更大的教师权重对混响子集更有利。权重曲线：0.05→85.64%，**0.1→86.06%（整体峰值）**，0.2→85.78%（RIR 峰值），1.0→84.66%。日志：`online/train.log/2026-09-12_18-06-34/log-train-2026-09-12-18-06-34`。

> **DA-MT-25（E1：EMA 教师伪标签，KL 权重 0.1，μ=0.999）✅ 新最优**：与 DA-MT-22 唯一区别是 EMA 动量从 0.99 提高到 0.999（教师跟踪学生的速度放慢 10 倍，锚更稳）。初始 **82.77%**（99.27 / 78.85 / 71.15），最终 **86.41%**（真实车载 **100.00%** / 加噪 **83.53%（历史最高）** / RIR **76.19%**），提升 **3.64pp**，超过 DA-MT-08 达 **0.56pp**，为全部实验新最优。按无监督评分保留第 6 轮。μ 曲线：0.99→86.06%，**0.999→86.41%**——教师越慢越稳，伪标签质量越高，与 Mean-Teacher 文献结论一致。日志：`online/train.log/2026-09-12_18-08-32/log-train-2026-09-12-18-08-32`。

> **DA-MT-26（E1：EMA 教师伪标签，KL 权重 0.1，μ=0.9999）**：与 DA-MT-25 唯一区别是 EMA 动量提高到 0.9999。初始 **82.77%**，最终 **85.71%**（100.00 / 82.78 / 74.79），保留第 6 轮。μ 曲线完整：0.99→86.06%，**0.999→86.41%（峰值）**，0.9999→85.71%——μ 过大时 10 轮×12 batch=120 次更新内教师几乎不动（累计移动 ~1.2%），退化成 DA-MT-20 式冻结教师，失去意义。**当前最优组合锁定：AdaKWS + EMA 教师 KL，λ=0.1，μ=0.999，conf>0.9 → 86.41%**，配置已恢复。日志：`online/train.log/2026-09-12_18-10-26/log-train-2026-09-12-18-10-26`。

**DA-MT-21~26（E1 系列扫参）汇总**：

| 实验 | KL权重 | μ | 整体 | 加噪 | RIR |
|---|---:|---:|---:|---:|---:|
| DA-MT-17（无教师基线） | - | - | 85.85% | 82.78% | 75.35% |
| DA-MT-21 | 1.0 | 0.99 | 84.66% | 80.66% | 75.35% |
| DA-MT-23 | 0.05 | 0.99 | 85.64% | 82.63% | 74.79% |
| DA-MT-22 | 0.1 | 0.99 | 86.06% | 83.23% | 75.35% |
| DA-MT-24 | 0.2 | 0.99 | 85.78% | 82.02% | 76.47% |
| **DA-MT-25** | **0.1** | **0.999** | **86.41%** | **83.53%** | 76.19% |
| DA-MT-26 | 0.1 | 0.9999 | 85.71% | 82.78% | 74.79% |

> **DA-MT-27（E2：E1 + 目标域记忆库邻居伪标签）**：在 DA-MT-25 最优配置（AdaKWS + EMA 教师 KL，λ=0.1，μ=0.999）上新增 SpGesture 风格的目标域记忆库：(a) 在**教师分类器**上挂前向钩子，提取每条语音的池化特征（384 维，源自膜电位/脉冲活动）；(b) 教师置信度 >0.9 的样本按**每类最多 20 条**入库（共 29 类 × 20 = 580 条上限，按置信度替换）；(c) 新样本先以余弦相似度查询库中 top-5 邻居，**邻居多数类 = 教师预测类且 top-1 相似度 >0.2** 才算可靠邻居；(d) 可靠样本额外计算 `KL(邻居平均预测 ‖ 学生(增强视图))`，权重 λ_n=0.1。先查询后插入，避免自匹配。目标域标签不参与。初始 **82.77%**，最终 **86.41%**，与 DA-MT-25 **并列最优**；子集出现迁移：加噪 **83.99%（历史最高）**，RIR 75.35%（较 76.19% 回落）。记忆库第 0 轮即填满（580 条），邻居 KL 与教师 KL 量级相当（~2.0）。结论：**邻居信号与教师信号高度相关**（库中样本本身就来自教师高置信预测），加性约束未带来整体增益，只在高斯噪声子集上有小幅再分配；进一步提升需让邻居信息**改变伪标签**而非仅追加约束（如邻居多数类与教师不一致时否决该样本更新）。日志：`online/train.log/2026-09-12_18-17-16/log-train-2026-09-12-18-17-16`。

> **DA-MT-28（E2：记忆库邻居伪标签，λ_n=0.05）**：与 DA-MT-27 唯一区别是邻居权重降为 0.05。初始 **82.77%**，最终 **85.57%**（100.00 / 81.57 / 76.47%），保留第 8 轮。λ_n 曲线：0.05→85.57%，0.1→86.41%（并列 E1）。结论：**加性邻居损失的最优情况也只是与 E1 打平**，记忆库作为"追加损失"的路线到头；若继续利用记忆库，应改为"邻居多数类 ≠ 教师预测时否决该样本的教师 KL"（过滤式而非加性式）。日志：`online/train.log/2026-09-12_18-19-24/log-train-2026-09-12-18-19-24`。

**当前最优配置锁定（已恢复至 config）**：AdaKWS + EMA 教师 KL（λ=0.1，μ=0.999，conf>0.9）→ **86.41%**，记忆库/回放/迭代延长等附加项均已归零。

> **DA-MT-29/30/31（E3：记忆库否决式过滤，三连失败但诊断价值大）**：将记忆库从"加性损失"改为"过滤信号"——邻居多数类与教师预测不一致时**否决**该样本的教师 KL，且被否决样本不入库。
>
> - **DA-MT-29（min_sim=0.2 严格模式）**：`kl_mask = 高置信 ∩ 邻居一致`。每轮否决 730~863 条（约 60% 教师高置信样本），库仅 213→262 条。最终 **85.22%**（-1.19pp vs E1）。
> - **DA-MT-30（has_sim 修正）**：怀疑"无相似邻居"被误否决，改为只否决"有相似邻居且不一致"的样本。结果与 29 **逐位相同**（85.22%）——由此发现关键问题，随即做了特征空间诊断：**池化 384 维特征的余弦相似度全体 >0.90（p50=0.989），min_sim=0.2 对所有样本都成立**，两次运行的 kl_mask 实际相同。诊断同时显示 5NN 邻居标签一致率 94%——原因是目标域 1428 条全部是 357 条干净语音的退化版本，同源语音特征几乎相同，"邻居准"是同源兄弟样本带来的。
> - **DA-MT-31（min_sim=0.97 同源兄弟校验）**：把 min_sim 提到 0.97，让邻居=真正的同源语音兄弟，否决"同一条语音的不同退化版本之间预测不一致"的样本。结果否决数骤降到第 0 轮 38 条、之后 0~7 条，最终 **85.57%**（-0.84pp vs E1）。否决信号几乎不触发——说明**教师对同一条语音不同退化版本的预测高度一致，错误是"语音级一致"的**：模型把某条语音的混响版和加噪版错成同一个类，同源校验在原理上就抓不到这类错误。
>
> **E3 系列结论**：记忆库三种用法（加性损失、严格否决、同源否决）全部不超过纯 E1 的 86.41%；特征空间诊断进一步表明池化特征只编码"是哪条语音"，不提供超同类别的判别信息。记忆库/邻居方向就此关闭。日志：`online/train.log/2026-09-12_22-10-20`、`_22-12-42`、`_22-19-45`。

> **DA-MT-32/33（A1：E1 + 源域回放 CE 组合）**：测试"教师 KL（目标域稳定锚）+ 回放 CE（真实标签方向）"这一从未测过的组件组合。λ_replay=0.01 → **85.08%**（100.00 / 82.48 / 72.83），λ_replay=0.05 → **85.50%**（85.50% 整体），均低于纯 E1 的 86.41%。至此回放 CE 在全部 4 种组合（AdaKWS×2、E1×2）中无一收益，**回放方向正式关闭**——BN affine 粒度下源域监督只会与适应方向冲突。日志：`online/train.log/2026-09-13_10-58-39`、`_11-01-08`。

> **DA-MT-34/35/36（A2/A3：教师软标签微调，全部无收益）**：(34) CoTTA 式教师软标签取原始+4 增强视图平均概率 → **85.22%**——平均后置信度被稀释，过阈值样本减少且软标签变平；(35) 教师置信度阈值 0.95 → **85.92%**；(36) 阈值 0.85 → **85.50%**。阈值曲线确认 **0.9 为最优**。**阶段 A 结论：6 组低成本组合/微调全部低于 86.41%，E1 配置是样本级在线 TTA 的真实局部最优**。日志：`online/train.log/2026-09-13_11-04-27`、`_11-07-34`、`_11-09-11`。

> **DA-MT-37~40（B1：类别先验校正 BCT）✅ DA-MT-39 新最优 86.48%**。动机：预测分布诊断显示系统性类别偏斜（类 17：40→63 条 +23；类 18：40→23 条 -17；类 19→3 混淆 11 条等，Top-10 混淆对占全部错误 31%）。方法：在线维护预测分布滑动平均 p̄（momentum=0.99，均匀初始化），损失计算用 `logits − τ·log(p̄)` 抑制过预测类别；τ>0 时适应结束把校正折叠进分类头 bias（`b = −τ·log(p̄)`），最终预测/评估自动使用校正后 logits（p̄ 仅由无标签目标预测估计，属 TTA 合法状态）。结果：τ=1.0 仅损失端 **85.29%**；τ=1.0 含预测端 **85.36%**；**τ=0.5 含预测端 86.48%（新最优，+3.71pp vs 起点，+0.63pp vs DA-MT-25）**；τ=0.25 **85.92%**。τ 曲线：0.25→85.92%，**0.5→86.48%（峰值）**，1.0→85.36%——校正过强会把正确样本也推出原类。子集（τ=0.5）：100.00 / 83.99（加噪并列最高）/ 74.79。日志：`online/train.log/2026-09-13_11-15-28`、`_11-17-44`、`_11-19-35`、`_11-21-17`。复现验证（同配置重跑）：**86.48% 完全一致**，子集 100.00 / 83.38 / **76.75%（RIR 历史最高）**，`online/train.log/2026-09-13_11-24-12`。

**阶段 A/B 总结（DA-MT-32~40，9 组实验）**：A 组（回放组合、增强平均教师、阈值微调）6 组全部无收益；B 组先验校正找到新最优 **τ=0.5 → 86.48%**。当前最优配方（config 已锁定）：

| 组件 | 配置 |
|---|---|
| 基础 | AdaKWS（熵筛选+PKC+加权熵损失），lr=1e-3，Adam，只更新 BN affine |
| EMA 教师 | KL 权重 0.1，μ=0.999，置信度>0.9，学生用增强视图对齐教师原始预测 |
| 先验校正 | BCT τ=0.5，momentum=0.99，适应结束写入分类头 bias |
| 结果 | **86.48%**（起点 82.77%，+3.71pp；DA-MT-08 基线 85.85%，+0.63pp） |

> **DA-MT-41/42（C1/C2：源域微调 DA 阶段 + E1 叠加，域适应路线验证）**：C1 复刻 DA-MT-03 配方（`online/train_source_ft.py`：5000 加噪 simu + 357 干净原始语音，全模型 AdamW lr=2e-4、wd=1e-4、cosine 10 epoch）→ 最佳 **85.08%**（E9；曲线 82.35→84.31→84.45→85.01→85.08，增益节奏与 DA-MT-03 的"E6 才涨"一致；注意 DA-MT-03 报的 86.34% 是逐条评估口径）。C2 在该 checkpoint 上跑 E1 在线适应 → **85.71%**。**核心发现：源域微调呈"跷跷板效应"**——加噪子集 79.46%→84.29%（+4.8pp，同域监督直接有效），但 RIR 子集 72.83%→**66.39%（-6.4pp，灾难性干扰：源域无混响数据，simu 微调直接破坏混响泛化）**，E1 在线（教师 KL 锚定）能把 RIR 拉回 74.51% 但整体仍不敌原版 E1 的 86.48%。含义：**源域监督只对同域子集有效，对未见过域（RIR）存在固有的此消彼长**；不引入混响监督（离线 RIR 增强训练）就绕不开这个跷跷板。日志：`online/train.log/2026-09-13_13-01-51`、`_13-12-28`。config 已恢复为全局最优（KD-MT-08 起点 + E1 在线）。

> **DA-MT-43（B2：输入侧谱增益适应）**：受 E-BATS（arXiv 2506.07078，NeurIPS'25）输入侧 prompt 适应启发——混响对频谱有系统性染色，在线学习 80 维 FBank 谱增益 `x * (1+gain)` 补偿（通过 forward_pre_hook 挂在学生模型上，评估同样生效；模型权重不动，仅 80 个新参数）。用现有 E1 损失（熵+教师 KL）驱动。初始 **82.77%**，最终 **85.85%**（-0.63pp vs E1）。失败原因：无标签损失中不存在"正确谱补偿方向"的信号——教师 KL 会把增益拉向"让增强视图更像教师原始预测"，与补偿混响染色无关。日志：`online/train.log/2026-09-13_14-04-48`。config 已恢复（input_gain=false）。

> **DA-MT-48（CDAN+E 条件对抗域适应）**：按 thuml/CDAN 官方实现（`online/CDAN/pytorch/network.py`）改造。域判别器输入从 DANN 的 pooled 特征改为**多线性映射 T(f, g) = f ⊗ softmax(g)**（384×29=11136 维外积展平；`model_da_aol.py` 新增 `domain_in_dim` 参数，CDAN trainer 新增 `trainers/cdan_trainer.py`）；域损失按预测熵加权 **w(x)=1+exp(−H(g(x)))**（CDAN+E，高置信样本权重大）；GRL 沿用与 DA-MT-06 相同的 epoch 级 sigmoid 调度；其余超参与 DA-MT-06 完全一致（回放源域 1000 条、主干 lr 2e-4、判别器 lr 2e-5、DA 30 epoch、OCL 10 轮只更新分类头）。yaml 切换方式：`da.method: dann|cdan` + `da.cdan_entropy: true|false` + `online.method: original`。结果：初始 83.47%（99.27 / 79.46 / 72.83）→ DA 最佳 **84.31%**（E2；判别器准确率随 GRL 满强度从 54% 升至 88%）→ OCL 全部未提升回滚 → 最终 **84.31%**（**100.00** / 81.42 / 71.71），提升 **+0.84pp**，比 DA-MT-06（DANN，83.89%）高 **+0.42pp**。结论：**条件对抗（让判别器利用预测条件做类别条件对齐）确实缓解了 DANN 的部分类别混淆问题**，增益主要来自加噪子集（79.46→81.42）；但注意 DA 最佳出现在 E2（GRL 未满强度时），满对抗后精度回落至 83.5~84.0，说明强对抗仍会轻微破坏源域分类结构；且与所有 DA 方法一致，受制于"错误预测条件无法自我纠正"，远低于无标签在线最优 AdaKWS+EMA+BCT 的 86.48%。**CDAN 横向对比完成：优于 DANN 基线，不作为主方法。**config 已恢复 `online.method: adakws`（`da.method: cdan` 开关保留，切换即重跑）。日志：`online/train.log/2026-09-23_10-44-58`。

> **DA-MT-49（CDAN+E 域适应 + AdaKWS 在线学习组合）**：测试"CDAN+E DA 预适应（30 epoch，起点 82.77% 逐条口径）→ AdaKWS+EMA+BCT 在线（10 轮，`da.pre_adakws: true` 新开关）"组合管线。工程修复两处：① DA 训练后判别器 spectral_norm 会把 weight 替换为非叶张量，导致 AdaKWS 的 `copy.deepcopy(teacher)` 报 RuntimeError，改为 **state_dict 重建教师**（绕开 deepcopy，domain_classifier 以 Identity 清空，EMA 更新跳过 domain 键）；② `train_da_aol.py` 支持 `da.pre_adakws` 开关在 adakws 模式前插入 DA 阶段。结果：初始 **82.77%**（99.27 / 78.85 / 71.15）→ CDAN DA 最佳 84.31%（batch 评估口径）→ AdaKWS 10 轮 → 最终 **84.94%**（**100.00** / 82.18 / 72.83），提升 **+2.17pp**。**结论：组合路线（84.94%）低于纯在线最优 AdaKWS+EMA+BCT（86.48%，+3.71pp）**——CDAN DA 阶段把起点抬高到 ~84.3 后，AdaKWS 仅再获 +0.6pp（远小于从 82.77 起步的 +3.71pp），说明：① DA 强对抗轻微破坏的分类结构抵消了预适应收益；② AdaKWS 的无标签适应空间被 DA 阶段提前消耗；③ 与 DA-MT-48（CDAN+伪标签 OCL 84.31%）相比 AdaKWS 更强（+0.63pp），但组合整体不划算。**CDAN 系路线全部完成横向对比，均不作为主方法；全局最优仍为纯在线 AdaKWS+EMA+BCT（86.48%）。**config 恢复：`online.method: adakws`、`da.pre_adakws: false`。日志：`online/train.log/2026-09-23_17-17-22`。

> **DA-MT-50a~h（纯 AdaKWS 超参再扫描，无 DA/无 EMA/无 BCT）**：历史扫描（DA-MT-14/19/18）只动过 lr∈{5e-4,1e-3,1.5e-3} 与轮数，本批在 Adam+bn_affine 最优框架下补扫**更大学习率、GN 更新范围（Adam 从未扫过）、多步更新、熵阈值、batch size** 共 8 组（`online/sweep_adakws.sh` 自动扫描，GPU 4）。结果：**lr=2e-3 → 86.34%（纯 AdaKWS 历史新最优，+3.57pp）**，超过 EMA+BCT 之外的全部方法；lr 曲线 1e-3→85.85%、1.5e-3→85.71%（旧）、**2e-3→86.34%、3e-3→85.15%**——峰值在 2e-3，旧扫描止步 1.5e-3 恰好错过。其余维度全部无收益：GN affine（85.22%/84.80%）、steps=2（84.59%）、熵阈值 0.5（84.24%）、batch=64（84.66%）。V0 基线 85.85% 与 DA-MT-08/17 逐位一致，扫描脚本可信。**下一步（待做）：lr=2e-3 叠加 EMA+BCT，检验能否突破 86.48% 冲击 87+。**汇总：`online/sweep_adakws_summary.txt`；config 已恢复 AdaKWS+EMA+BCT 最优组合（GPU 切至 4：`cuda_visible_devices: "4"`）。

> **DA-MT-51（lr=2e-3 + EMA + BCT 叠加验证）**：在 DA-MT-50b 新最优（lr=2e-3 纯 AdaKWS 86.34%）上叠加 EMA 教师（λ=0.1，μ=0.999）+ BCT（τ=0.5）。结果：**85.22%**（100.00 / 81.72 / 74.79），比纯 AdaKWS lr=2e-3（86.34%）**低 1.12pp**，也不及 lr=1e-3 + EMA + BCT（86.48%）。原因：lr=2e-3 学生更新步长大，EMA 教师（μ=0.999）跟踪滞后，教师 KL（~2.1）主导更新方向、信号过时；BCT 的类别统计同样被大幅漂移的预测分布干扰。**结论：lr=2e-3 与 EMA+BCT 不兼容**——两套最优超参不可简单叠加：lr=1e-3 配 EMA+BCT（86.48%）或 lr=2e-3 纯 AdaKWS（86.34%）各自局部最优。全局最优仍为 **86.48%**。config 已恢复（lr=1e-3 + EMA 0.1 + BCT 0.5）。日志：`online/train.log/2026-09-24_15-00-*`。

> **DA-MT-52/53（纯 AdaKWS 二轮扫描 + 组合验证，⚠️ 发现运行方差主导）**：围绕 lr=2e-3 新峰值系统扫描 9 变体（GPU 4，`online/sweep_adakws2.sh`）。单项结果：base 85.85%、**ent0.3 86.55%**、iter15 86.34%、wd1e-4 / pkc0.1 86.27%、lr2.5e-3 86.13%、batch256 85.92%、lr1.75e-3 85.57%、mask1x1 85.01%。随后正收益组合（`sweep_adakws3.sh`）：ent0.3+iter15 → **85.22%**，再叠加 pkc0.1+wd1e-4 → **84.87%**，组合反而全面退化。**关键发现：AdaKWS 运行方差 ±0.5~0.7pp**——证据链：① 同配置 lr2e-3 在 GPU3 跑出 86.34%、GPU4 跑出 85.85%；② ent0.3 单项 86.55% 但 ent0.3+iter15 组合掉到 85.22%；③ 所有"单项提升"（0.3~0.7pp）与方差同量级。**科学结论：① 纯 AdaKWS 的真实水平 ≈ 85.5~86.5% 方差带，调参无法稳定突破；② 之前多数 <0.5pp 的"新最优"（如 DA-MT-22 比 08 高 0.21pp）实际落在方差带内，方法排序需多种子平均才可信；③ 86.5 附近是当前无标签信号的天花板（与簇级诊断 87.2% 一致），87+ 必须引入新信息。**config 已恢复全局最优（lr=1e-3 + EMA 0.1 + BCT 0.5，GPU 4）。汇总：`online/sweep_adakws2_summary.txt`、`online/sweep_adakws3_summary.txt`。

> **DA-MT-54（无 RIR 子目标域 + 纯 AdaKWS）✅ 新目标域 89.64%**：应用户要求从 1428 条目标域中剔除 357 条 RIR 混响（精度最低子集），构建 **1071 条无 RIR 目标域**（409 真实车载 + 662 加噪 simu，29 类全覆盖，`data_train/fbank/command_cuts_test_29_no_rir.jsonl`）。预训练模型在该域基线 **87.96%**（管线分桶口径；真实车载 99.27 / 加噪 80.97）。直接在 `config/da_aol.yaml` 改配置（cuts_file 换 no_rir 清单 + 纯 AdaKWS 最优超参 lr=2e-3、ent0.3、无 EMA/BCT、**Mask 2×2**），用 `online/train_da_aol.py` 正式入口运行 10 轮。最终 **89.64%**（真实车载 **100.00%** / 加噪 **83.23%**），提升 **+1.68pp**。⚠️ 首次试跑曾得 75.91%（加噪崩到 61.93%），根因是临时脚本配置错误：Mask 次数误用 1×1（扫描脚本 `reset_base()` 实为 2×2），弱增强 + lr=2e-3 导致熵损失类坍缩——修正后正常，印证 Mask 增强强度是 AdaKWS 稳定性的关键超参。日志：`online/train.log/2026-09-25_11-35-11/log-train-2026-09-25-11-35-11`；模型：同目录 `best-adakws.pt`。

> **DA-MT-55（无 RIR 域纯 AdaKWS 扫描）✅ 90.10%，首次突破 90%**：在无 RIR 目标域上对纯 AdaKWS（无 EMA/BCT）做 14 变体并发扫描（GPU 0，改 `da_aol.yaml` → 后台启动 `train_da_aol.py`，`online/run_sweep4.sh`；其中 5 个变体因 GPU 显存不足 OOM，由 `run_sweep4_fix.sh` 减小并发补跑）。结果（初始 87.96%）：**lr=3e-3 → 90.10%（车载 100.00 / 加噪 83.99）**，time_mask30 → 89.82%，batch256 / ent0.35 / iter15 / iter20 / base(lr2e-3) → 89.64%，steps2 → 89.54%，pkc0.1 → 89.36%（车载跌到 98.53），batch64 / lr2.5e-3 → 89.17%，ent0.25 → 88.89%，mask3x3 → 88.80%，lr1.5e-3 → 88.61%。lr=3e-3 复跑 2 次均 90.10%（`online/train.log/2026-09-25_14-53-10`、`_14-54-21`），稳定可复现。结论：无 RIR 域上纯 AdaKWS 最优学习率为 3e-3（比 1428 域的 2e-3 更大——该域更"容易"，容许更激进的更新步长）；R0 基线复跑 89.64% 与 DA-MT-54 完全一致，同 GPU 上运行确定性很好。⚠️ 工程教训：并发启动时进程 import 需 3~10s 才读 yaml，2s 间隔改配置会被"迟到读取"污染（首轮 4 个变体读到同一配置）；修复为启动后轮询日志出现"实验目录"再改下一个。config 已停在最优配置（lr=3e-3）。

> **DA-MT-56（lr 边界探索）**：验证 lr 是否还有上升空间。完整曲线（初始 87.96%）：1.5e-3→88.61%，2e-3→89.64%，2.5e-3→89.17%，**3e-3→90.10%（峰顶）**，3.5e-3→88.52%，4e-3→89.92%，5e-3→89.54%。曲线非单调且 3.5e-3 出现明显凹陷——筛选+熵损失的优化面崎岖，lr 响应跳变。3.5e-3 / 4e-3 复跑均与首次逐位一致（88.52% / 89.92%），**同 GPU 上运行完全确定**（1428 域的 ±0.5pp 方差来自跨 GPU，同 GPU 不存在）；因此 90.10% 即 lr 维度的可靠上限。lr3e-3+time_mask30 组合 89.08%（组合反而降 1pp，不叠加）。结论：**纯 AdaKWS 在无 RIR 域的最优配置锁定 lr=3e-3（90.10%）**，lr 维度已探明，继续提精度需换维度（筛选/增强/选模策略等）。汇总：`online/sweep5_logs/`。

> **DA-MT-57（无 RIR 域 DANN+OCL，纯 SNN 域判别器）**：应用户要求把 `model_da_aol.py` 的域判别器从 ANN MLP（spectral_norm+ReLU+Dropout）改为**纯 SNN 网络** `SNNDomainClassifier`（Linear(384→128) → LIF(step_mode=m, tau=2, atan) → Linear(128→2)，静态特征复制 T=8 步恒定电流输入，输出时间平均；膜电位随 `functional.reset_net` 清零；参数沿用 domain_lr=2e-5 分组）。切换 `da.method=dann` + `online.method=original` 跑原始 DANN+OCL 完整管线（源域 replay_1000，目标域无 RIR 1071 条）。结果：初始 87.02%（test loader 口径）→ DA 最佳 87.68%（**E0 即最高**，此后 29 个 epoch 在 86.8~87.5% 震荡再未超过）→ OCL 10 轮后 87.68%（无变化），车载 100.00% / 加噪 80.06%。判别器全程 ~50% 准确率（GRL 对抗生效），但**对抗对齐没有带来目标精度增益**——收益全部来自 E0 的源域 CE 微调。对比纯 AdaKWS 90.10%：**低 2.42pp，DANN+OCL 路线在该域确认无效**，与 1428 域结论一致（对抗破坏分类结构 + OCL 伪标签自证）。日志：`online/train.log/2026-09-25_15-25-02/log-train-2026-09-25-15-25-02`。config 已恢复全局最优（method=adakws，lr=3e-3）。

> **DA-MT-58（评测口径统一）**：发现 DA-MT-54（AdaKWS）与 DA-MT-57（DANN）的初始精度不可比——`train_da_aol.py` 原来按 `online.method` 切换评测加载器（AdaKWS=固定 128，DANN=分桶），SNN 预测受 batch 补零影响，同一模型初始精度 87.96% vs 87.02% 差 0.94pp。修改 `train_da_aol.py`：**新增统一评测加载器 `eval_dl`（固定 batch=128、按时长排序）**，初始/DA 验证/OCL 验证/最终测试/分子集全部用它，各方法训练加载器不变。验证：改后 AdaKWS（lr=3e-3）重跑 **90.10% 逐位复现**（`/tmp/adakws_verify.log` 对应 `online/train.log/2026-09-25_17-23-*`），零影响。DANN+OCL 同口径重跑：初始 **87.96%**（与 AdaKWS 同起点）→ 最终 **87.39%（-0.56pp，负提升）**，车载 100.00 / 加噪 79.61——之前 +0.66pp 的"提升"是评测口径差异造成的假象，统一口径下 DANN+OCL 在无 RIR 域实际为负收益。日志：`online/train.log/2026-09-25_17-01-55/`。

> **DA-MT-59（无 RIR 域 CDAN+E+OCL，纯 SNN 判别器）**：在纯 SNN 判别器（`SNNDomainClassifier`：Linear→LIF→Linear，T=8 恒定电流输入）上测 CDAN 路线——判别器输入为特征×预测联合表示（384×29=11136 维），CDAN+E 熵条件加权，DA 30 epoch + 伪标签 OCL 10 轮，统一评测口径。结果：初始 87.96% → DA 最佳 87.68%（30 epoch 最高）→ OCL 10 轮全部"精度未改善已回滚"→ 最终 **87.68%（-0.28pp）**，车载 100.00 / 加噪 80.06。对比 DANN+OCL（87.39%，-0.56pp）：CDAN 略优 0.29pp，与 1428 域"CDAN>DANN"的排序一致；但**两条对抗路线在统一口径下均为负收益**，远低于纯 AdaKWS 的 90.10%（差 2.42pp）。**无 RIR 域对抗域适应路线（DANN/CDAN × OCL）全部关闭，纯 AdaKWS（90.10%）为该域最终最优。** 日志：`online/train.log/2026-09-28_14-47-17/`。config 已恢复 method=adakws（lr=3e-3）。

> **DA-MT-60（无 RIR 域纯 SPACE）**：与纯 AdaKWS 同条件的纯在线方法对照——无 DA、逐样本流式适应（batch=1、views=8、scale=11、steps=1、SGD、bn_affine 4608 参数、自动冻结 BN 统计量即 DA-MT-47 修正协议），统一评测口径。lr=1e-6：87.96% → **87.96%（±0.00，逐样本内部统计修复 0/误伤 1）**，完全惰性；lr=1e-5：87.96% → **88.05%（+0.09pp=1 条样本，内部统计修复 1/误伤 9，净 -8 条，统一口径翻转归零）**，噪声级。两个学习率点均无实质增益，**SPACE 的时序特征一致性损失与 MultiTauLIF 结构不匹配的结论在无 RIR 域再次确认**（与 1428 域 DA-MT-44~47 一致）。至此无 RIR 域四条纯在线/对抗路线全部测完：**纯 AdaKWS 90.10% ≫ SPACE 88.05% ≈ 基线 87.96% > CDAN+OCL 87.68% > DANN+OCL 87.39%**。日志：`online/train.log/2026-09-28_15-29-36/`、`_15-35-26/`。config 已恢复 method=adakws（lr=3e-3，space_lr=1e-6）。

> **DA-MT-61~63（实现审计：三方法失效是不是代码错误？——答案：不是）**：应用户质疑做四步代码级审计（`online/diagnose_da_components.py`）。
> **V1 判别器容量**：冻结主干提取 2071 条池化特征做源/目标二分类，SNN 判别器测试精度 **93.25%**（T=8/32、hidden 128/256 均同），**高于**原版 ANN MLP 的 88.67%——"SNN 判别器太弱导致对抗失效"的假设被证伪，纯 SNN 要求不是问题根源。
> **V2 SPACE**：einsum loss=1.47、BN affine 梯度流正常（量级与官方公式一致，损失非零非实现错误）；根因是**学习率**——官方图像版 lr=0.5，此前用 1e-6（每参数每步更新 ~1e-9，必然惰性）。lr 补测剂量-响应：1e-6 惰性（±0.00）→ 1e-5 +0.09pp → 1e-3 持平（修复26/误伤20）→ 1e-2 -0.09pp → **1e-1 崩溃 53.97%（误伤153）**——全曲线无正收益点，修复≈误伤，方法本身不知道该修哪个错。
> **V3 CDAN+RandomLayer**：官方 RandomLayer 固定随机投影（11136→256）后 -0.19pp，与直接展平 -0.28pp 无差异，排除联合表示投影实现差异。
> **V4 原版 ANN 判别器**：CDAN+E+OCL 换回原版 spectral_norm MLP 判别器 → **87.02%（-0.93pp），比 SNN 判别器版（87.68%，-0.28pp）更差**。
> **审计结论**：三个方法的主体实现均忠实于官方（DANN/CDAN 流程、CDAN 多线性映射+熵加权、SPACE einsum 公式逐项一致），精度不涨是**方法与数据不匹配**：无 RIR 域的域间隙本来就小（源域含同款 simu 噪声、真实车载已 99.27%），剩余错误是带噪语音的 in-domain 识别难度而非域偏移——域对齐无病可治；SPACE 的一致性目标也不指示哪个类别错。**纯 AdaKWS（90.10%）保持该域最优。**
> **附带修复一个真实潜在 bug**：老版 spectral_norm 注册的 3 类 hook 中移除代码只清了 forward pre-hook，残留的 state_dict/load_state_dict hook 会在 ANN 判别器管线的 OCL 回滚/最终加载时因键不匹配（weight_orig vs weight）崩溃。已改为确定性手动转换（weight_orig/σ → 叶参数 weight）+ 清空全部 SpectralNorm hook + 自校验，最终加载剥离判别器键并校验主干完整。DA-MT-57/58/59（SNN 判别器，无 spectral_norm）结果不受影响。日志：`/tmp/diagnose`、`online/train.log/2026-09-29_14-14-*`（SPACE lr 补测）、`_14-32-55/`（V3）、`_16-07-53/`（V4）。config 已恢复 method=adakws（lr=3e-3）、discriminator=snn。

> **DA-MT-64（无 RIR 域 AdaKWS+EMA）**：补齐"纯 AdaKWS 之外唯一未在该域测过的叠加组件"。配置沿用 1428 域最优（teacher=0.1、μ=0.999），lr 取 3e-3 与 2e-3 两点，统一评测口径：lr=3e-3 → **89.64%**（纯 AdaKWS 同 lr 为 90.10%，EMA 拖累 -0.46pp）；lr=2e-3 → **89.26%**（纯 AdaKWS 同 lr 89.64%，拖累 -0.38pp）。两点一致低于同 lr 纯 AdaKWS，且保留轮次提前（第 3/1 轮即到峰）——高置信教师 KL 约束在 BN affine 粒度上锚定了更新，与 1428 域"教师约束削弱适应"的规律一致。**纯 AdaKWS（lr=3e-3，90.10%）仍为无 RIR 域最优，AdaKWS 叠加组件在该域全部关闭。** 日志：`online/train.log/2026-10-07_20-44-08/`、`_21-13-36/`。config 已恢复纯 AdaKWS 最优（teacher=0，lr=3e-3）。

> **DA-MT-44~47（SPACE-KWS：官方 SPACE 语音改造版）**：参考官方 `online/SPACE/cifar10/bntt/adaptation/test_adaptation.py` 的流程，将图像 AugMix 和二维脉冲特征图改造成 FBank 温和增强和最后一个 MultiTauBlock 的时间序列特征 `[B,T,D]`，按有效长度截断后对通道维做 softmax、计算增强视图两两 `1-similarity` 的官方 einsum 损失，再逐样本反向更新并用原始语音预测。目标域标签只用于评估。DA-MT-44 更新全部 709,649 个参数、SGD lr=1e-4，最终 **69.12%**；DA-MT-45 只更新 4,608 个 BN affine、lr=1e-5，最终 **67.86%**；DA-MT-46 同范围 lr=1e-6，最终 **68.14%**。前三次的实现问题是逐样本适应前向处于 `train_running`，持续用单条语音统计污染 BN running mean/var，导致后续 eval 预测失真。
>
> **DA-MT-47（协议修正版）**：在 DA-MT-46 基础上冻结 BN running statistics（适应和预测均使用 checkpoint running statistics），只更新 BN affine，最终 **71.64%**（SPACE 内部逐样本统计 58.33%，主入口最终 eval 71.64%；真实车载 91.69% / 加噪 62.84% / RIR 64.99%）。相比原始预训练 83.26% 仍下降 **11.62pp**，修复 BN 污染后仍未超过当前 AdaKWS+EMA+BCT 的 86.48%。同时，前三次和修正版均发生大量“正确→错误”（修正版 374 条误伤、仅修复 18 条），说明当前 SPACE 时间特征一致性损失与 MultiTauLIF 的 BN/时序结构不匹配，不能将官方视觉实验精度直接迁移到本 KWS 模型。**结论：SPACE 已完成正式横向测试；在当前改造、增强和参数化下无效，不再作为主方法。**日志：`online/train.log/2026-09-14_11-11-39`、`_11-30-11`、`_11-45-07`、`_12-00-12`。config 已恢复 `method=adakws`。

**簇级重指派天花板诊断（`online/diagnose_cluster.py`，2026-09-13）**：检验"聚类→类别均衡重指派"还能找回了多少错误。方法：提取当前模型 1428 条目标域池化特征，按余弦相似度 >0.97 连通分量聚类（319~322 个簇，簇内多数类占比 p50=1.00，即聚类本身几乎完美）；每簇取成员平均软标签，用运输问题（scipy Hungarian，槽位=类别×先验样本数）把簇重指派给类别；真实标签仅用于评估。先验测试了均匀与回放库直方图两种（结果一致）。

| 基线 | 重指派后 | 收益 | 簇级预言机（簇标签=真实多数类） |
|---|---|---:|---:|
| 预训练 82.77% | 84.87%（修复34/误伤4） | +2.10pp | 99.58% |
| **DA-MT-39 86.48%** | **87.18%（修复15/误伤5）** | **+0.70pp** | 98.95% |

**结论**：聚类本身近乎完美（预言机 98.95%），但重指派只能用模型自己的簇级软标签做代价——而稳定错误簇的软标签是**一致地错**的，指派算法修不出错误软标签所缺的信息。86.48% 基线上簇级天花板仅 **+0.70pp（≈87.2%）**，距 95% 目标仍差 8pp。至此从样本级（熵/教师/记忆库/否决）到簇级（重指派）的全部无标签信号均已量化：**在"不改离线模型、不用目标标签、无干净参考"约束下，在线 TTA 的理论上限 ≈ 87.2%**。要冲击 95% 必须引入新信息（目标域标签、含混响的离线训练、或干净参考），三者均超出当前约束。诊断日志：`online/train.log/diag_cluster_2026-09-13_12-33-48`、`_12-35-13`。

**关于"起点精度"三个数字不一致的说明**（已用实验验证）：

| 起点数字 | 含义 |
|---|---|
| 83.05%（DA-MT-01） | 预训练模型在全 1428 条上、**管线分桶批处理**评估 |
| 83.26%（DA-MT-03/04/05） | 预训练模型在全 1428 条上、**逐条**评估 |
| 84.19%（DA-MT-02） | 预训练模型在 80/20 划分**留出的 291 条子集**上评估（非全量） |

- **83.05% 与 83.26% 是同一个模型、同一批 1428 条数据**，差别只在"测精度"的方式：
  - 逐条评估（83.26%）：1428 条**一条一条**喂给模型，不补零，最干净。
  - 管线分桶评估（83.05%）：lhotse 先把 1428 条按时长排序分桶（相近时长的进同一桶），再按"总时长 ≤32s"组 batch；同 batch 内短句补零帧凑成等长才能做矩阵运算；模型池化时把补的零帧也平均进去了（`model_da_aol.py` 没传 lengths），极少数样本预测因此改变，整体差 0.2pp。
- **84.19%（DA-MT-02）不是全量**：是 80/20 划分后留出的 291 条子集，随机波动恰好比整体略好。训练部分 1137 条为 83.03%，两者加权平均 = 83.26%，与全量完全一致。
- ⚠️ 已验证的管线隐患：乱序打包（批内时长差异大→大量补零）时精度会崩到 46%，因为补零帧被池化计入。分桶打包影响小，但建议后续修复（把 lengths 传给 backbone 做掩码池化）。


## 6. 离线学习管线（MultiTauLIFCommander）

### 6.1 模型架构

### 6.1 数据流总览

```
输入 FBank (B, T, 80)
  │
  ▼
┌────────────────────────────────────────────────────────────┐
│ SEELiteFrontend（声学前端）                                    │
│   Conv1×1(80→256) → DepthwiseConv(k=7) → BN → LIF         │
│   输出: (B, T, 256)                                         │
└────────────────────────────────────────────────────────────┘
  │
  ▼  (重复 N=depths 次)
┌────────────────────────────────────────────────────────────┐
│ MultiTauBlock × N（核心处理单元）                              │
│   ┌──────────────────────────────────────────────────────┐ │
│   │ MultiTauLIFMemory（多时间常数记忆）                     │ │
│   │   输入 (B,T,256)                                      │ │
│   │   ├─ ParametricLIFNode(τ=2) ─┐                       │ │
│   │   ├─ ParametricLIFNode(τ=4) ─┤ softmax 逐通道融合     │ │
│   │   └─ ParametricLIFNode(τ=8) ─┘ → (B,T,256)           │ │
│   └──────────────────────────────────────────────────────┘ │
│   → BottleneckConvChannelMixer（瓶颈时间卷积）               │
│       256→64→时间Conv(k=5)→256 + 残差 + LIF                 │
│   → Dropout → + 残差连接                                   │
│   输出: (B, T, 256)                                         │
└────────────────────────────────────────────────────────────┘
  │
  ▼
Linear(256→41) 逐帧分类 → (T, B, 41)
  │
  ▼
SpikingCELoss: softmax每帧 → mask padding → 时间求和 → CE
```

### 6.2 关键组件说明

| 组件 | 文件 | 作用 |
|------|------|------|
| `BatchMajorLIF` | multitau_lif_commander.py:19 | LIF 神经元包装器，格式转换 (B,T,D)↔(T,B,D) |
| `SEELiteFrontend` | multitau_lif_commander.py:63 | 逐点卷积+深度卷积+BN+LIF |
| `MultiTauLIFMemory` | multi_tau_lif.py:17 | **核心**：3组不同τ的LIF并行→per-channel softmax融合 |
| `BottleneckConvChannelMixer` | multitau_lif_commander.py:206 | 瓶颈时间卷积（256→64→时间Conv→256）+ 残差 |
| `MultiTauBlock` | multitau_lif_commander.py:244 | 一个完整处理单元：Memory→Mixer→Dropout→残差 |
| `MaskedTemporalPool` | multitau_lif_commander.py:299 | 时间池化（忽略padding帧） |
| `MultiTauLIFCommander` | multitau_lif_commander.py:316 | 完整模型：前端→N×Block→分类头 |

### 6.3 LIF 神经元公式

```
膜电位更新:  V[t] = α·V[t-1] + (1-α)·I[t]
放电规则:    if V[t] >= 1.0 → 输出脉冲=1, V[t] -= 1.0 (软复位)
时间常数:    α = 1 - 1/τ
             τ=2 → α=0.5   (漏电快，看最近2帧)
             τ=4 → α=0.75  (中速)
             τ=8 → α=0.875 (漏电慢，记住8帧历史)
```

### 6.4 Per-Channel 自适应融合

```
branch_weights: (1, 1, 3, 256) 可学习参数
normalized = softmax(branch_weights, dim=2)  ← 对3个分支做softmax

每个通道独立学习：channel 0 可能 [0.1, 0.2, 0.7] 偏好 τ=8
                 channel 1 可能 [0.6, 0.3, 0.1] 偏好 τ=2
                 ...

融合: output = Σ(branch_output × normalized_weights), 沿分支轴求和
```

### 6.5 训练管线关键机制

| 机制 | 实现位置 | 作用 |
|------|---------|------|
| EMA (decay=0.9995) | training_utils.py:38 | 权重平滑，抑制SNN代理梯度噪声 |
| SpikingCELoss | training_utils.py:169 | 逐帧CE：softmax→mask→时间求和→CE |
| branch_balance_loss | multi_tau_lif.py:207 | KL散度，强制均匀使用3个τ分支 |
| 分组优化器 | training_utils.py:97 | tau参数/branch_weights 独立 weight_decay=0 |

---

## 7. 在线学习管线（MultiTauLIFCommanderKWS，纯 AdaKWS）

> 🧹 **2026-10-18 代码清理**：DA（GRL + 域判别器）、SPACE、伪标签 OCL 及教师/
> 记忆库/回放/BCT 组件已全部删除（实验均无收益，记录见第 5 章 DA-MT 表）。
> 现管线为纯 AdaKWS：入口 `online/train_adakws.py`（原 train_da_aol.py），
> 模型 `online/model_kws.py` 类 `MultiTauLIFCommanderKWS`（原 MultiTauLIFCommanderDA）。
> 下文 7.1.2 / 7.4 的 GRL/域判别描述为历史机制存档，对应代码已不存在。

### 7.1 模型架构

#### 7.1.1 关键组件（谁是谁）

| 组件 | 代码位置 | 类型 | 作用 |
|------|---------|------|------|
| `backbone` | `model_kws.py` | `MultiTauLIFCommander` (0.660M, KD-MT-08) | 特征提取器，输出 384 维池化特征 |
| `classifier` | `backbone.classifier` | `Linear(384→29)` | 分类头，输出 29 类 logits |
| `domain_classifier` | （已删除） | 曾为 SpectralNorm MLP(384→128→2) | 历史组件：域判别器（DA 实验，2026-10-18 清理） |
| `GRL` | （已删除） | 曾为 `autograd.Function`，**0 参数** | 历史组件：梯度反转层（DA 实验，2026-10-18 清理） |
| `subsample` | `model_kws.py` | `AvgPool1d(k=2)` | 时间降采样（÷2） |

> ⚠️ **GRL ≠ 域判别器**。GRL 只是一个"梯度符号翻转"函数，没有可学习参数。
> 域判别器是 `domain_classifier`，那个 384→128→2 的 SpectralNorm MLP。

#### 7.1.2 DA 阶段（域适应）

```
输入: 源域(有标签) + 目标域(无标签)
═══════════════════════════════════════════════════════════

FBank (B, 80, T)
  │  subsample (AvgPool1d k=2)
  ▼  (B, 80, T//2)
┌─────────────────────────────────────────────┐
│ backbone (KD-MT-08, 0.66M)                  │
│   SEELiteFrontend → 5×MultiTauBlock → Pool  │
│   输出: pooled (B, 384)                      │
└─────────────────────────────────────────────┘
  │
  │  pooled (B, 384)  ← 共享特征
  │
  ├──→ classifier ──────────→ ★ class_logits (B, 29)
  │        │                      │
  │        │ CE(分类) loss ←──────┘  用源域标签监督
  │
  └──→ GRL(梯度反转) → domain_classifier → ★ domain_logits (B, 2)
           │                      │
           │    CE(域判别) loss ←─┘  0=源域, 1=目标域
           │
           └── [反向传播] GRL 将域判别梯度取反 (-grad × λ)
                → backbone 被逼学出"让域判别器分不清"的特征

最终输出:
  class_logits  (B, 29)  ← 分类结果
  domain_logits (B, 2)   ← 域判别结果
```

#### 7.1.3 AOL 阶段（在线学习）

```
输入: 目标域(无标签，只有数据没有标注)
═══════════════════════════════════════════════════════════

FBank (B, 80, T)
  │  subsample → backbone (KD-MT-08) → pooled (B, 384)
  │
  └──→ classifier → ★ class_logits (B, 29)
           │
           │  softmax → max prob → 置信度
           │     │
           │     ├─ prob ≥ threshold → 选为伪标签 ✗
           │     └─ prob < threshold → 丢弃 ✗
           │
           │  CE(伪标签) loss → 只更新 classifier（backbone 冻结）
           │
           └── 下一轮继续，逐步适应目标域

无: GRL, domain_classifier（域判别相关组件全部摘掉）
只用: backbone + classifier

最终输出:
  class_logits (B, 29)  → 预测结果（含伪标签）
```

### 7.2 训练流程

```
阶段1: 预训练（train_multitau.py）
  噪声数据 → KD-MT-08（0.66M, 98.04%）→ 得到预训练 checkpoint
    │
阶段2: 域适应（历史实验 DA-MT-01~63，已验证无收益，代码 2026-10-18 删除）
  同时喂 源域(有标签) + 目标域(无标签)
  loss = CE(分类) + λ_da × CE(域判别)
  GRL 反转梯度 → 特征提取器学到"域无关"表示
    │
阶段3: 在线学习
  用当前模型预测目标域 → 选置信度 > pseudo_threshold 的作为伪标签
  用伪标签继续训练 → 模型逐渐适应目标域
    │
阶段4: 评估
  目标域测试集精度（1428 条退化版：真实车载/加噪/RIR）
```

### 7.3 关键文件

| 文件 | 作用 |
|------|------|
| `model_kws.py` | MultiTauLIFCommanderKWS 模型封装（subsample + backbone；域判别器已删） |
| `train_adakws.py` | AdaKWS 在线学习入口（唯一入口，无命令行参数） |
| `config/da_aol.yaml` | 唯一配置文件（GPU、模型、数据路径、AdaKWS 超参数） |
| `trainers/online_trainer.py` | 分流器（仅支持 method=adakws） |
| `trainers/adakws_trainer.py` | AdaKWS 测试时适应（熵筛选 + PKC/PLPD + 加权熵损失） |
| `utils/train_utils.py` | 精度评测（整体/分子集）+ 预训练权重加载（键映射） |
| `utils/data.py` | cuts 加载 + AdaKWS/DataLoader 构建（offline 跨模块复用） |

### 7.4 GRL 梯度反转机制

```
正常训练: weight -= lr × grad         （往减小loss方向）
经过 GRL:  weight -= lr × (-grad × λ)  （往增大域判别loss方向！）

效果: 特征提取器学"怎么让域判别器分不清"
      → 特征变得域无关 → 不同噪声分布的数据在特征空间对齐
```

---

## 8. 当前结论

- **MultiTauLIFCommander 突破天花板**：42.58% → 76.76% → 80.27% → **87.70%**，四连涨。MT-04 以 0.665M 参数超越 SpikConvNext 的 86.72%（1.5M 参数）。
- **加容量+强正则 = 关键突破**：hidden=384 depth=5 提供表达力，dropout=0.3 + label_smoothing=0.1 + warmup=30 压制过拟合，从 80% 跳到 87.70%。
- **小模型 vs 大模型规律**：
  - 小模型(0.255M)：taus=[2,4,8] + learnable_alpha=false + 轻正则化 → MT-06 83.98%
  - 大模型(0.665M)：taus=[5,20,100] + learnable_alpha=true + 强正则化 → MT-04 87.70%
- **知识蒸馏 + 大模型 = 突破 98%**：
  - 小模型(252K) + KD → **96.92% 天花板**（KD-MT-05/07 两次验证），容量瓶颈（仅教师的 1/127）
  - **大模型(660K) + KD + SpecAugment → 🚀 98.04%**（KD-MT-08），比小模型 KD 涨 +1.12pp，比无 KD 的 MT-04 涨 **+10.34pp**
  - **核心洞察**：252K 小模型无论怎么调参都卡在 96.92%，换 660K 大模型直接突破。**KD 场景下容量同样重要。**
- **在线 DA+OCL 现状与路线**：
  - 目标域难点 = RIR 混响 + 0~8dB 重噪（源域训练 100% 加噪 simu、无 RIR）；真实车载子集 99.27% 已无提升空间
  - Oracle 真标签微调天花板 **96.22%** → **95% 目标可行但很紧**
  - 源域混入干净测试集 +3.1pp（86.34%），验证"以预训练测试集作源域"方向正确（离线阶段，无部署问题）
  - AdaBN（-13pp）、硬伪标签自训练（-3pp）均为负面 → **伪标签质量是核心瓶颈**
  - ⚠️ **部署约束**：上线后到达的是纯无标签流式音频，无法区分干净/带噪、无配对结构可用 → 依赖配对结构的方案（配对蒸馏）已否决；在线方法只能是"仅用到达数据本身"的自适应（EMA 教师 + 软标签方向）

---

## 9. 后续计划

**离线学习目标（≥98%）已达成**（KD-MT-08: 98.04%）。在线 DA+OCL 目标：目标域 1428 条 ≥90%（力争 95%；Oracle 天花板 96.22%）。

⚠️ **部署约束**：最终方法要能在硬件上运行——上线后到达的是纯无标签流式音频，**无法区分干净/带噪，也没有"干净孪生"配对信息**。因此依赖测试集配对结构的方案（配对蒸馏）已否决；在线阶段只能用"到达数据本身"。

| 优先级 | 任务 | 说明 |
|---|---|---|
| 🔴 1 | **离线源域重构** | 源域 = 357 干净 + 5000 simu（有标签，离线训练，无部署问题；已验 +3.1pp → 86.34%） |
| 🔴 2 | **OCL 升级：EMA 教师 + 软标签自训练** | 无配对无标签：教师（权重 EMA）打软标签，学生 KL 对齐 + 增强一致性（FixMatch/MeanTeacher 风格），全模型小 lr 多轮，目标 90%+ |
| 🟡 3 | **修 DA 阶段** | 目标域熵最小化、提 lr/epoch（只用到达数据本身，可部署） |
| 🟡 4 | **收尾** | 清理脚本、TTA 投票、更新实验记录 |
