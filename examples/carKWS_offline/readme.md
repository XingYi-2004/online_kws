# carKWS_offline — 中文车载 29 类指令离线学习（ANN 教师 + SNN 蒸馏）

基于 AlphaSNN 框架 Lightning 管线复现：

| 阶段 | 模型 | 目标精度 | 超参数来源 |
|---|---|---:|---|
| ANN 教师 | TdnnLstm（32.10M） | 99.72% | 原版训练日志（tdnn_lstm_6ann_768_new2） |
| SNN 学生 | MultiTauLIFCommander（0.660M） | **≥98%** | experiment4（raw 98.04% E224 / EMA 98.04% E253） |

## 目录

```
conf/ann_teacher.yaml   ANN 教师训练
conf/offline_kd.yaml    SNN 知识蒸馏
local/export_kws.py     Lightning ckpt → 在线兼容格式（{"model": ...}）
local/eval_kws.py       固定口径评测（1500s / 30 buckets / seed 312）
data/                   command_cuts_{train,test}_29.jsonl + command_feats_{train,test}/
```

## 环境

```bash
conda activate snn
cd AlphaSNN
```

## 1. ANN 教师训练

```bash
python -m modules.launch cmd=train conf=examples/carKWS_offline/conf/ann_teacher.yaml version=ann01
```

| 超参数 | 值 |
|---|---|
| 优化器 | AdamW lr=2e-4，wd=5e-4 |
| 调度 | LinearLR 0.1→1.0 ×3 epoch + Cosine(T_max=17, η=1e-6) |
| Epoch / seed | 20 / 42（每 epoch fix_random_seed(42+epoch)） |
| 数据打包 | DynamicBucketing 200s / 30 buckets |
| SpecAugment | warp80，时遮罩 100×10，频遮罩 27×2 |
| 损失 | CrossEntropy |

产物：`exp/logs/carKWS_ANN/ann01/checkpoints/epoch={0..19}.ckpt`（全量保存）。
教师导出（对应原版 decode_ann --epoch 17 --avg 9）：

```bash
python examples/carKWS_offline/local/export_kws.py --mode teacher \
    --ckpt-dir exp/logs/carKWS_ANN/ann01/checkpoints \
    --start-epoch 9 --end-epoch 17 --out kd_teacher/pretrained_ft.pt
```

## 2. SNN 知识蒸馏

```bash
python -m modules.launch cmd=train conf=examples/carKWS_offline/conf/offline_kd.yaml version=kd01
```

| 超参数 | 值 |
|---|---|
| 模型 | MultiTauLIFCommander depths=5 hidden=384，taus[2,4,8]+learnable，AvgPool×2 |
| 教师 | TdnnLstm（`kd_teacher/tdnn_ann_29class.pt`，冻结），α=0.7，T=10.0 |
| 损失 | (1−α)·CE(ls=0.1) + α·T²·KL + 0.01·branch_balance |
| 优化器 | AdamW lr=2e-3，wd=5e-3（tau/branch 分组同 wd） |
| 调度 | warmup 30 epoch + cosine（min_lr=0），300 epoch |
| EMA | decay=0.9995（框架 torch_ema，验证/测试时应用） |
| SpecAugment | 时≤15×1，频≤5×1（批级，仅训练） |
| 梯度裁剪 | 5.0 |
| 数据打包 | DynamicBucketing 1500s / 30 buckets，seed 312 |
| 双通道验证 | 每 epoch 分别评 raw（valid/acc）与 EMA（valid/acc_ema） |

产物（`exp/logs/carKWS_SNN_KD/kd01/checkpoints/`）：
- `best-acc.ckpt` — valid/acc（raw）最佳 → 对应历史 best-acc.pt
- `best-ema.ckpt` — valid/acc_ema 最佳 → 对应历史 best-ema.pt
- `last.ckpt` — 最后一轮

## 3. 导出 + 固定评测

```bash
python examples/carKWS_offline/local/export_kws.py \
    --ckpt exp/logs/carKWS_SNN_KD/kd01/checkpoints/best-acc.ckpt \
    --ckpt-ema exp/logs/carKWS_SNN_KD/kd01/checkpoints/best-ema.ckpt

python examples/carKWS_offline/local/eval_kws.py \
    --ckpt exp/logs/carKWS_SNN_KD/kd01/checkpoints/export/best-acc.pt \
    --ckpt-ema exp/logs/carKWS_SNN_KD/kd01/checkpoints/export/best-ema.pt
```

⚠️ 评测口径敏感：`num_buckets` 的 lhotse 默认是 10，必须 30（差 1~2pp 的历史教训）。
迁移金标准：旧 experiment4 权重 + 本脚本 → 350/357 = 98.04%。

## 4. 进入在线学习

导出的 `export/best-ema.pt` 路径填入
`examples/carKWS_online/conf/online_adakws.yaml` 的 `paths.pretrained_model`，
见 `../carKWS_online/readme.md`。

## 已知差异（相对 experiment4 重建版）

- 训练循环换成 Lightning（框架提供 EMA/checkpoint/调度器），RNG 消耗顺序不同：
  目标是"同配方复现 ≥98%"，非逐位一致。
- 训练批顺序由 DynamicBucketingSampler(seed=0) 内部状态决定（与重建版同参数）。
- 验证集即 357 条干净测试集（与原版/重建版相同的口径限制），报告结果时需注明。
