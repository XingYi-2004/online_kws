# carKWS_online — 中文车载 29 类指令在线学习（AdaKWS 测试时适应）

加载离线 SNN checkpoint，对无标签目标域（`data/command_cuts_test_29_no_rir.jsonl`，
1071 条 = 409 真实车载 + 662 加噪 simu）做 AdaKWS 测试时适应。
目标：**≥90%**（当前冠军 90.20%，experiment4 EMA 起步）。

## 目录

```
conf/online_adakws.yaml   唯一配置（预训练路径 / 目标域 / AdaKWS 超参数）
local/train_adakws.py     在线学习主入口（算法在 alpha/asr/kws/online/trainers/）
data/                     目标域清单 + command_feats_test.lca
```

## 使用

```bash
conda activate snn
cd AlphaSNN

# 1) 先完成离线训练并导出（见 ../carKWS_offline/readme.md），
#    然后把导出路径填入 conf/online_adakws.yaml 的 paths.pretrained_model

# 2) 运行（入口自动读取本目录配置，无命令行参数）
python examples/carKWS_online/local/train_adakws.py
```

## 关键超参数（冠军配置，勿改）

| 项 | 值 |
|---|---|
| 预训练 | 离线导出 `best-ema.pt`（干净测试 98.04%） |
| 目标域 | 1071 条 no-RIR（`command_cuts_test_29_no_rir.jsonl`） |
| 迭代 / batch / steps | 10 轮 / 128 条 / 每 batch 1 步 |
| 优化器 | Adam lr=2.2e-3，wd=0 |
| 熵阈值 / PKC 阈值 | 0.3（冠军值）/ 0.05 |
| BN 模式 / 更新范围 | train_running / bn_affine（4,608 参数） |
| Mask | 时≤20 帧×2，频≤5 带×2 |
| seed / grad clip | 42 / 5.0 |

## 输出

`alpha/asr/kws/online/train.log/<时间戳>/`：
- `log-train-<时间戳>` 全程日志（含初始/最终精度与子集分解）
- `init-model.pt` 适应前模型；`best-adakws.pt` 无监督评分最佳模型

## 注意

- 目标域真标签不参与适应（仅初始/最终精度统计使用）。
- 启动日志必须出现 `成功加载 99 个预训练参数`；若提示随机初始化请停止。
- 若评测标准 1428 条域，把 `cuts_file` 改为 `command_cuts_test_29.jsonl`
  （口径不可与 1071 条结果混用）。
