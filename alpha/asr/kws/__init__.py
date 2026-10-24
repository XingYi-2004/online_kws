"""车载语音指令 KWS（29 类中文指令）。

子模块：
  system.py        Lightning 训练系统（AnnKwsTeacher / MultiTauKwsKd，
                   由 alpha/asr/system.py 末尾注册给框架 get_model）
  dataset.py       Lhotse cuts 数据集与 DataLoader（预计算 FBank）
  tdnn_lstm.py     TdnnLstm ANN（教师/基线，唯一权威定义）
  schedulers.py    LinearWarmupCosineLR（ANN 教师调度）
  model_kws.py     推理/在线适应封装（subsample + MultiTauLIFCommander）
  utils/           配置加载、标签/精度评测、checkpoint 键映射
  online/          在线学习（AdaKWS source-free 测试时适应）训练器
  tools/           效率测量独立工具

配置与数据在 examples/carKWS_offline/ 与 examples/carKWS_online/。
"""
