"""车载语音指令 KWS（29 类中文指令）。

子模块：
  system.py        Lightning 训练系统（AnnKwsTeacher / MultiTauKwsKd，
                   由 alpha/asr/system.py 末尾注册给框架 get_model）
  dataset.py       Lhotse cuts 数据集与 DataLoader（预计算 FBank）
  schedulers.py    LinearWarmupCosineLR（ANN 教师调度）
  utils/           配置加载、标签/精度评测、checkpoint 加载（直读 Lightning ckpt）
  online/          在线学习（AdaKWS source-free 测试时适应）训练器
  tools/           效率测量独立工具

模型定义在 alpha/asr/model/（TdnnLstm、MultiTauLIFCommanderKWS，
随框架模型注册表路由）；配置与数据在 examples/carKWS_offline/ 与
examples/carKWS_online/。
"""
