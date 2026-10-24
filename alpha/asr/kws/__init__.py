"""车载语音指令 KWS（29 类中文指令）。

子模块：
  offline/  离线训练（MultiTauLIFCommander + KD 蒸馏，干净测试 98.04%）与评测
  online/   在线学习（AdaKWS source-free 测试时适应，87.86% → 90.20%）
  conf/     两份入口配置（offline_kd.yaml / online_adakws.yaml）
  utils/    数据加载（lhotse CutSet + 预计算 FBank）、评测、checkpoint 键映射
  tools/    效率测量、回放库构建等独立工具

2026-10-24 自 online_kws/SNN 与 online_kws/SNN_aol 迁入 AlphaSNN 框架，
训练/评测算法逻辑未做任何改动。
"""
