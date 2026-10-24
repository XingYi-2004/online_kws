"""在线学习入口：分流到 AdaKWS 测试时适应。"""

import logging


def train_online(params, model, target_dl, test_dl, device, tb_writer=None):
    """在线学习：method=adakws → AdaKWS 测试时适应。

    历史：伪标签 OCL / SPACE / DANN / CDAN 分支已随实验清理
    （全部记录见 AlphaSNN/README.md 的 DA-MT 系列与 ONLINE_LEARNING_HANDOFF.md）。
    """
    method = str(getattr(params.online, "method", "adakws")).lower()
    if method != "adakws":
        raise ValueError(f"仅支持 online.method=adakws（历史方法已清理），当前: {method!r}")
    logging.info("在线方法: AdaKWS（source-free 测试时适应，目标域标签不参与适应）")

    from alpha.asr.kws.online.trainers.adakws_trainer import train_adakws

    return train_adakws(params, model, target_dl, test_dl, device, tb_writer)
