"""YAML 配置加载与路径解析。"""

import logging
from pathlib import Path

from icefall.utils import AttributeDict
from omegaconf import OmegaConf

# 项目根目录 = AlphaSNN（框架根；utils → kws → asr → alpha → AlphaSNN）
PROJECT_ROOT = Path(__file__).resolve().parents[4]
# 代码目录 = kws/online/（exp_dir 以此为基准解析）
ONLINE_DIR = Path(__file__).resolve().parents[1] / "online"


def load_config(config_path) -> AttributeDict:
    """加载 YAML 配置，返回 AttributeDict。

    - 相对路径（exp_dir / 数据目录 / 模型路径）统一解析为基于 PROJECT_ROOT 的绝对路径
    - exp_dir 每次运行会附加时间戳子目录
    """
    conf = OmegaConf.load(config_path)
    params = _to_attr_dict(OmegaConf.to_container(conf, resolve=True))
    resolve_paths(params)
    return params


def _to_attr_dict(obj):
    """递归将嵌套 dict 转换为 AttributeDict。"""
    if isinstance(obj, dict):
        return AttributeDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(v) for v in obj]
    return obj


def resolve_paths(params: AttributeDict) -> None:
    """把配置中的相对路径解析为绝对路径。

    - exp_dir：以 online/ 目录为基准
    - 其余路径（pretrained_model / 数据目录）：以项目根 AlphaSNN 为基准
    """
    sys = params.system

    sys.exp_dir = _abs(sys.exp_dir, ONLINE_DIR)
    params.paths.pretrained_model = _abs(params.paths.pretrained_model)

    for key in ("fbank_dir", "manifest_dir"):
        for part in ("source", "target"):
            if part not in params.data:   # 纯 AdaKWS 配置已无 source 块
                continue
            node = params.data[part]
            if key in node:
                node[key] = _abs(node[key])

    # 每次运行生成带时间戳的子目录
    import datetime

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.exp_dir = sys.exp_dir / run_ts


def _abs(p, base=PROJECT_ROOT):
    """Path 转绝对路径：已绝对则原样，相对则基于 base 解析。"""
    p = Path(p)
    return p if p.is_absolute() else base / p


def log_config(params: AttributeDict) -> None:
    """打印配置。icefall 的 AttributeDict.__str__ 内部用 json.dumps，
    遇到 Path 会报错，所以这里手动转字符串再打。"""
    import json

    logging.info(json.dumps(params, default=str, indent=2, ensure_ascii=False))
