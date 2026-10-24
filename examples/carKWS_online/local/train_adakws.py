"""在线学习主入口（纯 AdaKWS 测试时适应）。

启动：
  cd AlphaSNN && conda activate snn
  python examples/carKWS_online/local/train_adakws.py

所有训练配置在 examples/carKWS_online/conf/online_adakws.yaml 中修改
（GPU、模型、数据路径、超参数），不需要任何命令行参数。
训练器（AdaKWS 算法）代码位于 alpha/asr/kws/online/trainers/。

历史说明：本入口曾支持 DANN/CDAN 对抗域适应、SPACE、伪标签 OCL 及教师/
记忆库/回放等叠加组件（DA-MT-01~64 系列实验），全部验证无收益，代码已清理；
实验记录见 AlphaSNN/README.md，方法结论见 ONLINE_LEARNING_HANDOFF.md。
"""

import logging
import os
import sys
from pathlib import Path

# 框架根（local → carKWS_online → examples → AlphaSNN）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 唯一配置文件（想换配置直接改它，或修改此路径）
CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "online_adakws.yaml"


def run(rank, world_size, params):
    import torch  # 在 main() 中已设置 CUDA_VISIBLE_DEVICES 后导入

    logging.info("=" * 60)
    logging.info("开始在线学习（AdaKWS）")
    logging.info("=" * 60)

    system = params.system

    from lhotse.utils import fix_random_seed

    fix_random_seed(system.seed)
    logging.info(
        f"随机种子: {system.seed}, GPU数量: {world_size}, "
        f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}"
    )

    if world_size > 1:
        from icefall.dist import setup_dist

        setup_dist(rank, world_size, system.master_port)

    system.exp_dir.mkdir(parents=True, exist_ok=True)
    from icefall.utils import setup_logger

    log_file = system.exp_dir / system.log_filename
    setup_logger(f"{log_file}")
    logging.info(f"实验目录: {system.exp_dir}")
    from alpha.asr.kws.utils.config import log_config

    log_config(params)

    tb_writer = None
    if system.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = SummaryWriter(log_dir=f"{system.exp_dir}/tensorboard")

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    # 运行时状态
    params.batch_idx_train = 0

    online_method = str(getattr(params.online, "method", "adakws")).lower()
    if online_method != "adakws":
        raise ValueError(
            f"仅支持 online.method=adakws（历史 DA/SPACE/OCL 方法已清理），当前: {online_method!r}"
        )

    # ---- 创建模型 ----
    logging.info("创建模型...")
    from alpha.asr.kws.model_kws import MultiTauLIFCommanderKWS

    model = MultiTauLIFCommanderKWS(dict(params.model))
    logging.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---- 加载预训练模型 ----
    logging.info("加载预训练模型...")
    from alpha.asr.kws.utils.train_utils import load_pretrained_model

    load_pretrained_model(params, model)
    model.to(device)
    logging.info(f"模型已移动到: {device}")

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[rank])

    # ---- 准备数据 ----
    logging.info("准备数据...")
    from alpha.asr.kws.dataset import build_adakws_dataloader, load_cuts

    # 目标域（测试集，全部数据；无标签，仅输入参与适应）
    target_cuts = load_cuts(params.data.target.manifest_dir, params.data.target.cuts_file)
    batch_size = int(params.online.adakws_batch_size)
    # 适应加载器与统一评测加载器同口径（固定 batch、按时长排序）。
    # 初始/最终精度必须出自同一个 eval_dl——SNN 预测受补零影响，口径必须统一。
    target_dl = build_adakws_dataloader(target_cuts, params.data.target, batch_size)
    eval_dl = build_adakws_dataloader(target_cuts, params.data.target, batch_size)
    logging.info("数据准备完成:")
    logging.info(f"  目标域: {len(target_cuts)} 条数据（全部数据，标签不参与适应）")
    logging.info(f"  统一评测加载器：固定 batch={batch_size}（按时长排序）")

    # ---- 初始测试 ----
    from alpha.asr.kws.utils.train_utils import compute_subset_accuracy, compute_test_accuracy

    logging.info("进行初始测试...")
    initial_acc = compute_test_accuracy(model, eval_dl, device, world_size)
    logging.info(f"初始测试精度: {initial_acc:.4f}")
    subset_names = {"real_car": "真实车载", "simu_noise": "加噪simu", "rir_reverb": "RIR混响"}
    for key, r in compute_subset_accuracy(model, eval_dl, device).items():
        logging.info(f"  初始[{subset_names.get(key, key)}]: {r['acc']:.4f} (n={r['n']})")

    # 保存在线适应前模型（适应把模型带坏时可回滚到这里）
    init_model_path = system.exp_dir / "init-model.pt"
    torch.save({"model": model.state_dict()}, init_model_path)
    logging.info(f"已保存在线适应前初始模型: {init_model_path}")

    # ---- 在线学习（AdaKWS）----
    logging.info("=" * 40)
    logging.info("开始在线学习")
    logging.info("=" * 40)
    from alpha.asr.kws.online.trainers.online_trainer import train_online

    train_online(params, model, target_dl, eval_dl, device, tb_writer)

    # ---- 最终测试（加载无监督评分选出的最佳模型） ----
    logging.info("=" * 40)
    logging.info("最终测试")
    logging.info("=" * 40)
    best_model_path = system.exp_dir / "best-adakws.pt"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        logging.info("已加载最佳模型进行最终测试")

    final_acc = compute_test_accuracy(model, eval_dl, device, world_size)
    logging.info(f"最终测试精度: {final_acc:.4f}")
    for key, r in compute_subset_accuracy(model, eval_dl, device).items():
        logging.info(f"  最终[{subset_names.get(key, key)}]: {r['acc']:.4f} (n={r['n']})")
    logging.info(f"精度提升: {final_acc - initial_acc:.4f}")
    logging.info("训练完成！")

    if world_size > 1:
        import torch.distributed as dist
        from icefall.dist import cleanup_dist

        dist.barrier()
        cleanup_dist()


def main():
    # GPU 选择：必须在任何 torch 相关 import 之前写入环境变量
    try:
        from omegaconf import OmegaConf  # 不依赖 torch
    except ModuleNotFoundError:
        raise SystemExit(
            "缺少依赖 omegaconf，当前 python 环境不对。\n"
            "请先激活 snn 环境再运行：\n"
            "  conda activate snn\n"
            "  python examples/carKWS_online/local/train_adakws.py"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = OmegaConf.load(CONFIG_PATH).system.cuda_visible_devices

    import numpy as np

    np.int = int  # numpy 2.x 兼容 spikingjelly cupy 后端
    import torch

    from alpha.asr.kws.utils.config import PROJECT_ROOT, load_config

    params = load_config(CONFIG_PATH)
    world_size = params.system.world_size

    # 数据清单里的特征路径是相对路径（相对 AlphaSNN 根），统一切过去，保证从任意目录启动都能运行
    os.chdir(PROJECT_ROOT)

    if world_size > 1:
        import torch.multiprocessing as mp

        mp.spawn(run, args=(world_size, params), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, params=params)


if __name__ == "__main__":
    main()
