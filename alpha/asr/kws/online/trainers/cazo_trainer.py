"""CAZO：曲率感知零阶（ZO）测试时适应，直接作用于 MT-SNN 的全部 BN affine。

移植 CAZO（CVPR 2026, github.com/Hollyming/CAZO）的优化机制——对角曲率感知
的零阶梯度估计与扰动采样——但适应对象改为 Tent/AdaKWS 一脉的标准口径：
归一化层 affine（cazo_param_scope=bn_affine 4608 参数 / bn_gn_affine 10368
参数，后者对齐 AdaKWS 90.10% 冠军配置，与反传基线严格同参同起点）。

历史（cazo01/02，已废弃）：曾按原版在 blocks.3 注入 LoRA 支路仅更新支路。
失败根因：①支路零初始化，原版依赖源域预训练支路（FOA 范式），本项
目无此前置；②稀疏梯度使曲率矩阵 D 塌缩 → 扰动范数正反馈爆炸（58→377）；
③CAZO 的省显存卖点对 0.66M 参数的 SNN 无意义。

当前设计：
  - 参数：全模型 BN affine，从预训练值出发（非零起点，每批只是小修正）；
  - 优化：每批 k 个扰动 ±ε 各一次纯前向 → g ≈ (1/k)Σ ΔL/(2ε)·z；
    曲率 D ← (1−ν)D + ν·g²，扰动按 z/√H̃（H̃=D/(1−(1−ν)^t)）采样，
    H̃ 加下限 D_FLOOR 防塌缩（扰动范数封顶 ≈ √(1/D_FLOOR) 倍基准）；
  - BN 固定 eval 模式（checkpoint running stats 归一化，固定基准）；
  - 损失：批均值预测熵（尺度与批大小解耦），纯前向、零反传；
  - 无 wrapper → best-cazo.pt 为原生 state_dict 键，跨脚本可直接加载。

入口：examples/carKWS_online/local/train_cazo.py。
"""

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from alpha.asr.kws.utils.train_utils import batch_lengths as _batch_lengths

# H̃ 下限（初始值 1 的 10%）：零梯度维度的扰动缩放封顶 √10 ≈ 3.2 倍，
# 防止 cazo02 型 D 塌缩正反馈
_D_FLOOR = 0.1


# ---------------------------------------------------------------------------
# ZO 基础件（对应原版 Optimizer/SGD/SGD_Momentum 与参数扁平化）
# ---------------------------------------------------------------------------
class _CazoSGD:
    """原版 CAZO 自实现 SGD：w ← w − lr·g。"""

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, grad_estimate: torch.Tensor) -> torch.Tensor:
        return -self.lr * grad_estimate


class _CazoSGDMomentum:
    """原版 CAZO 自实现动量 SGD：m ← β·m + (1−β)·g；w ← w − lr·m。"""

    def __init__(self, lr: float, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        self.momentum: Optional[torch.Tensor] = None

    def step(self, grad_estimate: torch.Tensor) -> torch.Tensor:
        if self.momentum is None:
            self.momentum = (1 - self.beta) * grad_estimate
        else:
            self.momentum = self.beta * self.momentum + (1 - self.beta) * grad_estimate
        return -self.lr * self.momentum


def _collect_norm_affine(core: nn.Module, scope: str) -> tuple[list[nn.Parameter], list[str]]:
    """收集归一化层 affine：bn_affine=全部 BatchNorm（AdaKWS 同口径）；
    bn_gn_affine=BatchNorm+GroupNorm（AdaKWS 90.10% 冠军配置口径，10368 参数）。"""
    if scope == "bn_affine":
        norm_types = (nn.modules.batchnorm._BatchNorm,)
    elif scope == "bn_gn_affine":
        norm_types = (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)
    else:
        raise ValueError(f"不支持的 cazo_param_scope={scope!r}，可选 'bn_affine' 或 'bn_gn_affine'")
    parameters: list[nn.Parameter] = []
    names: list[str] = []
    for name, module in core.named_modules():
        if isinstance(module, norm_types):
            if module.weight is not None:
                parameters.append(module.weight)
                names.append(f"{name}.weight")
            if module.bias is not None:
                parameters.append(module.bias)
                names.append(f"{name}.bias")
    return parameters, names


def _load_flat(parameters: list[nn.Parameter], flat: torch.Tensor) -> None:
    """把 flat 向量按参数顺序写回（copy_ 语义，不换存储）。"""
    pointer = 0
    with torch.no_grad():
        for param in parameters:
            num = param.numel()
            param.data.copy_(flat[pointer : pointer + num].view_as(param))
            pointer += num


def _entropy_mean(model: nn.Module, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """批次预测熵均值（ZO 损失尺度与批大小解耦；纯前向，无梯度）。"""
    functional.reset_net(model)
    logits = model(features, lengths)
    probabilities = F.softmax(logits, dim=1)
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# 主训练器（接口对齐 adakws_trainer.train_adakws）
# ---------------------------------------------------------------------------
def train_cazo(
    params,
    model: nn.Module,
    target_dl,
    test_dl,
    device,
    tb_writer: Optional[SummaryWriter] = None,
):
    """Run CAZO curvature-aware ZO adaptation on all BN affine parameters.

    只更新全部 BN affine（与 AdaKWS bn_affine 同参同起点），按无监督熵评分
    保留最佳迭代并保存 best-cazo.pt。
    """
    cfg = params.online
    k = int(getattr(cfg, "cazo_k", 20))
    epsilon = float(getattr(cfg, "cazo_epsilon", 0.1))
    nu = float(getattr(cfg, "cazo_nu", 0.8))
    optimizer_name = str(getattr(cfg, "cazo_optimizer", "sgd")).lower()
    lr = float(getattr(cfg, "cazo_lr", 0.01))
    beta = float(getattr(cfg, "cazo_beta", 0.9))
    num_iterations = int(getattr(cfg, "num_iterations", 1))
    param_scope = str(getattr(cfg, "cazo_param_scope", "bn_affine")).lower()

    # 主干全冻结 + 全程 eval 语义（BN 用 checkpoint running stats 归一化）
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    core = model.module if hasattr(model, "module") else model
    bn_parameters, bn_names = _collect_norm_affine(core, param_scope)
    total_params = sum(p.numel() for p in bn_parameters)
    if not bn_parameters:
        raise RuntimeError(f"cazo_param_scope={param_scope} 未匹配到任何归一化层 affine 参数")

    if optimizer_name == "sgd":
        optimizer = _CazoSGD(lr)
    elif optimizer_name == "sgd_momentum":
        optimizer = _CazoSGDMomentum(lr, beta)
    else:
        raise ValueError(f"不支持的 cazo_optimizer={optimizer_name!r}，可选 'sgd' 或 'sgd_momentum'")

    logging.info(
        "开始 CAZO 在线适应：更新范围=%s（%d 层，%d 参数）；"
        "扰动数k=%d，ε=%g，ν=%g，H̃下限=%g；优化器=%s lr=%g；"
        "主干冻结、BN固定eval（checkpoint running stats）；目标域标签不参与适应",
        param_scope, len(bn_names), total_params, k, epsilon, nu, _D_FLOOR, optimizer_name, lr,
    )

    # ZO 状态：对角 Hessian 估计 D（初始全 1）、上一步梯度、时间步
    curvature = torch.ones(total_params, device=device)
    prev_grad_estimate: Optional[torch.Tensor] = None
    time_step = 0

    best_unsupervised_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_iteration = -1

    for iteration in range(num_iterations):
        total_score = 0.0
        total_batches = 0

        for batch_idx, batch in enumerate(target_dl):
            features = batch["inputs"].to(device).permute(0, 2, 1)
            lengths = _batch_lengths(batch, device)

            base = parameters_to_vector(bn_parameters).detach().clone()

            # 曲率感知扰动采样：z ~ N(0, H̃⁻¹)，H̃ = D/(1−(1−ν)^t)（偏差校正+下限防塌缩）
            if time_step == 0:
                h_diag = curvature
            else:
                h_diag = curvature / (1 - (1 - nu) ** time_step)
            h_diag = h_diag.clamp_min(_D_FLOOR)
            perturbations = torch.randn(k, total_params, device=device) / torch.sqrt(h_diag)

            # RGE 零阶梯度估计：每扰动 ±ε 各一次纯前向
            grad_estimate = torch.zeros(total_params, device=device)
            with torch.no_grad():
                for perturbation in perturbations:
                    _load_flat(bn_parameters, base + epsilon * perturbation)
                    loss_plus = _entropy_mean(model, features, lengths)
                    _load_flat(bn_parameters, base - epsilon * perturbation)
                    loss_minus = _entropy_mean(model, features, lengths)
                    grad_estimate += (loss_plus - loss_minus) / (2 * epsilon) * perturbation
            grad_estimate /= k

            # 曲率估计更新（从第二步起，用当前梯度估计的二阶矩）
            if prev_grad_estimate is not None:
                curvature = (1 - nu) * curvature + nu * grad_estimate**2
            prev_grad_estimate = grad_estimate.detach().clone()
            time_step += 1

            # 应用更新并计算本批最终熵（无监督评分，逐样本均值）
            update = optimizer.step(grad_estimate)
            with torch.no_grad():
                _load_flat(bn_parameters, base + update)
                score = _entropy_mean(model, features, lengths).item()

            total_score += score
            total_batches += 1
            params.batch_idx_train += 1

            if tb_writer:
                tb_writer.add_scalar("cazo/entropy", score, params.batch_idx_train)
                tb_writer.add_scalar("cazo/grad_norm", grad_estimate.norm().item(), params.batch_idx_train)
            logging.info(
                "CAZO 迭代 %d 批 %d：平均熵 %.6f，梯度范数 %.4g，扰动范数均值 %.4g",
                iteration,
                batch_idx,
                score,
                grad_estimate.norm().item(),
                perturbations.norm(dim=1).mean().item(),
            )

        unsupervised_score = total_score / max(total_batches, 1)
        logging.info(
            "CAZO 迭代 %d/%d：平均熵 %.6f，无监督评分 %.6f",
            iteration,
            num_iterations - 1,
            total_score / max(total_batches, 1),
            unsupervised_score,
        )

        # 目标域标签从不参与该评分，按无监督评分保留最佳迭代
        if total_batches > 0 and unsupervised_score < best_unsupervised_score:
            best_unsupervised_score = unsupervised_score
            best_state = copy.deepcopy(model.state_dict())
            best_iteration = iteration

    model.load_state_dict(best_state)
    logging.info("CAZO 适应完成：按无监督评分保留迭代 %d 的模型", best_iteration)

    best_path = Path(params.system.exp_dir) / "checkpoints" / "best-cazo.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, best_path)
    logging.info("CAZO 最佳模型已保存: %s（原生 state_dict 键，可直接跨脚本加载）", best_path)
