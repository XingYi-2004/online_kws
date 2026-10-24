from __future__ import annotations

from torch import Tensor, nn
from spikingjelly.activation_based.neuron import BaseNode


class SpikeRateMonitor:
    """Track mean spike firing rate across all spiking nodes via forward hooks."""

    def __init__(self, model: nn.Module) -> None:
        self._hooks: list = []
        self._records: list[tuple[float, int]] = []
        for module in model.modules():
            if isinstance(module, BaseNode):
                self._hooks.append(module.register_forward_hook(self._hook))

    def _hook(self, module: nn.Module, input: tuple, output: Tensor) -> None:
        if isinstance(output, Tensor):
            self._records.append((float(output.detach().sum()), output.numel())) # TODO

    def reset(self) -> None:
        self._records.clear()

    def get_mean_rate(self) -> float | None:
        if not self._records:
            return None
        total_numel = sum(r[1] for r in self._records)
        return sum(r[0] for r in self._records) / total_numel if total_numel > 0 else 0.0

    @property
    def is_active(self) -> bool:
        return bool(self._hooks)
