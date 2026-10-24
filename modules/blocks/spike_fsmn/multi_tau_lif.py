"""Multi-time-constant SpikingJelly LIF memory interpreted as an FSMN block."""

from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import Tensor, nn
from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.neuron import ParametricLIFNode


ResetMode = Literal["soft", "hard", "pure"]
BranchFusionMode = Literal["mean", "channel_adaptive"]


class MultiTauLIFMemory(nn.Module):
    r"""Multi-branch SpikingJelly LIF memory used as an FSMN-style filter bank.

    FSMN uses explicit FIR delay taps to encode history:

    .. math::
        \tilde{p}_t = p_t + \sum_i a_i \odot p_{t-i}

    A LIF neuron contains the leaky integrator

    .. math::
        V[t] = \alpha V[t-1] + (1-\alpha) I[t],
        \quad \alpha = 1 - 1/\tau

    If we directly read membrane voltage, the leak coefficient ``alpha`` plays
    the role of the FSMN tap decay. With soft reset, the unrolled dynamics are

    .. math::
        V[t] = \sum_{i=0}^{t}\alpha^i(1-\alpha)I[t-i]
        - \sum_{i=0}^{t}\alpha^iS[t-i]V_{th}

    so the LIF block is an IIR history integrator plus adaptive negative
    feedback from previous spikes. Compared with FSMN delay-line buffers, this
    "zero coefficient, zero buffer" memory keeps temporal history inside the
    neuron state, which is lighter and closer to neuromorphic execution.

    Each time constant runs on its own :class:`ParametricLIFNode`, so every
    branch gets an independent (and, on the CuPy backend, correct) tau. A single
    scalar ``w`` per node matters: the CuPy kernel reads only ``decay[0]`` and
    applies it to every neuron, so a shared node with a vector ``w`` would
    silently collapse all branches onto ``w[0]``. A learnable per-branch,
    per-channel :attr:`branch_weights` tensor fuses the branch outputs through a
    softmax over the branch axis.

    Args:
        dim: Hidden dimension ``D``.
        taus: Time constants for parallel SpikingJelly LIF branches.
        v_th: Fixed shared firing threshold (a hyperparameter, not learned).
        surrogate: Surrogate gradient name. Currently supports ``"atan"``.
        learnable_alpha: If ``True``, the per-branch tau logits are trainable;
            otherwise they are frozen at their initial values.
        mode: Reset/output mode: ``"soft"``, ``"hard"``, or ``"pure"``.
        fuse: Whether to fuse branch outputs inside this block. If ``False``,
            forward returns branch features shaped ``(batch, time, n_branches, dim)``.
        branch_fusion_mode: ``"mean"`` for parameter-free averaging or
            ``"channel_adaptive"`` for per-branch, per-channel softmax fusion.
        detach_reset: Whether to detach reset from the surrogate-gradient graph.
        decay_input: Whether input is scaled by reciprocal tau in LIF charging.
        surrogate_alpha: Smoothness factor for ``surrogate.ATan``.
        backend: SpikingJelly backend, ``"torch"`` or ``"cupy"``.

    Inputs:
        x: Tensor shaped ``(batch, time, dim)``.
        state: Optional membrane state shaped ``(n_branches, batch, dim)``.

    Returns:
        A tuple ``(output, next_state)``. If ``fuse=True``, ``output`` is shaped
        ``(batch, time, dim)``. If ``fuse=False``, ``output`` is shaped
        ``(batch, time, n_branches, dim)``. ``next_state`` is shaped
        ``(n_branches, batch, dim)``.
    """

    _VALID_MODES = ("soft", "hard", "pure")

    def __init__(
        self,
        dim: int,
        taus: Iterable[float] | None = None,
        v_th: float = 1.0,
        surrogate: str = "atan",
        learnable_alpha: bool = True,
        mode: ResetMode = "soft",
        fuse: bool = True,
        branch_fusion_mode: BranchFusionMode = "channel_adaptive",
        detach_reset: bool = False,
        decay_input: bool = True,
        surrogate_alpha: float = 2.0,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.taus = list(taus if taus is not None else [5.0, 20.0, 100.0])
        if not self.taus:
            raise ValueError("taus must contain at least one time constant")
        if any(tau <= 1.0 for tau in self.taus):
            raise ValueError("all taus must be > 1.0 so alpha is in (0, 1)")
        if surrogate != "atan":
            raise ValueError("only the ATan surrogate is supported")

        self.surrogate = surrogate
        self.mode = self._normalize_mode(mode)
        self.learnable_alpha = learnable_alpha
        self.fuse = fuse
        self.branch_fusion_mode = self._normalize_branch_fusion_mode(branch_fusion_mode)
        self.detach_reset = detach_reset
        self.decay_input = decay_input
        self.surrogate_alpha = surrogate_alpha
        self.backend = backend
        # Fixed firing threshold, matching a standard PLIF neuron. The SpikingJelly
        # CuPy kernel takes ``v_threshold`` as a plain float constant (no gradient),
        # so a learnable threshold would never train there; keep it a hyperparameter.
        self.v_threshold = 1e9 if self.mode == "pure" else float(v_th)
        self.nodes = self._build_nodes()
        self.branch_weights = (
            self._init_branch_weights()
            if fuse and self.branch_fusion_mode == "channel_adaptive"
            else None
        )
        self.track_branch_activity = False
        self.reset_branch_activity_stats()

    @property
    def n_branches(self) -> int:
        """Return the number of parallel LIF branches."""
        return len(self.taus)

    @property
    def node(self) -> ParametricLIFNode:
        """Return a representative branch node.

        All branch nodes share the same reset/surrogate/decay configuration and
        differ only in their scalar ``w`` (tau). This alias keeps callers that
        inspect shared node options working without exposing the branch list.
        """
        return self.nodes[0]

    @property
    def normalized_branch_weights(self) -> Tensor | None:
        """Return softmax-normalized branch weights for visualization."""
        if not self.fuse:
            return None
        if self.branch_weights is None:
            return self.nodes[0].w.new_full(
                (1, 1, self.n_branches, self.dim),
                1.0 / self.n_branches,
            )
        return torch.softmax(self.branch_weights, dim=2)

    @property
    def v_th(self) -> float:
        """Return the fixed shared firing threshold (a hyperparameter)."""
        return self.v_threshold

    def get_branch_weight_snapshot(self) -> dict[str, Tensor] | None:
        """Return detached branch-fusion data shaped for offline analysis."""
        if not self.fuse:
            return None

        with torch.no_grad():
            normalized_weights = self.normalized_branch_weights
            if normalized_weights is None:
                return None
            if self.branch_weights is None:
                logits = torch.zeros_like(normalized_weights)
            else:
                logits = self.branch_weights
            effective_alphas = torch.stack([
                (1.0 - node.w.sigmoid()).reshape(()) for node in self.nodes
            ])
            snapshot = {
                "logits": logits[0, 0].detach().clone(),
                "normalized_weights": normalized_weights[0, 0].detach().clone(),
                "taus": self.nodes[0].w.new_tensor(self.taus).detach().clone(),
                "effective_alphas": effective_alphas.detach().clone(),
                "effective_taus": torch.reciprocal(1.0 - effective_alphas).detach().clone(),
                "is_channel_adaptive": self.nodes[0].w.new_tensor(
                    int(self.branch_fusion_mode == "channel_adaptive")
                ).detach(),
            }
            activity = self.get_branch_activity_snapshot()
            if activity is not None:
                snapshot.update(activity)
            return snapshot

    def alpha_parameters(self) -> list[nn.Parameter]:
        """Return the per-branch parameters that control reciprocal tau/alpha."""
        return [node.w for node in self.nodes]

    def effective_alphas(self) -> list[Tensor]:
        """Return current branch leak coefficients ``alpha = 1 - 1 / tau``."""
        return [(1.0 - node.w.sigmoid()).reshape(()) for node in self.nodes]

    def effective_taus(self) -> list[Tensor]:
        """Return current branch time constants implied by PLIF parameters."""
        return [torch.reciprocal(node.w.sigmoid()).reshape(()) for node in self.nodes]

    def fusion_parameters(self) -> list[nn.Parameter]:
        """Return channel-adaptive fusion logits, if enabled."""
        return [] if self.branch_weights is None else [self.branch_weights]

    def branch_balance_loss(self) -> Tensor | None:
        """Return block-level KL divergence from uniform branch usage.

        The channel average is balanced while individual channels remain free to
        specialize in a particular temporal scale.
        """
        if self.branch_weights is None:
            return None
        usage = self.normalized_branch_weights[0, 0].mean(dim=1)
        loss = (usage * torch.log(
            usage.clamp_min(torch.finfo(usage.dtype).tiny) * self.n_branches
        )).sum()
        return loss.clamp_min(0.0)

    def set_branch_activity_tracking(self, enabled: bool) -> None:
        """Enable analysis-only accumulation of branch output statistics."""
        self.track_branch_activity = bool(enabled)
        self.reset_branch_activity_stats()

    def reset_branch_activity_stats(self) -> None:
        """Clear accumulated output statistics without touching model state."""
        self._branch_activity_sum: Tensor | None = None
        self._branch_activity_gram: Tensor | None = None
        self._branch_activity_count = 0

    def get_branch_activity_snapshot(self) -> dict[str, Tensor] | None:
        """Return accumulated branch activity rates and pairwise cosine similarity."""
        if self._branch_activity_sum is None or self._branch_activity_gram is None:
            return None
        if self._branch_activity_count <= 0:
            return None
        diagonal = self._branch_activity_gram.diagonal().clamp_min(0.0)
        denom = torch.sqrt(diagonal[:, None] * diagonal[None, :]).clamp_min(
            torch.finfo(self._branch_activity_gram.dtype).eps
        )
        cosine = self._branch_activity_gram / denom
        return {
            "branch_activity_rates": (
                self._branch_activity_sum / self._branch_activity_count
            ).detach().clone(),
            "branch_output_cosine": cosine.detach().clone(),
        }

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Run multi-tau LIF memory over a batch-major sequence.

        Args:
            x: Input tensor shaped ``(batch, time, dim)``.
            state: Optional membrane state shaped ``(n_branches, batch, dim)``.

        Returns:
            Output sequence and next membrane state.
        """
        if x.dim() != 3:
            raise ValueError(f"expected x to be 3-D (B, T, D), got {tuple(x.shape)}")
        batch, _, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected hidden dim {self.dim}, got {dim}")
        if state is None:
            state = x.new_zeros(self.n_branches, batch, dim)
        self._validate_state(state, batch, dim)

        x_seq = x.transpose(0, 1).contiguous()  # (T, B, D) multi-step layout

        branch_outputs = []
        next_states = []
        for index, node in enumerate(self.nodes):
            self._set_node_initial_state(node, state[index], x)
            spike_seq = node(x_seq)
            branch_outputs.append(self._select_output_sequence(node, spike_seq))
            next_states.append(self._node_next_state(node, x))

        branch_seq = torch.stack(branch_outputs, dim=2)  # (B, T, n_branches, D)
        self._update_branch_activity_stats(branch_seq)
        next_state = torch.stack(next_states, dim=0)      # (n_branches, B, D)
        return self._format_output(branch_seq), next_state

    def _build_nodes(self) -> nn.ModuleList:
        """Build one scalar-tau parametric LIF node per branch."""
        nodes = nn.ModuleList()
        for tau in self.taus:
            node = ParametricLIFNode(
                init_tau=float(tau),
                decay_input=self.decay_input,
                v_threshold=self.v_threshold,
                v_reset=None if self.mode == "soft" else 0.0,
                surrogate_function=self._build_surrogate_function(),
                detach_reset=self.detach_reset,
                step_mode="m",
                store_v_seq=self.mode == "pure",
                backend=self.backend,
            )
            node.w.requires_grad_(self.learnable_alpha)
            nodes.append(node)
        return nodes

    def _build_surrogate_function(self) -> nn.Module:
        """Construct the configured surrogate function."""
        if self.surrogate == "atan":
            return surrogate.ATan(alpha=self.surrogate_alpha)
        raise ValueError("only the ATan surrogate is supported")

    def _set_node_initial_state(self, node: ParametricLIFNode, state: Tensor, x: Tensor) -> None:
        """Reset one branch node and inject caller-provided membrane state."""
        functional.reset_net(node)
        node.v = state.to(device=x.device, dtype=x.dtype)
        node.v_reset = None if self.mode == "soft" else 0.0

    def _select_output_sequence(self, node: ParametricLIFNode, spike_seq: Tensor) -> Tensor:
        """Return spikes for LIF modes or membrane voltages for pure mode."""
        if self.mode != "pure":
            return spike_seq.transpose(0, 1)
        if getattr(node, "v_seq", None) is None:
            raise RuntimeError("pure mode requires the node to store v_seq")
        return node.v_seq.transpose(0, 1)

    def _node_next_state(self, node: ParametricLIFNode, x: Tensor) -> Tensor:
        """Return one branch's next membrane state shaped ``(batch, dim)``."""
        state = getattr(node, "v", None)
        if not isinstance(state, Tensor):
            return x.new_zeros(x.shape[0], self.dim)
        return state.detach()

    def _format_output(self, branch_seq: Tensor) -> Tensor:
        """Return fused features or raw branch features according to ``self.fuse``."""
        if not self.fuse:
            return branch_seq
        return self._fuse_branches(branch_seq)

    def _fuse_branches(self, branch_seq: Tensor) -> Tensor:
        if self.branch_fusion_mode == "mean":
            return branch_seq.mean(dim=2)
        return (branch_seq * self.normalized_branch_weights).sum(dim=2)


    def _update_branch_activity_stats(self, branch_seq: Tensor) -> None:
        if not self.track_branch_activity:
            return
        flat = branch_seq.detach().float().permute(2, 0, 1, 3).reshape(self.n_branches, -1)
        activity_sum = flat.sum(dim=1)
        activity_gram = flat @ flat.transpose(0, 1)
        if self._branch_activity_sum is None:
            self._branch_activity_sum = activity_sum
            self._branch_activity_gram = activity_gram
        else:
            self._branch_activity_sum = self._branch_activity_sum + activity_sum
            self._branch_activity_gram = self._branch_activity_gram + activity_gram
        self._branch_activity_count += flat.shape[1]

    def _init_branch_weights(self) -> nn.Parameter:
        """Initialize branch fusion logits for a simple mean over branches."""
        logits = torch.zeros((1, 1, self.n_branches, self.dim), dtype=torch.float32)
        return nn.Parameter(logits)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @classmethod
    def _normalize_mode(cls, mode: ResetMode) -> str:
        if mode not in cls._VALID_MODES:
            valid = ", ".join(cls._VALID_MODES)
            raise ValueError(f"unknown mode {mode!r}; expected one of: {valid}")
        return mode

    @staticmethod
    def _normalize_branch_fusion_mode(mode: BranchFusionMode) -> str:
        valid = ("mean", "channel_adaptive")
        if mode not in valid:
            raise ValueError(
                f"unknown branch_fusion_mode {mode!r}; expected one of: {', '.join(valid)}"
            )
        return mode

    def _validate_state(self, state: Tensor, batch: int, dim: int) -> None:
        expected = (self.n_branches, batch, dim)
        if tuple(state.shape) != expected:
            raise ValueError(f"expected state shape {expected}, got {tuple(state.shape)}")
