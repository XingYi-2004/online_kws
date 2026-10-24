import numpy as np


class _NumpyCompatProxy:
    """Expose removed NumPy aliases only to legacy third-party modules."""

    int = int

    def __getattr__(self, name):
        return getattr(np, name)


def ensure_spikingjelly_cupy_numpy_compat():
    """Make SpikingJelly 0.0.0.14 CuPy kernels compatible with NumPy >= 1.24."""
    from spikingjelly.activation_based.auto_cuda import base as auto_cuda_base

    if not hasattr(auto_cuda_base.np, 'int'):
        auto_cuda_base.np = _NumpyCompatProxy()
