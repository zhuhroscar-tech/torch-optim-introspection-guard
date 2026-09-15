"""torch-optim-introspection-guard: detect and safely neutralize
PyTorch's get_optimizer_state_dict() step-counter mutation bug
(pytorch/pytorch#164929)."""
from .core import (
    OptimizerDiagnosis,
    OptimizerSpec,
    PROBE_OPTIMIZERS,
    TorchUnavailableError,
    diagnose,
    safe_get_optimizer_state_dict,
)

__version__ = "0.1.0"
__all__ = [
    "OptimizerDiagnosis",
    "OptimizerSpec",
    "PROBE_OPTIMIZERS",
    "TorchUnavailableError",
    "diagnose",
    "safe_get_optimizer_state_dict",
    "__version__",
]
