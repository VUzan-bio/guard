"""Module 7: Multiplex panel optimization and spatial array design."""

from guard.multiplex.optimizer import MultiplexOptimizer
from guard.multiplex.pooling import compute_primer_pools, compute_amplicon_pad_specificity
from guard.multiplex.kinetics import estimate_all_targets, estimate_time_to_result

__all__ = [
    "MultiplexOptimizer",
    "compute_primer_pools",
    "compute_amplicon_pad_specificity",
    "estimate_all_targets",
    "estimate_time_to_result",
]
