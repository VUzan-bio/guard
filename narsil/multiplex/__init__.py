"""Module 7: Multiplex panel optimization and spatial array design."""

from narsil.multiplex.optimizer import MultiplexOptimizer
from narsil.multiplex.pooling import compute_primer_pools, compute_amplicon_pad_specificity
from narsil.multiplex.kinetics import estimate_all_targets, estimate_time_to_result

__all__ = [
    "MultiplexOptimizer",
    "compute_primer_pools",
    "compute_amplicon_pad_specificity",
    "estimate_all_targets",
    "estimate_time_to_result",
]
