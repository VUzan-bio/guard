"""Narsil-ML: Dual-branch CRISPR-Cas12a guide scoring with physics-informed attention.

Architecture:
    CNN (target DNA) + RNA-FM (crRNA) -> RLPA -> efficiency + discrimination

Usage:
    from narsil_ml import NarsilML

    model = NarsilML(use_rnafm=True, use_rloop_attention=True, multitask=True)
    output = model(
        target_onehot=target_dna,      # (batch, 4, 34)
        crrna_rnafm_emb=crrna_emb,     # (batch, 20, 640)
        wt_target_onehot=wt_dna,       # (batch, 4, 34)
    )
    efficiency = output["efficiency"]        # (batch, 1)
    discrimination = output["discrimination"] # (batch, 1)
"""

from .narsil_ml import NarsilML

__version__ = "0.1.0"
__all__ = ["NarsilML"]
