"""GUARD-Net: Dual-branch scoring with physics-informed attention
and multi-task efficiency + discrimination prediction.

Architecture:
    CNN branch (target DNA one-hot) + RNA-FM branch (crRNA embeddings)
    -> concatenate per-position features
    -> optional RLPA (R-Loop Propagation Attention)
    -> adaptive average pool -> fixed-size vector
    -> efficiency head (always)
    -> discrimination head (optional, multi-task)

CRITICAL biological distinction:
    - CNN branch sees TARGET DNA (34 nt: 4 PAM + 20 protospacer + 10 flanking)
    - RNA-FM branch sees crRNA SPACER (20 nt, as RNA)
    - These are DIFFERENT molecules. The RNA-FM branch handles the
      20->34 position alignment internally (zero-padding at PAM/flanking).
    - For discrimination: only the target DNA changes (mutant vs wildtype).
      The crRNA is the SAME for both conditions.

Features can be incrementally enabled:
    GUARDNet()                                         -> CNN only
    GUARDNet(use_rnafm=True)                           -> CNN + RNA-FM
    GUARDNet(use_rnafm=True, use_rloop_attention=True) -> + RLPA
    GUARDNet(..., multitask=True)                      -> + discrimination

Trainable parameters: ~150K (RNA-FM frozen, CNN ~65K, proj ~60K, RLPA ~25K).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .branches.cnn_branch import CNNBranch
from .branches.rnafm_branch import RNAFMBranch
from .heads.discrimination_head import DiscriminationHead


class GUARDNet(nn.Module):

    def __init__(
        self,
        # CNN branch config
        cnn_branches: int = 32,
        cnn_out_dim: int = 64,
        # RNA-FM branch config
        use_rnafm: bool = True,
        rnafm_embed_dim: int = 640,
        rnafm_proj_dim: int = 64,
        # Attention
        use_rloop_attention: bool = False,
        # Multi-task
        multitask: bool = False,
        # Optional scalar features (e.g. Evo 2 LLR, GC%, MFE)
        n_scalar_features: int = 0,
        # Head params
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        # -- Branch 1: CNN on target DNA (trainable) --
        self.cnn = CNNBranch(
            in_channels=4, branches=cnn_branches, out_dim=cnn_out_dim,
        )

        # -- Branch 2: RNA-FM projection on crRNA (projection trainable) --
        self.use_rnafm = use_rnafm
        if self.use_rnafm:
            self.rnafm = RNAFMBranch(
                embed_dim=rnafm_embed_dim, proj_dim=rnafm_proj_dim,
            )

        # -- Fused dimension --
        fused_dim = cnn_out_dim
        if self.use_rnafm:
            fused_dim += rnafm_proj_dim
        self.fused_dim = fused_dim

        # -- Optional RLPA --
        self.use_attention = use_rloop_attention
        if self.use_attention:
            from .attention.rloop_attention import RLoopAttention
            self.attention = RLoopAttention(
                d_model=fused_dim, d_ff=fused_dim * 2, dropout=0.1,
            )

        # -- Pooling --
        self.pool = nn.AdaptiveAvgPool1d(1)

        # -- Dense input = pooled fused features + optional scalars --
        self.n_scalar_features = n_scalar_features
        dense_input_dim = fused_dim + n_scalar_features

        # -- Efficiency head (always present) --
        self.efficiency_head = nn.Sequential(
            nn.Linear(dense_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # -- Discrimination head (optional) --
        self.multitask = multitask
        if self.multitask:
            self.disc_head = DiscriminationHead(dense_input_dim, hidden_dim)

        # For visualisation
        self._attn_weights: torch.Tensor | None = None

    def encode(
        self,
        target_onehot: torch.Tensor,
        crrna_rnafm_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared encoder: fuse branches -> optional attention -> per-position features.

        Args:
            target_onehot:   (batch, 4, 34) one-hot target DNA for CNN.
            crrna_rnafm_emb: (batch, 20, 640) pre-computed RNA-FM embeddings
                             for the crRNA spacer. None if use_rnafm=False.

        Returns:
            (batch, 34, fused_dim) per-position fused features.
        """
        cnn_feat = self.cnn(target_onehot)  # (batch, 34, cnn_out_dim)
        branches = [cnn_feat]

        if self.use_rnafm and crrna_rnafm_emb is not None:
            rnafm_feat = self.rnafm(crrna_rnafm_emb)  # (batch, 34, rnafm_proj_dim)
            branches.append(rnafm_feat)

        fused = torch.cat(branches, dim=-1)  # (batch, 34, fused_dim)

        if self.use_attention:
            fused, self._attn_weights = self.attention(fused)

        return fused

    def _pool_and_append_scalars(
        self,
        fused: torch.Tensor,
        scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool per-position features -> fixed-size vector, append scalars."""
        # (batch, fused_dim, 34) -> (batch, fused_dim, 1) -> (batch, fused_dim)
        pooled = self.pool(fused.permute(0, 2, 1)).squeeze(-1)
        if scalar_features is not None and self.n_scalar_features > 0:
            pooled = torch.cat([pooled, scalar_features], dim=-1)
        return pooled

    def forward(
        self,
        target_onehot: torch.Tensor,
        crrna_rnafm_emb: torch.Tensor | None = None,
        scalar_features: torch.Tensor | None = None,
        # For multi-task: wildtype TARGET DNA (crRNA stays the same)
        wt_target_onehot: torch.Tensor | None = None,
        # wt_crrna_rnafm_emb is NOT needed -- same guide for both conditions
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            target_onehot:    (batch, 4, 34) mutant target DNA (perfect match to guide).
            crrna_rnafm_emb:  (batch, 20, 640) crRNA spacer RNA-FM embeddings.
            scalar_features:  (batch, n_scalar_features) optional scalars (Evo 2 LLR, etc).
            wt_target_onehot: (batch, 4, 34) wildtype target DNA (one SNP mismatch).
                              Only needed when multitask=True.

        Returns:
            dict with keys:
                "efficiency":     (batch, 1) predicted trans-cleavage activity.
                "discrimination": (batch, 1) predicted disc ratio (if multitask + wt provided).
                "attn_weights":   (batch, 34, 34) attention map (if RLPA enabled).
        """
        # Encode mutant target (perfect match to crRNA)
        mut_feat = self.encode(target_onehot, crrna_rnafm_emb)
        mut_pooled = self._pool_and_append_scalars(mut_feat, scalar_features)

        output: dict[str, torch.Tensor] = {
            "efficiency": self.efficiency_head(mut_pooled),
        }

        # Multi-task: discrimination from paired targets
        if self.multitask and wt_target_onehot is not None:
            # Encode wildtype target with the SAME crRNA
            wt_feat = self.encode(wt_target_onehot, crrna_rnafm_emb)
            wt_pooled = self._pool_and_append_scalars(wt_feat, scalar_features)
            output["discrimination"] = self.disc_head(mut_pooled, wt_pooled)

        if self._attn_weights is not None:
            output["attn_weights"] = self._attn_weights

        return output

    def count_trainable_params(self) -> int:
        """Count trainable parameters (RNA-FM is frozen, only projection trains)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """Count all parameters including frozen."""
        return sum(p.numel() for p in self.parameters())
