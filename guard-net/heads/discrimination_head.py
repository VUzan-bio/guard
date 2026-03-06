"""Discrimination head -- predict MUT/WT activity ratio.

Computes the discrimination ratio D = A_MUT / A_WT from the contrastive
difference between mutant-target and wildtype-target pooled representations.

The head sees [mut, wt, mut-wt, mut*wt]. The difference and interaction
terms force the shared encoder to learn features sensitive to single-
nucleotide changes in the TARGET DNA. This is exactly what diagnostic
SNP detection requires.

Biology: the crRNA guide is FIXED. The encoder sees the same crRNA
(via RNA-FM) for both mutant and wildtype. Only the CNN features differ
because the target DNA differs by one nucleotide at the SNP position.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiscriminationHead(nn.Module):
    """Predict discrimination ratio from paired encoder representations.

    Input: two pooled vectors (mut_pooled, wt_pooled), each of size input_dim.
    Features: [mut, wt, mut-wt, mut*wt] = 4x input_dim.
    Output: scalar > 0 (Softplus ensures positivity).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, 1),
            nn.Softplus(),  # disc ratio > 0 always
        )

    def forward(
        self, mut_pooled: torch.Tensor, wt_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mut_pooled: (batch, input_dim) encoder output for mutant target.
            wt_pooled:  (batch, input_dim) encoder output for wildtype target.

        Returns:
            (batch, 1) predicted discrimination ratio.
        """
        combined = torch.cat([
            mut_pooled,
            wt_pooled,
            mut_pooled - wt_pooled,
            mut_pooled * wt_pooled,
        ], dim=-1)
        return self.head(combined)
