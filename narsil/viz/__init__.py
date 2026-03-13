"""NARSIL visualisation module.

Publication-quality figures for CRISPR-Cas12a crRNA design analysis.
All plots follow Nature Methods / NAR figure conventions:
- Matplotlib with custom rcParams (no seaborn defaults)
- Two-column journal width (180 mm) or single-column (88 mm)
- Consistent colour palette across all figures
- Vector output (PDF/SVG) by default
"""

from narsil.viz.style import apply_style, PALETTE, save_figure
from narsil.viz.discrimination import DiscriminationHeatmap
from narsil.viz.ranking import CandidateRankingPlot
from narsil.viz.multiplex import MultiplexMatrixPlot
from narsil.viz.benchmark import ModelBenchmarkPlot
from narsil.viz.active_learning import ActiveLearningPlot
from narsil.viz.target_overview import TargetDashboard

__all__ = [
    "apply_style",
    "PALETTE",
    "save_figure",
    "DiscriminationHeatmap",
    "CandidateRankingPlot",
    "MultiplexMatrixPlot",
    "ModelBenchmarkPlot",
    "ActiveLearningPlot",
    "TargetDashboard",
]
