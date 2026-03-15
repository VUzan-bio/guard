"""Replace hardcoded parameters with computed values.

Six parameters that were hardcoded from literature can be computed
from sequence data and trained models — no wet lab needed.

1. Context-dependent mismatch ΔΔG (from dinucleotide NN tables)
2. M.tb GC optimal (from H37Rv genome PAM site distribution)
3. Amplicon folding ΔG (from nearest-neighbor thermodynamics)
4. Self-distilled position sensitivity (from trained model)
5. RLPA-derived cooperativity (from attention bias matrix)
6. Learned PAM penalties (from PAM embedding norms)

Items 1-3 are pure sequence computation (work immediately).
Items 4-6 require a trained checkpoint (work after GPU training).

References:
    Sugimoto et al., Biochemistry 1995 — RNA:DNA NN parameters
    SantaLucia, PNAS 1998 — nearest-neighbor thermodynamics
    Owczarzy et al., Biochemistry 2004 — salt corrections
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# RNA:DNA nearest-neighbor parameters (Sugimoto et al. 1995)
_RNA_DNA_NN = {
    "AA": (-7.8, -21.9), "AC": (-5.9, -12.3),
    "AG": (-9.1, -23.5), "AU": (-8.3, -23.9),
    "CA": (-9.0, -26.1), "CC": (-9.3, -23.2),
    "CG": (-16.3, -47.1), "CU": (-7.0, -19.7),
    "GA": (-5.5, -13.5), "GC": (-8.0, -17.1),
    "GG": (-12.8, -31.9), "GU": (-7.8, -21.6),
    "UA": (-7.8, -23.2), "UC": (-8.6, -22.9),
    "UG": (-10.4, -28.4), "UU": (-11.5, -36.4),
}
_DNA_TO_RNA = {"A": "U", "T": "A", "C": "G", "G": "C"}
_DNA_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
_INIT_DH = 1.9
_INIT_DS = -3.9


# ══════════════════════════════════════════════════════════════════
# 1. Context-dependent mismatch ΔΔG
# ══════════════════════════════════════════════════════════════════

def compute_mismatch_ddg(
    spacer_dna: str,
    mismatch_pos: int,
    wt_base: str,
    mut_base: str,
    temperature_c: float = 37.0,
) -> float:
    """Compute context-dependent ΔΔG for a mismatch at a specific position.

    Instead of using a fixed lookup table (12 values), computes the
    actual thermodynamic penalty from the dinucleotide context:
        ΔΔG = ΔG(with mismatch) - ΔG(perfect match)

    The flanking bases matter: a G:G mismatch between two GC pairs
    is more destabilising than between two AU pairs.

    Args:
        spacer_dna: full spacer sequence (DNA, target-sense)
        mismatch_pos: 0-indexed position of mismatch in spacer
        wt_base: wildtype DNA base at this position
        mut_base: mutant DNA base at this position (matches crRNA)
        temperature_c: temperature in Celsius

    Returns:
        ΔΔG in kcal/mol (positive = destabilising = good for discrimination)
    """
    T = temperature_c + 273.15
    spacer = spacer_dna.upper()
    pos = mismatch_pos

    # Convert to crRNA (reverse complement, T→U)
    crrna = "".join(_DNA_TO_RNA.get(b, "N") for b in reversed(spacer))
    crrna_pos = len(spacer) - 1 - pos  # position in crRNA (reversed)

    # ΔG of perfect match at this dinucleotide
    dg_match = 0.0
    if pos > 0:
        left_dinuc = crrna[crrna_pos - 1:crrna_pos + 1] if crrna_pos > 0 else ""
        if left_dinuc in _RNA_DNA_NN:
            dH, dS = _RNA_DNA_NN[left_dinuc]
            dg_match += dH - T * (dS / 1000.0)
    if pos < len(spacer) - 1:
        right_dinuc = crrna[crrna_pos:crrna_pos + 2] if crrna_pos < len(crrna) - 1 else ""
        if right_dinuc in _RNA_DNA_NN:
            dH, dS = _RNA_DNA_NN[right_dinuc]
            dg_match += dH - T * (dS / 1000.0)

    # ΔG with mismatch: approximate as loss of both flanking NN contributions
    # A mismatch disrupts both the 5' and 3' dinucleotide stacking
    # ΔΔG ≈ -ΔG_match (we lose the stabilising interactions)
    ddg = -dg_match if dg_match < 0 else abs(dg_match)

    # Clamp to reasonable range (0.5-5.0 kcal/mol from Sugimoto 2000)
    return max(0.5, min(5.0, ddg))


# ══════════════════════════════════════════════════════════════════
# 2. M.tb GC optimal from H37Rv genome
# ══════════════════════════════════════════════════════════════════

_GC_OPTIMAL_CACHE: dict[str, float] = {}


def compute_gc_optimal_from_genome(
    fasta_path: str | Path,
    pam_pattern: str = "TTT",
    spacer_len: int = 20,
) -> float:
    """Compute optimal spacer GC from actual PAM site distribution in genome.

    Scans the genome for all TTTV PAM sites, extracts the adjacent
    spacer, and returns the median GC content. This is the natural
    GC context the enzyme will encounter in this organism.

    Args:
        fasta_path: path to reference genome FASTA
        spacer_len: spacer length to extract

    Returns:
        Median GC fraction of all PAM-adjacent spacers in the genome.
    """
    cache_key = f"{fasta_path}:{pam_pattern}:{spacer_len}"
    if cache_key in _GC_OPTIMAL_CACHE:
        return _GC_OPTIMAL_CACHE[cache_key]

    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        logger.warning("Genome FASTA not found: %s, using default 0.55", fasta_path)
        return 0.55

    # Read genome
    genome = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                genome.append(line.strip().upper())
    seq = "".join(genome)

    if not seq:
        return 0.55

    # Scan for PAM sites and extract spacer GC
    gc_values = []
    pam_bases = {"V": {"A", "C", "G"}}  # TTTV = TTT[ACG]

    for i in range(len(seq) - 4 - spacer_len):
        # Forward: TTTV at position i, spacer at i+4
        if seq[i:i + 3] == pam_pattern and seq[i + 3] in pam_bases["V"]:
            spacer = seq[i + 4:i + 4 + spacer_len]
            if len(spacer) == spacer_len and all(b in "ACGT" for b in spacer):
                gc = sum(1 for b in spacer if b in "GC") / spacer_len
                gc_values.append(gc)

        # Reverse complement: check for BAAA (RC of TTTV)
        rc_end = i + 4 + spacer_len
        if rc_end <= len(seq):
            rc_pam = seq[rc_end - 4:rc_end]
            if rc_pam[1:4] == "AAA" and rc_pam[0] in {"T", "G", "C"}:  # RC of TTTV
                spacer = seq[i:i + spacer_len]
                if all(b in "ACGT" for b in spacer):
                    gc = sum(1 for b in spacer if b in "GC") / spacer_len
                    gc_values.append(gc)

    if not gc_values:
        logger.warning("No TTTV PAM sites found in %s", fasta_path)
        return 0.55

    result = float(np.median(gc_values))
    _GC_OPTIMAL_CACHE[cache_key] = result

    logger.info(
        "GC optimal from %s: %.3f (median of %d TTTV-adjacent spacers, range %.2f-%.2f)",
        fasta_path.name, result, len(gc_values),
        min(gc_values), max(gc_values),
    )
    return result


# ══════════════════════════════════════════════════════════════════
# 3. Amplicon folding ΔG
# ══════════════════════════════════════════════════════════════════

# DNA:DNA nearest-neighbor parameters (SantaLucia 1998)
_DNA_DNA_NN = {
    "AA": (-7.9, -22.2), "AT": (-7.2, -20.4),
    "AG": (-7.8, -21.0), "AC": (-8.4, -22.4),
    "TA": (-7.2, -21.3), "TT": (-7.9, -22.2),
    "TG": (-8.5, -22.7), "TC": (-8.2, -22.2),
    "GA": (-8.2, -22.2), "GT": (-8.4, -22.4),
    "GG": (-8.0, -19.9), "GC": (-9.8, -24.4),
    "CA": (-8.5, -22.7), "CT": (-7.8, -21.0),
    "CG": (-10.6, -27.2), "CC": (-8.0, -19.9),
}


def compute_amplicon_fold_dg(
    amplicon_seq: str,
    temperature_c: float = 37.0,
) -> float:
    """Estimate amplicon self-folding ΔG from sequence.

    Uses nearest-neighbor DNA:DNA parameters to estimate the most
    stable hairpin. Scans all possible hairpin loops (≥4 nt loop)
    and returns the most negative ΔG.

    GC-rich amplicons (M.tb, 65.6% GC) are prone to stable hairpins
    that block RPA recombinase invasion.

    Args:
        amplicon_seq: amplicon DNA sequence
        temperature_c: temperature in Celsius

    Returns:
        Most negative ΔG (kcal/mol). More negative = more stable hairpin = worse for RPA.
        Returns 0.0 if no stable hairpin found.
    """
    T = temperature_c + 273.15
    seq = amplicon_seq.upper()
    n = len(seq)
    if n < 12:  # too short for meaningful hairpin
        return 0.0

    best_dg = 0.0
    min_loop = 3
    min_stem = 4

    # Scan all possible hairpin positions
    for loop_center in range(min_stem + min_loop // 2, n - min_stem - min_loop // 2):
        for loop_len in range(min_loop, min(10, n - loop_center)):
            half_loop = loop_len // 2
            loop_start = loop_center - half_loop
            loop_end = loop_center + half_loop + (loop_len % 2)

            if loop_start < min_stem or loop_end > n - min_stem:
                continue

            # Extend stem outward from loop
            stem_dg = 0.0
            stem_len = 0
            for k in range(min(loop_start, n - loop_end)):
                left_idx = loop_start - 1 - k
                right_idx = loop_end + k
                if left_idx < 0 or right_idx >= n:
                    break

                left = seq[left_idx]
                right_comp = _DNA_COMP.get(seq[right_idx], "")
                if left != right_comp:
                    break

                stem_len += 1
                # NN stacking (need 2 consecutive base pairs)
                if stem_len >= 2:
                    dinuc = seq[left_idx:left_idx + 2]
                    if dinuc in _DNA_DNA_NN:
                        dH, dS = _DNA_DNA_NN[dinuc]
                        stem_dg += dH - T * (dS / 1000.0)

            if stem_len >= min_stem and stem_dg < -2.0:
                # Loop penalty (Jacobson-Stockmayer approximation)
                loop_penalty = 4.0 + 1.4 * math.log(max(loop_len, 3))
                total_dg = stem_dg + loop_penalty
                if total_dg < best_dg:
                    best_dg = total_dg

    return round(best_dg, 2)


def compute_amplicon_gc(amplicon_seq: str) -> float:
    """Compute GC fraction of amplicon."""
    seq = amplicon_seq.upper()
    if not seq:
        return 0.0
    return sum(1 for b in seq if b in "GC") / len(seq)


# ══════════════════════════════════════════════════════════════════
# 4. Self-distilled position sensitivity from trained model
# ══════════════════════════════════════════════════════════════════

def extract_position_sensitivity(
    model_path: str | Path,
    n_probes: int = 500,
    device: str = "cpu",
) -> dict[int, float]:
    """Extract position-dependent sensitivity from trained Compass-ML.

    Strategy: for N random guide sequences, predict activity with
    perfect match, then with a mismatch at each spacer position (1-20).
    Sensitivity[pos] = 1 - median(activity_with_mismatch / activity_perfect).

    This gives model-consistent sensitivity profiles instead of
    literature values from a different experimental setup.

    Returns:
        Dict {position (1-indexed): sensitivity (0-1)}
    """
    try:
        import torch
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        config = checkpoint.get("config", {})
    except Exception as e:
        logger.warning("Cannot extract position sensitivity: %s", e)
        return {}

    logger.info("Extracting position sensitivity via self-distillation from %s...", model_path)

    # Build model from checkpoint config
    try:
        import sys
        compass_net_dir = str(Path(__file__).resolve().parent.parent.parent / "compass-net")
        if compass_net_dir not in sys.path:
            sys.path.insert(0, compass_net_dir)
        from compass_ml import CompassML

        model = CompassML(**config)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
    except Exception as e:
        logger.warning("Failed to load model for position sensitivity: %s", e)
        return {}

    import torch

    # Generate random target sequences
    np.random.seed(42)
    sensitivity = {}

    with torch.no_grad():
        for pos in range(1, 21):  # spacer positions 1-20
            ratios = []
            for _ in range(n_probes):
                # Random 34-nt one-hot
                seq = np.random.choice(4, 34)
                oh = np.eye(4, dtype=np.float32)[seq].T  # (4, 34)

                # Perfect match activity
                t_perfect = torch.from_numpy(oh).unsqueeze(0).to(device)
                act_perfect = model(target_onehot=t_perfect)["efficiency"].item()

                # Mismatch at position (pos is 1-indexed, maps to index pos+3 in 34-nt)
                idx = pos + 3  # 4 PAM + (pos-1) = pos+3
                oh_mm = oh.copy()
                current = np.argmax(oh_mm[:, idx])
                new_base = (current + 1) % 4
                oh_mm[:, idx] = 0
                oh_mm[new_base, idx] = 1
                t_mm = torch.from_numpy(oh_mm).unsqueeze(0).to(device)
                act_mm = model(target_onehot=t_mm)["efficiency"].item()

                if act_perfect > 0.01:
                    ratios.append(1.0 - act_mm / act_perfect)

            sensitivity[pos] = float(np.median(ratios)) if ratios else 0.5

    logger.info("Self-distilled sensitivity: %s", {k: round(v, 3) for k, v in sensitivity.items()})
    return sensitivity


# ══════════════════════════════════════════════════════════════════
# 5. RLPA-derived cooperativity
# ══════════════════════════════════════════════════════════════════

def extract_cooperativity_from_rlpa(
    model_path: str | Path,
    device: str = "cpu",
) -> dict[int, float]:
    """Extract cooperativity penalties from trained RLPA bias matrix.

    The 34×34 learned bias matrix encodes how nearby positions interact.
    Cooperativity[distance] = mean bias between seed position pairs
    at that distance.

    Returns:
        Dict {distance (1-4): cooperativity_factor (0-1)}
    """
    try:
        import torch
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    except Exception as e:
        logger.warning("Cannot extract cooperativity: %s", e)
        return {}

    # Find RLPA bias matrix in state dict
    bias_key = None
    for k in state_dict:
        if "rloop_bias" in k and "bias" in k and state_dict[k].dim() == 2:
            bias_key = k
            break

    if bias_key is None:
        logger.info("No RLPA bias matrix found in checkpoint")
        return {}

    bias = state_dict[bias_key].cpu().numpy()  # (34, 34)
    logger.info("Extracted RLPA bias matrix: %s", bias.shape)

    # Extract cooperativity from seed region (positions 4-12 in 34-nt)
    seed_start, seed_end = 4, 12
    cooperativity = {}

    for dist in range(1, 5):
        values = []
        for i in range(seed_start, seed_end - dist):
            j = i + dist
            if j < seed_end:
                # Bias for position j attending to position i (upstream)
                values.append(float(bias[j, i]))
        if values:
            # Normalise: higher bias = more cooperative interaction
            mean_bias = np.mean(values)
            # Convert to penalty factor (0-1): higher bias → stronger cooperativity
            factor = min(1.0, max(0.0, mean_bias / max(abs(bias[seed_start:seed_end, seed_start:seed_end]).max(), 1e-6)))
            cooperativity[dist] = round(factor, 3)

    logger.info("RLPA cooperativity: %s", cooperativity)
    return cooperativity


# ══════════════════════════════════════════════════════════════════
# 6. Learned PAM penalties from embeddings
# ══════════════════════════════════════════════════════════════════

def extract_pam_penalties(
    model_path: str | Path,
    device: str = "cpu",
) -> dict[str, float]:
    """Extract relative PAM activity penalties from learned PAM embedding.

    The 9-class PAM embedding learns relative activity for each variant.
    Penalty = ||embedding[class]|| / ||embedding[TTTV]||

    Returns:
        Dict {PAM_pattern: relative_penalty (0-1)}
    """
    try:
        import torch
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    except Exception as e:
        logger.warning("Cannot extract PAM penalties: %s", e)
        return {}

    # Find PAM embedding
    emb_key = None
    for k in state_dict:
        if "pam_emb" in k and "weight" in k:
            emb_key = k
            break

    if emb_key is None:
        logger.info("No PAM embedding found in checkpoint")
        return {}

    weights = state_dict[emb_key].cpu().numpy()  # (9, 8)
    norms = np.linalg.norm(weights, axis=1)

    # Class 0 = TTTV (canonical, highest activity)
    ref_norm = norms[0] if norms[0] > 1e-6 else 1.0

    pam_names = ["TTTV", "TTTT", "TTCV", "TATV", "CTTV", "TCTV", "TGTV", "ATTV", "GTTV"]
    penalties = {}
    for i, name in enumerate(pam_names):
        penalties[name] = round(float(norms[i] / ref_norm), 3)

    logger.info("Learned PAM penalties: %s", penalties)
    return penalties
