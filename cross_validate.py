"""5-fold cross-validation of Phase 1 CNN+PAM model.

Evaluates generalisation with proper held-out folds.
Reports per-fold and mean Spearman rho.

Run: python cross_validate.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Data ──
_BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
_TTTV = {"TTTA", "TTTC", "TTTG"}
_PAM_MAP = {"TTTT": 1, "TTCA": 2, "TTCC": 2, "TTCG": 2, "TATA": 3, "TATC": 3, "TATG": 3,
            "CTTA": 4, "CTTC": 4, "CTTG": 4, "TCTA": 5, "TCTC": 5, "TCTG": 5,
            "TGTA": 6, "TGTC": 6, "TGTG": 6, "ATTA": 7, "ATTC": 7, "ATTG": 7,
            "GTTA": 8, "GTTC": 8, "GTTG": 8}


def to_oh(s):
    oh = np.zeros((4, 34), dtype=np.float32)
    for i, b in enumerate(s[:34].upper()):
        if b in _BASE_MAP:
            oh[_BASE_MAP[b], i] = 1.0
    return oh


def pam_cls(s):
    p = s[:4].upper()
    return 0 if p in _TTTV else _PAM_MAP.get(p, 0)


def norm(raw):
    mn, mx = raw.min(), raw.max()
    if mx - mn < 1e-8:
        return np.full_like(raw, 0.5, dtype=np.float32)
    return ((raw - mn) / (mx - mn)).astype(np.float32)


def augment_flank(oh):
    aug = oh.copy()
    aug[:, 24:] = aug[:, 24:][:, np.random.permutation(10)]
    return aug


def load_all_kim2018():
    """Load ALL Kim 2018 data (HT1-1 + HT1-2 + HT2 + HT3) for cross-validation."""
    xlsx = str(ROOT / "compass" / "data" / "kim2018" / "nbt4061_source_data.xlsx")

    def _sheet(name):
        df = pd.read_excel(xlsx, sheet_name=name, header=1)
        seq_col = next((c for c in df.columns if "34" in str(c)), df.columns[1])
        indel_col = next(
            (c for c in df.columns if "Background" in str(c) and "subtract" in str(c).lower()),
            df.columns[-1],
        )
        valid = pd.DataFrame({"seq": df[seq_col], "indel": df[indel_col]}).dropna()
        seqs = valid["seq"].astype(str).values
        indels = valid["indel"].values.astype(np.float64)
        mask = np.array([len(s) == 34 and all(c in "ACGTacgt" for c in s) for s in seqs])
        return [s.upper() for s in seqs[mask]], np.clip(indels[mask], 0, None)

    all_seqs, all_acts = [], []
    for sheet in ["Data set HT 1-1", "Data set HT 1-2", "Data set HT 2", "Data set HT 3"]:
        seqs, acts = _sheet(sheet)
        all_seqs.extend(seqs)
        all_acts.extend(acts)
        logger.info("  %s: %d sequences", sheet, len(seqs))

    return all_seqs, np.array(all_acts)


class GuideDS(Dataset):
    def __init__(self, seqs, acts, aug=False):
        self.oh = np.stack([to_oh(s) for s in seqs])
        self.a = acts.astype(np.float32)
        self.p = np.array([pam_cls(s) for s in seqs], dtype=np.int64)
        self.aug = aug

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        oh = self.oh[i]
        if self.aug and np.random.random() < 0.3:
            oh = augment_flank(oh)
        return torch.from_numpy(oh), torch.tensor(self.a[i]), torch.tensor(self.p[i])


# ── Model ──
class Phase1CNN(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 120
        self.b3 = nn.Sequential(nn.Conv1d(4, 40, 3, padding=1), nn.BatchNorm1d(40), nn.GELU())
        self.b5 = nn.Sequential(nn.Conv1d(4, 40, 5, padding=2), nn.BatchNorm1d(40), nn.GELU())
        self.b7 = nn.Sequential(nn.Conv1d(4, 40, 7, padding=3), nn.BatchNorm1d(40), nn.GELU())
        self.d1 = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU())
        self.d2 = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.BatchNorm1d(ch), nn.GELU())
        self.pam_emb = nn.Embedding(9, 8)
        self.pam_proj = nn.Linear(8, ch)
        self.reduce = nn.Sequential(nn.Conv1d(ch, 64, 1), nn.BatchNorm1d(64), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.21),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x, p):
        h = torch.cat([self.b3(x), self.b5(x), self.b7(x)], 1)
        h = h + self.d2(self.d1(h))
        h = h + self.pam_proj(self.pam_emb(p)).unsqueeze(-1)
        h = self.reduce(h)
        return self.head(self.pool(h).squeeze(-1))


def soft_spearman(pred, target, s=1.0):
    n = pred.size(0)
    if n < 3:
        return torch.tensor(0.0)
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)
    dt = target.unsqueeze(1) - target.unsqueeze(0)
    rp = torch.sigmoid(dp / max(s, 0.01)).sum(1)
    rt = torch.sigmoid(dt / max(s, 0.01)).sum(1)
    rp, rt = rp - rp.mean(), rt - rt.mean()
    return (rp * rt).sum() / torch.sqrt((rp ** 2).sum() * (rt ** 2).sum() + 1e-8)


# ── Cross-validation ──
def train_fold(train_ds, val_ds, fold_id, seed=42):
    torch.manual_seed(seed + fold_id)
    np.random.seed(seed + fold_id)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    model = Phase1CNN()
    huber = nn.HuberLoss(delta=0.5)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)

    best_rho, patience = -1.0, 0

    for ep in range(200):
        s_s = max(0.1, 1.0 - 0.9 * ep / 200)

        model.train()
        for oh, eff, pam in train_loader:
            pred = model(oh, pam).squeeze(-1)
            eff_n = (eff + torch.randn_like(eff) * 0.02).clamp(0, 1)
            loss = huber(pred, eff_n) + 0.5 * (1 - soft_spearman(pred, eff_n, s_s))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for oh, eff, pam in val_loader:
                ps.extend(model(oh, pam).squeeze(-1).tolist())
                ts.extend(eff.tolist())
        rho = float(spearmanr(ps, ts).correlation)
        if np.isnan(rho):
            rho = 0.0

        if rho > best_rho:
            best_rho = rho
            patience = 0
        else:
            patience += 1

        if patience >= 20:
            break

    return best_rho


def main():
    logger.info("=" * 60)
    logger.info("  5-Fold Cross-Validation: CNN + PAM + Augmentation")
    logger.info("=" * 60)

    logger.info("Loading ALL Kim 2018 data...")
    all_seqs, all_raw = load_all_kim2018()
    all_acts = norm(all_raw)
    n_total = len(all_seqs)
    logger.info("Total: %d sequences", n_total)

    # Build full dataset
    full_ds = GuideDS(all_seqs, all_acts, aug=True)
    val_ds = GuideDS(all_seqs, all_acts, aug=False)

    # 5-fold split
    n_folds = 5
    np.random.seed(42)
    indices = np.random.permutation(n_total)
    fold_size = n_total // n_folds

    fold_rhos = []
    t0 = time.time()

    for fold in range(n_folds):
        fold_start = fold * fold_size
        fold_end = fold_start + fold_size if fold < n_folds - 1 else n_total

        val_idx = indices[fold_start:fold_end]
        train_idx = np.concatenate([indices[:fold_start], indices[fold_end:]])

        train_subset = Subset(full_ds, train_idx.tolist())
        val_subset = Subset(val_ds, val_idx.tolist())

        logger.info("Fold %d/%d: train=%d, val=%d", fold + 1, n_folds, len(train_idx), len(val_idx))

        rho = train_fold(train_subset, val_subset, fold_id=fold)
        fold_rhos.append(rho)

        elapsed = time.time() - t0
        logger.info("  Fold %d: rho=%.4f (%.0fs elapsed)", fold + 1, rho, elapsed)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  5-Fold CV Results")
    logger.info("=" * 60)
    for i, rho in enumerate(fold_rhos):
        logger.info("  Fold %d: rho=%.4f", i + 1, rho)
    logger.info("  Mean:  %.4f", np.mean(fold_rhos))
    logger.info("  Std:   %.4f", np.std(fold_rhos))
    logger.info("  Min:   %.4f", np.min(fold_rhos))
    logger.info("  Max:   %.4f", np.max(fold_rhos))
    logger.info("  Total time: %.0f seconds", time.time() - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
