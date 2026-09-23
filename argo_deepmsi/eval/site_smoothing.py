"""Label-free site smoothing for bag-level models under processing shift.

Three families, none of which sees an MSI label, so they can be evaluated on the full
cohort without nesting (site identity and unlabelled features only):

- **Input moment matching** (AdaBN / CORAL analogues for a frozen MIL model): map each
  site's tile-feature distribution onto a reference site before the pretrained Wagner
  transformer, per site (diagonal or full covariance) or per bag (instance statistics).
- **Score normalisation** (speaker-verification Z-norm / rank-norm analogues): re-centre
  patient scores within each site using that site's unlabelled score distribution.
- **Shift-invariant bag descriptors**: summarise each slide by statistics of its tiles
  after removing the slide's own mean and scale, so a bag-level affine site offset cancels
  while sparse within-bag outliers (candidate MSI morphology) remain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MomentAccumulator:
    """Streaming per-group mean / covariance of tile features."""

    def __init__(self, dim: int, full_cov: bool = True):
        self.n: dict[str, int] = {}
        self.s: dict[str, np.ndarray] = {}
        self.ss: dict[str, np.ndarray] = {}
        self.dim, self.full_cov = dim, full_cov

    def add(self, group: str, X: np.ndarray) -> None:
        X = X.astype(np.float64)
        if group not in self.n:
            self.n[group] = 0
            self.s[group] = np.zeros(self.dim)
            self.ss[group] = np.zeros((self.dim, self.dim) if self.full_cov else self.dim)
        self.n[group] += len(X)
        self.s[group] += X.sum(0)
        self.ss[group] += X.T @ X if self.full_cov else (X**2).sum(0)

    def moments(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        n, mu = self.n[group], self.s[group] / self.n[group]
        if self.full_cov:
            return mu, self.ss[group] / n - np.outer(mu, mu)
        return mu, self.ss[group] / n - mu**2


def _sqrtm_psd(C: np.ndarray, inverse: bool = False, eps: float = 1e-4) -> np.ndarray:
    w, V = np.linalg.eigh(C + eps * np.trace(C) / len(C) * np.eye(len(C)))
    w = np.clip(w, 1e-12, None)
    return (V * (w ** (-0.5 if inverse else 0.5))) @ V.T


def moment_match(
    X: np.ndarray,
    src: tuple[np.ndarray, np.ndarray],
    ref: tuple[np.ndarray, np.ndarray],
    *,
    mode: str = "diag",
    alpha: float = 1.0,
) -> np.ndarray:
    """Map tiles from the source moments onto the reference moments.

    ``mode='diag'`` matches per-dimension mean and scale (AdaBN-style);
    ``mode='coral'`` whitens with the source covariance and re-colours with the
    reference covariance. ``alpha`` interpolates between identity (0) and full (1).
    """
    mu_s, c_s = src
    mu_r, c_r = ref
    if mode == "diag":
        sd_s = np.sqrt(np.clip(c_s if c_s.ndim == 1 else np.diag(c_s), 1e-12, None))
        sd_r = np.sqrt(np.clip(c_r if c_r.ndim == 1 else np.diag(c_r), 1e-12, None))
        Y = (X - mu_s) / sd_s * sd_r + mu_r
    elif mode == "coral":
        A = _sqrtm_psd(c_s, inverse=True) @ _sqrtm_psd(c_r)
        Y = (X - mu_s) @ A + mu_r
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return ((1 - alpha) * X + alpha * Y).astype(np.float32)


def site_score_normalise(patients: pd.DataFrame, method: str, score_col: str = "score") -> pd.Series:
    """Label-free per-site normalisation of patient scores.

    ``znorm``: (logit − site median) / site MAD — median/MAD keep the MSS majority as
    the anchor so a site's MSI prevalence moves the centre little. ``rank``: within-site
    percentile.
    """
    p = patients[score_col].clip(1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    out = pd.Series(index=patients.index, dtype=float)
    for _, idx in patients.groupby("site").groups.items():
        v = logit.loc[idx]
        if method == "znorm":
            mad = (v - v.median()).abs().median() or 1.0
            out.loc[idx] = (v - v.median()) / mad
        elif method == "rank":
            out.loc[idx] = v.rank(pct=True)
        else:
            raise ValueError(f"unknown method {method!r}")
    return out


def invariant_bag_descriptor(X: np.ndarray, q: float = 0.9) -> np.ndarray:
    """Per-slide descriptor invariant to a bag-level per-dimension affine shift.

    Tiles are standardised by the slide's own mean/std; the descriptor concatenates
    the upper quantile (sparse high-expressing tiles) and the mean absolute deviation
    beyond that quantile. A site offset that shifts/rescales every tile of a bag cancels.
    """
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    hi = np.quantile(Z, q, axis=0)
    tail = np.where(Z > hi, Z - hi, 0.0).mean(0)
    return np.concatenate([hi, tail]).astype(np.float32)
