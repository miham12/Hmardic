from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HmardicParams:
    # ── Uniform genome-wide binning ───────────────────────────────────────────
    bin_size: int | None = None

    # ── Signal/background model ──────────────────────────────────────────────
    decay: float = 1e6
    pseudo: float = 1e-10
    min_lambda: float = 1.0

    # ── Bin-size optimization (used only if bin_size is None) ───────────────
    trans_start: int = 10_000
    trans_end: int = 1_000_000
    trans_step: int = 1_000
    tolerance: float = 0.01
    w: int = 5

    # ── HMM parameters ───────────────────────────────────────────────────────
    p_bg: float = 1.0
    alpha: float = 5.0
    max_iter: int = 15
    tol: float = 1e-3

    # ── Output ───────────────────────────────────────────────────────────────
    return_all_bins: bool = False
