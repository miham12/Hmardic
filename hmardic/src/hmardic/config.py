from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HmardicParams:
    # ── Fixed-bin (fast) mode parameters ──────────────────────────────────────
    # If BOTH bin_size and factor are provided, we do NOT run optimization.
    bin_size: int | None = None
    cis_start_size: int = 1000
    factor: float | None = None

    # ── Signal/background model ──────────────────────────────────────────────
    decay: float = 1e6
    pseudo: float = 1e-10
    min_lambda: float = 1.0

    # ── Optimization (used only if bin_size/factor is None) ───────────────────
    trans_start: int = 10_000
    trans_end: int = 1_000_000
    trans_step: int = 1_000

    cis_factor_min: float = 1.10
    cis_factor_max: float = 2.00
    cis_factor_step: float = 0.01

    tolerance: float = 0.01
    w: int = 5  # moving window size for heuristic stopping

    # ── HMM parameters ───────────────────────────────────────────────────────
    p_bg: float = 1.0
    alpha: float = 5.0
    max_iter: int = 15
    tol: float = 1e-3

    # ── Post-processing ──────────────────────────────────────────────────────
    return_all_bins: bool = False

    @property
    def fixed_bins(self) -> bool:
        return self.bin_size is not None and self.factor is not None
