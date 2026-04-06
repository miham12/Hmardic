from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .binning import uniform_bins
from .overlaps import as_pyranges_intervals, overlaps_to_array


def _bins_coverage_uniform(
    contacts: pd.DataFrame,
    chrom_dict: Dict[str, int],
    bin_size: int,
) -> Tuple[float, float]:
    """Return mean/variance of overlap counts across uniform bins."""
    if contacts is None or contacts.empty:
        return 0.0, 0.0

    means = []
    vars_ = []
    for chrom, g in contacts.groupby("chr", observed=True):
        chrom = str(chrom)
        length = int(chrom_dict.get(chrom, 0))
        if length <= 0:
            continue

        bins = uniform_bins(length, int(bin_size))
        bins["chr"] = chrom
        n_bins = len(bins)
        if n_bins == 0:
            continue

        pr_bins = as_pyranges_intervals(bins[["chr", "start", "end", "bin_index"]], extra_cols=["bin_index"])
        pr_cont = as_pyranges_intervals(g[["chr", "start", "end"]])
        counts = overlaps_to_array(pr_bins, pr_cont, id_col="bin_index", n_bins=n_bins)

        means.append(float(np.mean(counts)))
        vars_.append(float(np.var(counts)))

    if not means:
        return 0.0, 0.0
    return float(np.mean(means)), float(np.mean(vars_))


def optimize_trans_bin_size(
    trans_contacts: pd.DataFrame,
    chrom_dict: Dict[str, int],
    *,
    gene_chrom: str,
    start: int,
    end: int,
    step: int,
    tolerance: float,
    w: int,
) -> Tuple[int, pd.DataFrame]:
    """Heuristic search for a good uniform genome-wide bin size."""
    gene_chrom = str(gene_chrom)
    if trans_contacts is None or trans_contacts.empty:
        return int(start), pd.DataFrame(columns=["bin_size", "mean", "var", "cost"])

    df = trans_contacts[trans_contacts["chr"].astype(str) != gene_chrom].copy()
    df["chr"] = df["chr"].astype(str)

    records = []
    for bs in range(int(start), int(end) + 1, int(step)):
        mean, var = _bins_coverage_uniform(df, chrom_dict, bs)
        cost = (2.0 * mean - var) / (float(bs) ** 2) if bs > 0 else 0.0
        records.append((bs, mean, var, cost))

    out = pd.DataFrame(records, columns=["bin_size", "mean", "var", "cost"])
    if out.empty:
        return int(start), out

    costs = out["cost"].to_numpy(dtype=float)
    best_idx = int(np.argmax(costs))
    best_bs = int(out.iloc[best_idx]["bin_size"])

    if len(costs) >= w + 1:
        for i in range(w, len(costs)):
            window = costs[i - w:i]
            if np.max(window) - np.min(window) < tolerance * (abs(np.mean(window)) + 1e-12):
                best_idx = int(np.argmax(costs[:i]))
                best_bs = int(out.iloc[best_idx]["bin_size"])
                break

    return best_bs, out
