from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from .binning import uniform_bins
from .cis import preprocess_bins_for_rna
from .overlaps import as_pyranges_intervals, overlaps_to_array


def _bins_coverage_uniform(
    contacts: pd.DataFrame,
    chrom_dict: Dict[str, int],
    bin_size: int,
) -> Tuple[float, float]:
    """Return (mean, var) of overlaps counts across uniform bins over all chromosomes in contacts."""
    if contacts is None or contacts.empty:
        return 0.0, 0.0

    # group per chromosome to avoid building huge genome-wide bins
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

        pr_bins = as_pyranges_intervals(bins[["chr","start","end","bin_index"]], extra_cols=["bin_index"])
        pr_cont = as_pyranges_intervals(g[["chr","start","end"]])
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
    """Heuristic search for a good trans bin size.

    Uses the same score as your earlier code:
      cost = (2*mean - var) / bin_size^2

    Stops when the moving window of cost stops improving.
    """
    gene_chrom = str(gene_chrom)
    # keep only trans chromosomes (just in case)
    if trans_contacts is None or trans_contacts.empty:
        return int(start), pd.DataFrame(columns=["bin_size","mean","var","cost"])

    df = trans_contacts[trans_contacts["chr"].astype(str) != gene_chrom].copy()
    df["chr"] = df["chr"].astype(str)

    records = []
    for bs in range(int(start), int(end) + 1, int(step)):
        mean, var = _bins_coverage_uniform(df, chrom_dict, bs)
        cost = (2.0 * mean - var) / (float(bs) ** 2) if bs > 0 else 0.0
        records.append((bs, mean, var, cost))

    out = pd.DataFrame(records, columns=["bin_size","mean","var","cost"])
    if out.empty:
        return int(start), out

    # smoothing & stopping: choose best in the first region where costs plateau
    costs = out["cost"].to_numpy(dtype=float)
    best_idx = int(np.argmax(costs))
    best_bs = int(out.iloc[best_idx]["bin_size"])

    # optional heuristic plateau detection
    if len(costs) >= w + 1:
        for i in range(w, len(costs)):
            window = costs[i-w:i]
            # if window change small, stop
            if np.max(window) - np.min(window) < tolerance * (abs(np.mean(window)) + 1e-12):
                best_idx = int(np.argmax(costs[:i]))
                best_bs = int(out.iloc[best_idx]["bin_size"])
                break

    return best_bs, out


def _bins_coverage_cis_factor(
    cis_contacts: pd.DataFrame,
    chrom_dict: Dict[str, int],
    *,
    gene_chrom: str,
    gene_start: int,
    gene_end: int,
    bin_size: int,
    cis_start: int,
    factor: float,
) -> Tuple[float, float]:
    """Return (mean,var) of overlaps over cis geometric bins for a candidate factor."""
    gene_chrom = str(gene_chrom)
    length = int(chrom_dict.get(gene_chrom, 0))
    if length <= 0:
        return 0.0, 0.0

    ups, _, dns = preprocess_bins_for_rna(int(gene_start), int(gene_end), length, int(bin_size), int(cis_start), float(factor))
    bins = pd.DataFrame(ups + dns)
    if bins.empty:
        return 0.0, 0.0
    bins["chr"] = gene_chrom
    bins["bin_index"] = np.arange(len(bins), dtype=int)

    pr_bins = as_pyranges_intervals(bins[["chr","start","end","bin_index"]], extra_cols=["bin_index"])
    if cis_contacts is None or cis_contacts.empty:
        counts = np.zeros(len(bins), dtype=float)
    else:
        cont = cis_contacts.copy()
        cont["chr"] = cont["chr"].astype(str)
        pr_cont = as_pyranges_intervals(cont[["chr","start","end"]])
        counts = overlaps_to_array(pr_bins, pr_cont, id_col="bin_index", n_bins=len(bins))

    return float(np.mean(counts)), float(np.var(counts))


def optimize_cis_factor(
    cis_contacts: pd.DataFrame,
    chrom_dict: Dict[str, int],
    *,
    gene_chrom: str,
    gene_start: int,
    gene_end: int,
    cis_start: int,
    factor_min: float,
    factor_max: float,
    factor_step: float,
    tolerance: float,
    w: int,
    max_linear_size: int,
) -> Tuple[float, pd.DataFrame]:
    """Heuristic search for cis geometric growth factor.

    cost = (2*mean - var) / factor^2 (as in your previous version)
    """
    records = []
    facs = np.arange(float(factor_min), float(factor_max) + 1e-12, float(factor_step))
    for fac in facs:
        mean, var = _bins_coverage_cis_factor(
            cis_contacts, chrom_dict,
            gene_chrom=gene_chrom, gene_start=gene_start, gene_end=gene_end,
            bin_size=max_linear_size, cis_start=cis_start, factor=fac
        )
        cost = (2.0 * mean - var) / (float(fac) ** 2) if fac > 0 else 0.0
        records.append((fac, mean, var, cost))

    out = pd.DataFrame(records, columns=["factor","mean","var","cost"])
    if out.empty:
        return float(factor_min), out

    costs = out["cost"].to_numpy(dtype=float)
    best_idx = int(np.argmax(costs))
    best_fac = float(out.iloc[best_idx]["factor"])

    if len(costs) >= w + 1:
        for i in range(w, len(costs)):
            window = costs[i-w:i]
            if np.max(window) - np.min(window) < tolerance * (abs(np.mean(window)) + 1e-12):
                best_idx = int(np.argmax(costs[:i]))
                best_fac = float(out.iloc[best_idx]["factor"])
                break

    return best_fac, out
