from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import pyranges as pr

from .binning import uniform_bins
from .nonspecific import NonspecificIndex
from .overlaps import as_pyranges_intervals, overlaps_to_array


@dataclass(frozen=True)
class TransBinCache:
    """Cached trans bins for fixed-bin mode (fast path).

    For each chromosome:
      - bins_by_chr[chr] is a DataFrame with columns:
          dna_chr, bin_index, start, end, center, ns_count, ns_total, bkg
      - pr_bins_by_chr[chr] is PyRanges of the bins with bin_index preserved (0..n-1)
      - n_bins_by_chr[chr] is the number of bins on that chromosome
    """
    bin_size: int
    bins_by_chr: Dict[str, pd.DataFrame]
    pr_bins_by_chr: Dict[str, pr.PyRanges]
    n_bins_by_chr: Dict[str, int]


@dataclass
class PrecomputeContext:
    """Everything we can (and should) precompute once per run (or once per worker)."""
    chrom_dict: Dict[str, int]         # chr -> length
    chrom_names: List[str]             # list of chr names in consistent order
    ns_index: NonspecificIndex
    trans_cache: Optional[TransBinCache] = None
    trans_cache_by_bin_size: Dict[int, TransBinCache] = field(default_factory=dict)


def build_trans_bin_cache(
    chrom_sizes: pd.DataFrame,
    bin_size: int,
    ns_index: NonspecificIndex,
    *,
    pseudo: float,
) -> TransBinCache:
    """Build trans bins once + precompute nonspecific background per trans bin once
    using Dirichlet-style pseudocounts.
    """
    bins_by_chr: Dict[str, pd.DataFrame] = {}
    pr_bins_by_chr: Dict[str, pr.PyRanges] = {}
    n_bins_by_chr: Dict[str, int] = {}

    alpha = float(pseudo)

    for _, row in chrom_sizes.iterrows():
        chrom = str(row["chr"])
        length = int(row["length"])

        df = uniform_bins(length, int(bin_size))
        df["dna_chr"] = chrom

        n_bins = int(len(df))
        n_bins_by_chr[chrom] = n_bins

        # --- nonspecific background ---
        pr_ns = ns_index.pr_by_chr.get(chrom)
        ns_total = float(ns_index.ns_total_by_chr.get(chrom, 0))

        if pr_ns is None or n_bins == 0:
            # no background data → pure prior
            ns_counts = np.zeros(n_bins, dtype=float)
            denom = alpha * n_bins
            bkg = np.full(n_bins, 1.0 / n_bins, dtype=float) if denom > 0 else np.zeros(n_bins, dtype=float)
        else:
            pr_bins = as_pyranges_intervals(
                df.rename(columns={"dna_chr": "chr"})[["chr", "start", "end", "bin_index"]],
                chrom_col="chr",
                start_col="start",
                end_col="end",
                extra_cols=["bin_index"],
            )

            ns_counts = overlaps_to_array(
                pr_bins, pr_ns, id_col="bin_index", n_bins=n_bins
            ).astype(float, copy=False)

            denom = ns_total + alpha * n_bins

            if denom > 0:
                # vectorized Dirichlet posterior mean
                bkg = (ns_counts + alpha) / denom
            else:
                # theoretically unreachable, but safe fallback
                bkg = np.full(n_bins, 1.0 / n_bins, dtype=float)

        df["ns_count"] = ns_counts
        df["ns_total"] = ns_total
        df["bkg"] = bkg

        # bins PyRanges (once)
        pr_bins = as_pyranges_intervals(
            df.rename(columns={"dna_chr": "chr"})[["chr", "start", "end", "bin_index"]],
            chrom_col="chr",
            start_col="start",
            end_col="end",
            extra_cols=["bin_index"],
        )

        bins_by_chr[chrom] = df[
            ["dna_chr", "bin_index", "start", "end", "center", "ns_count", "ns_total", "bkg"]
        ].copy()
        pr_bins_by_chr[chrom] = pr_bins

    return TransBinCache(
        bin_size=int(bin_size),
        bins_by_chr=bins_by_chr,
        pr_bins_by_chr=pr_bins_by_chr,
        n_bins_by_chr=n_bins_by_chr,
    )


def build_context(
    chrom_sizes: pd.DataFrame,
    ns_index: NonspecificIndex,
    *,
    trans_cache: Optional[TransBinCache] = None,
) -> PrecomputeContext:
    chrom_dict = dict(zip(chrom_sizes["chr"].astype(str), chrom_sizes["length"].astype(int)))
    chrom_names = chrom_sizes["chr"].astype(str).tolist()
    cache_by_size: Dict[int, TransBinCache] = {}
    if trans_cache is not None:
        cache_by_size[int(trans_cache.bin_size)] = trans_cache
    return PrecomputeContext(
        chrom_dict=chrom_dict,
        chrom_names=chrom_names,
        ns_index=ns_index,
        trans_cache=trans_cache,
        trans_cache_by_bin_size=cache_by_size,
    )


def get_or_build_trans_bin_cache(
    ctx: PrecomputeContext,
    chrom_sizes: pd.DataFrame,
    bin_size: int,
    *,
    pseudo: float,
) -> TransBinCache:
    key = int(bin_size)
    cached = ctx.trans_cache_by_bin_size.get(key)
    if cached is not None:
        return cached

    built = build_trans_bin_cache(chrom_sizes, key, ctx.ns_index, pseudo=pseudo)
    ctx.trans_cache_by_bin_size[key] = built
    if ctx.trans_cache is None:
        ctx.trans_cache = built
    return built
