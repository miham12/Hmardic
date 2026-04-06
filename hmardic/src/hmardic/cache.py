from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .binning import uniform_bins
from .nonspecific import NonspecificIndex
from .overlaps import as_pyranges_intervals, overlaps_to_array


@dataclass(frozen=True)
class UniformBinCache:
    """Uniform genome bins with precomputed global background per bin."""

    bin_size: int
    bins_by_chr: Dict[str, pd.DataFrame]
    n_bins_by_chr: Dict[str, int]


@dataclass
class PrecomputeContext:
    """Objects reused across many RNAs within one run/worker."""

    chrom_dict: Dict[str, int]
    ns_index: NonspecificIndex
    bin_cache: Optional[UniformBinCache] = None
    bin_cache_by_bin_size: Dict[int, UniformBinCache] = field(default_factory=dict)


def build_uniform_bin_cache(
    chrom_sizes: pd.DataFrame,
    bin_size: int,
    ns_index: NonspecificIndex,
    *,
    pseudo: float,
) -> UniformBinCache:
    bins_by_chr: Dict[str, pd.DataFrame] = {}
    n_bins_by_chr: Dict[str, int] = {}
    alpha = float(pseudo)

    for row in chrom_sizes.itertuples(index=False):
        chrom = str(row.chr)
        length = int(row.length)

        df = uniform_bins(length, int(bin_size))
        df["dna_chr"] = chrom

        n_bins = int(len(df))
        n_bins_by_chr[chrom] = n_bins

        pr_ns = ns_index.pr_by_chr.get(chrom)
        ns_total = float(ns_index.ns_total_by_chr.get(chrom, 0))

        if pr_ns is None or n_bins == 0:
            denom = alpha * n_bins
            bkg = np.full(n_bins, 1.0 / n_bins, dtype=np.float64) if denom > 0 else np.zeros(n_bins, dtype=np.float64)
        else:
            pr_bins = as_pyranges_intervals(
                df.rename(columns={"dna_chr": "chr"})[["chr", "start", "end", "bin_index"]],
                chrom_col="chr",
                start_col="start",
                end_col="end",
                extra_cols=["bin_index"],
            )
            ns_counts = overlaps_to_array(pr_bins, pr_ns, id_col="bin_index", n_bins=n_bins).astype(np.float64, copy=False)
            denom = ns_total + alpha * n_bins
            bkg = (ns_counts + alpha) / denom if denom > 0 else np.full(n_bins, 1.0 / n_bins, dtype=np.float64)

        df["bkg"] = bkg
        bins_by_chr[chrom] = df[["dna_chr", "bin_index", "start", "end", "center", "bkg"]].copy()

    return UniformBinCache(
        bin_size=int(bin_size),
        bins_by_chr=bins_by_chr,
        n_bins_by_chr=n_bins_by_chr,
    )


def build_context(
    chrom_sizes: pd.DataFrame,
    ns_index: NonspecificIndex,
    *,
    bin_cache: Optional[UniformBinCache] = None,
) -> PrecomputeContext:
    chrom_dict = dict(zip(chrom_sizes["chr"].astype(str), chrom_sizes["length"].astype(int)))
    cache_by_size: Dict[int, UniformBinCache] = {}
    if bin_cache is not None:
        cache_by_size[int(bin_cache.bin_size)] = bin_cache
    return PrecomputeContext(
        chrom_dict=chrom_dict,
        ns_index=ns_index,
        bin_cache=bin_cache,
        bin_cache_by_bin_size=cache_by_size,
    )


def get_or_build_uniform_bin_cache(
    ctx: PrecomputeContext,
    chrom_sizes: pd.DataFrame,
    bin_size: int,
    *,
    pseudo: float,
) -> UniformBinCache:
    key = int(bin_size)
    cached = ctx.bin_cache_by_bin_size.get(key)
    if cached is not None:
        return cached

    built = build_uniform_bin_cache(chrom_sizes, key, ctx.ns_index, pseudo=pseudo)
    ctx.bin_cache_by_bin_size[key] = built
    if ctx.bin_cache is None:
        ctx.bin_cache = built
    return built
