from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
import pyranges as pr


def as_pyranges_intervals(
    df: pd.DataFrame,
    *,
    chrom_col: str = "chr",
    start_col: str = "start",
    end_col: str = "end",
    extra_cols: Optional[Sequence[str]] = None,
) -> pr.PyRanges:
    """Convert a (chr,start,end,...) dataframe into a PyRanges.

    extra_cols will be preserved (e.g. bin_index).
    """
    cols = [chrom_col, start_col, end_col]
    if extra_cols:
        cols += list(extra_cols)
    tmp = df[cols].copy()
    tmp = tmp.rename(columns={chrom_col: "Chromosome", start_col: "Start", end_col: "End"})
    # Ensure ints (PyRanges expects ints)
    tmp["Start"] = tmp["Start"].astype(int)
    tmp["End"] = tmp["End"].astype(int)
    tmp["Chromosome"] = tmp["Chromosome"].astype(str)
    return pr.PyRanges(tmp)


def overlaps_to_array(
    bins_pr: pr.PyRanges,
    query_pr: pr.PyRanges,
    *,
    id_col: str,
    out_col: str = "NumberOverlaps",
    n_bins: int,
) -> np.ndarray:
    """Count overlaps and return a dense numpy array of length n_bins indexed by id_col."""
    if n_bins <= 0:
        return np.zeros(0, dtype=float)
    if query_pr is None or len(query_pr) == 0:
        return np.zeros(n_bins, dtype=float)

    df = bins_pr.count_overlaps(query_pr).df
    if df.empty:
        return np.zeros(n_bins, dtype=float)

    idx = df[id_col].to_numpy(dtype=int, copy=False)
    val = df[out_col].to_numpy(dtype=float, copy=False)

    out = np.zeros(n_bins, dtype=float)
    # id_col is expected to be 0..n_bins-1; if not, caller should map beforehand.
    out[idx] = val
    return out
