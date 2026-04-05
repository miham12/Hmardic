# preprocess.py (DROP-IN replacement)
from __future__ import annotations

from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import HmardicParams
from .cache import PrecomputeContext, TransBinCache
from .nonspecific import NonspecificIndex
from .binning import uniform_bins
from .cis import preprocess_bins_for_rna
from .optimization import optimize_trans_bin_size, optimize_cis_factor

# PyRanges оставляем только для CIS background (там bins не uniform и их ~20k, это терпимо)
import pyranges as pr
from .overlaps import as_pyranges_intervals, overlaps_to_array


# ──────────────────────────────────────────────────────────────────────────────
# FAST uniform-bin overlaps (no PyRanges)
# ──────────────────────────────────────────────────────────────────────────────

def _counts_uniform_bins_single_chr(
    starts: np.ndarray,
    ends: np.ndarray,
    bin_size: int,
    n_bins: int,
) -> np.ndarray:
    """
    Count overlaps of intervals [start,end) with uniform bins of size bin_size.
    Returns counts array of length n_bins where bin i is [i*bin_size, (i+1)*bin_size).

    Complexity: O(Ncontacts + n_bins).
    """
    if n_bins <= 0:
        return np.zeros(0, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(n_bins, dtype=np.int64)

    s = starts.astype(np.int64, copy=False)
    e = ends.astype(np.int64, copy=False)

    # filter invalid
    m = e > s
    if not np.any(m):
        return np.zeros(n_bins, dtype=np.int64)
    s = s[m]
    e = e[m]

    i0 = s // bin_size
    i1 = (e - 1) // bin_size

    # clip into [0, n_bins-1]
    i0 = np.clip(i0, 0, n_bins - 1)
    i1 = np.clip(i1, 0, n_bins - 1)

    diff = np.zeros(n_bins + 1, dtype=np.int64)
    # add ranges via diff-array
    np.add.at(diff, i0, 1)
    np.add.at(diff, i1 + 1, -1)

    out = np.cumsum(diff[:-1], dtype=np.int64)  # int64 counts
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scaling_by_chr(
    rna_contacts: pd.DataFrame,
    n_cont: int,
    pseudo: float,
) -> Dict[str, float]:
    if rna_contacts is None or rna_contacts.empty or n_cont <= 0:
        return {}
    alpha = float(pseudo)

    vc = rna_contacts["chr"].value_counts()  # chr already str in preprocess_one_rna
    k = int(len(vc))
    denom = float(n_cont) + alpha * k
    if denom <= 0:
        return {}
    scaled = (vc + alpha) / denom
    return scaled.to_dict()


def _build_cis_bins_df(
    rna: str,
    rna_chr: str,
    rna_start: int,
    rna_end: int,
    chrom_length: int,
    bin_size: int,
    cis_start_size: int,
    factor: float,
    decay: float,
    pseudo: float,
    cis_share: float,
) -> pd.DataFrame:
    ups, _, dns = preprocess_bins_for_rna(
        rna_start, rna_end, chrom_length, bin_size, cis_start_size, factor
    )
    bins = pd.DataFrame(ups + dns)
    if bins.empty:
        return pd.DataFrame(columns=["rna","dna_chr","bin_index","start","end","center","sc","bkg","n_contacts"])

    bins["dna_chr"] = rna_chr
    bins["bin_index"] = np.arange(len(bins), dtype=np.int32)
    bins["center"] = ((bins["start"].to_numpy(np.int64) + bins["end"].to_numpy(np.int64)) // 2).astype(np.int64)
    bins["rna"] = rna

    center = bins["center"].to_numpy(dtype=np.float64, copy=False)
    start = bins["start"].to_numpy(dtype=np.float64, copy=False)
    end = bins["end"].to_numpy(dtype=np.float64, copy=False)

    d = np.where(
        end <= rna_start,
        rna_start - center,
        np.where(start >= rna_end, center - rna_end, 0.0),
    )
    sc = cis_share * np.exp(-d / (decay * float(bin_size)))
    bins["sc"] = sc.astype(np.float64)
    bins["bkg"] = float(pseudo)
    bins["n_contacts"] = 0.0

    return bins[["rna","dna_chr","bin_index","start","end","center","sc","bkg","n_contacts"]]


def _cis_background(
    cis_df: pd.DataFrame,
    ns_index: NonspecificIndex,
    *,
    pseudo: float,
) -> pd.DataFrame:
    """
    CIS background оставляем через PyRanges: bins не uniform (геометрия),
    зато их всего ~20k и это не основной bottleneck.
    """
    if cis_df.empty:
        return cis_df

    chrom = str(cis_df["dna_chr"].iloc[0])
    pr_ns = ns_index.pr_by_chr.get(chrom)
    ns_total = float(ns_index.ns_total_by_chr.get(chrom, 0))
    n_bins = int(len(cis_df))
    alpha = float(pseudo)

    if pr_ns is None or ns_total <= 0 or n_bins <= 0:
        cis_df["bkg"] = np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
        return cis_df

    pr_bins = as_pyranges_intervals(
        cis_df.rename(columns={"dna_chr": "chr"})[["chr", "start", "end", "bin_index"]],
        chrom_col="chr",
        start_col="start",
        end_col="end",
        extra_cols=["bin_index"],
    )
    ns_counts = overlaps_to_array(pr_bins, pr_ns, id_col="bin_index", n_bins=n_bins).astype(np.float64, copy=False)

    denom = ns_total + alpha * n_bins
    cis_df["bkg"] = (ns_counts + alpha) / denom if denom > 0 else np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
    return cis_df


def _add_contacts_counts_cis(cis_df: pd.DataFrame, rna_contacts: pd.DataFrame) -> pd.DataFrame:
    if cis_df.empty:
        cis_df["n_contacts"] = np.zeros(0, dtype=np.float64)
        return cis_df

    chrom = str(cis_df["dna_chr"].iloc[0])
    if rna_contacts is None or rna_contacts.empty:
        cis_df["n_contacts"] = 0.0
        return cis_df

    cont = rna_contacts[rna_contacts["chr"] == chrom][["chr","start","end"]]
    if cont.empty:
        cis_df["n_contacts"] = 0.0
        return cis_df

    pr_bins = as_pyranges_intervals(
        cis_df.rename(columns={"dna_chr":"chr"})[["chr","start","end","bin_index"]],
        chrom_col="chr", start_col="start", end_col="end", extra_cols=["bin_index"]
    )
    pr_cont = as_pyranges_intervals(cont, chrom_col="chr", start_col="start", end_col="end")
    counts = overlaps_to_array(pr_bins, pr_cont, id_col="bin_index", n_bins=len(cis_df))
    cis_df["n_contacts"] = counts.astype(np.float64, copy=False)
    return cis_df


# ──────────────────────────────────────────────────────────────────────────────
# TRANS building from cache WITHOUT dataframe copying
# ──────────────────────────────────────────────────────────────────────────────

def _trans_from_cache_fast(
    rna: str,
    rna_chr: str,
    trans_cache: TransBinCache,
    sc_by_chr: Dict[str, float],
    pseudo: float,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int,int]]]:
    """
    Build one big trans_df via numpy concatenation, no per-chr .copy().assign().
    Also return slices {chrom: (lo, hi)} so we can fill counts quickly if needed.
    """
    cols = ["dna_chr","bin_index","start","end","center","bkg"]

    parts_arrays: Dict[str, List[np.ndarray]] = {c: [] for c in cols}
    rna_col: List[np.ndarray] = []
    sc_col: List[np.ndarray] = []

    slices: Dict[str, Tuple[int,int]] = {}
    cursor = 0

    for chrom, base in trans_cache.bins_by_chr.items():
        if chrom == rna_chr:
            continue

        n = int(trans_cache.n_bins_by_chr.get(chrom, 0))
        if n <= 0:
            continue

        # base columns (try to avoid copies)
        for c in cols:
            parts_arrays[c].append(base[c].to_numpy(copy=False))
        rna_col.append(np.full(n, rna, dtype=object))
        sc_col.append(np.full(n, float(sc_by_chr.get(chrom, pseudo)), dtype=np.float64))

        slices[chrom] = (cursor, cursor + n)
        cursor += n

    if cursor == 0:
        df = pd.DataFrame(columns=["rna","dna_chr","bin_index","start","end","center","sc","bkg","n_contacts"])
        return df, {}

    out = {
        "rna": np.concatenate(rna_col, axis=0),
        "dna_chr": np.concatenate(parts_arrays["dna_chr"], axis=0),
        "bin_index": np.concatenate(parts_arrays["bin_index"], axis=0).astype(np.int32, copy=False),
        "start": np.concatenate(parts_arrays["start"], axis=0).astype(np.int64, copy=False),
        "end": np.concatenate(parts_arrays["end"], axis=0).astype(np.int64, copy=False),
        "center": np.concatenate(parts_arrays["center"], axis=0).astype(np.int64, copy=False),
        "sc": np.concatenate(sc_col, axis=0).astype(np.float64, copy=False),
        "bkg": np.concatenate(parts_arrays["bkg"], axis=0).astype(np.float64, copy=False),
        "n_contacts": np.zeros(cursor, dtype=np.float64),
    }
    df = pd.DataFrame(out)
    return df, slices


def _fill_trans_counts_fast(
    trans_df: pd.DataFrame,
    slices: Dict[str, Tuple[int,int]],
    rna_contacts: pd.DataFrame,
    trans_cache: TransBinCache,
) -> pd.DataFrame:
    if trans_df.empty or rna_contacts is None or rna_contacts.empty:
        trans_df["n_contacts"] = 0.0
        return trans_df

    n_contacts = trans_df["n_contacts"].to_numpy(np.float64, copy=False)

    # groupby chr, then uniform-bin counting via diff-array
    for chrom, g in rna_contacts.groupby("chr", observed=True):
        chrom = str(chrom)
        sl = slices.get(chrom)
        if sl is None:
            continue
        lo, hi = sl
        n_bins = int(trans_cache.n_bins_by_chr.get(chrom, 0))
        if n_bins <= 0:
            continue

        starts = g["start"].to_numpy(np.int64, copy=False)
        ends = g["end"].to_numpy(np.int64, copy=False)

        counts = _counts_uniform_bins_single_chr(starts, ends, trans_cache.bin_size, n_bins)
        # trans_df rows for this chr correspond to bin_index 0..n_bins-1 in order
        # because cache uniform_bins produces that order.
        n_contacts[lo:hi] = counts

    return trans_df


# ──────────────────────────────────────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_one_rna(
    rna_row: pd.Series,
    rna_contacts: pd.DataFrame,
    chrom_sizes: pd.DataFrame,
    params: HmardicParams,
    *,
    ctx: Optional[PrecomputeContext] = None,
) -> pd.DataFrame:
    rna_name = str(rna_row["rna"])
    rna_chr = str(rna_row["chr"])
    rna_start = int(rna_row["start"])
    rna_end = int(rna_row["end"])

    # normalize contacts once
    if rna_contacts is None or rna_contacts.empty:
        rna_contacts = pd.DataFrame(columns=["chr","start","end"])
        n_cont = 0
    else:
        rna_contacts = rna_contacts[["chr","start","end"]].copy()
        rna_contacts["chr"] = rna_contacts["chr"].astype(str)
        rna_contacts["start"] = rna_contacts["start"].astype(np.int64)
        rna_contacts["end"] = rna_contacts["end"].astype(np.int64)
        n_cont = int(len(rna_contacts))

    # precomputed chromosome lengths/order + nonspecific index
    if ctx is not None:
        chrom_dict = ctx.chrom_dict
        ns_index = ctx.ns_index
    else:
        chrom_dict = dict(zip(chrom_sizes["chr"].astype(str), chrom_sizes["length"].astype(int)))
        from .nonspecific import build_nonspecific_index
        ns_index = build_nonspecific_index(pd.DataFrame(columns=["chr","start","end"]))

    # resolve bin_size/factor (avoid optimization if fixed)
    bin_size = params.bin_size
    factor = params.factor

    if bin_size is None:
        trans_contacts = rna_contacts[rna_contacts["chr"] != rna_chr] if n_cont else pd.DataFrame(columns=["chr","start","end"])
        bin_size, _ = optimize_trans_bin_size(
            trans_contacts, chrom_dict, gene_chrom=rna_chr,
            start=params.trans_start, end=params.trans_end, step=params.trans_step,
            tolerance=params.tolerance, w=params.w,
        )

    if factor is None:
        cis_contacts = rna_contacts[rna_contacts["chr"] == rna_chr] if n_cont else pd.DataFrame(columns=["chr","start","end"])
        factor, _ = optimize_cis_factor(
            cis_contacts, chrom_dict,
            gene_chrom=rna_chr, gene_start=rna_start, gene_end=rna_end,
            cis_start=params.cis_start_size,
            factor_min=params.cis_factor_min, factor_max=params.cis_factor_max, factor_step=params.cis_factor_step,
            tolerance=params.tolerance, w=params.w, max_linear_size=int(bin_size),
        )

    bin_size = int(bin_size)
    factor = float(factor)

    # per-chr scaling
    sc_by_chr = _scaling_by_chr(rna_contacts, n_cont, params.pseudo)
    cis_share = float(sc_by_chr.get(rna_chr, params.pseudo))

    # CIS bins
    chrom_length = int(chrom_dict[rna_chr])
    cis_df = _build_cis_bins_df(
        rna_name, rna_chr, rna_start, rna_end,
        chrom_length, bin_size, int(params.cis_start_size), factor,
        float(params.decay), float(params.pseudo), cis_share,
    )
    cis_df = _cis_background(cis_df, ns_index, pseudo=float(params.pseudo))
    cis_df = _add_contacts_counts_cis(cis_df, rna_contacts)

    # TRANS bins (fast path: cache)
    trans_df = pd.DataFrame(columns=["rna","dna_chr","bin_index","start","end","center","sc","bkg","n_contacts"])
    if ctx is not None and ctx.trans_cache is not None and ctx.trans_cache.bin_size == bin_size:
        trans_cache = ctx.trans_cache
        trans_df, slices = _trans_from_cache_fast(
            rna_name, rna_chr, trans_cache, sc_by_chr, pseudo=float(params.pseudo)
        )
        trans_df = _fill_trans_counts_fast(trans_df, slices, rna_contacts, trans_cache)
    else:
        # fallback: build uniform bins on the fly, but still count overlaps via diff-array
        parts = []
        for chrom, length in chrom_dict.items():
            if chrom == rna_chr:
                continue
            df = uniform_bins(int(length), bin_size)
            n_bins = len(df)
            if n_bins == 0:
                continue
            df["dna_chr"] = chrom
            df["rna"] = rna_name
            df["sc"] = float(sc_by_chr.get(chrom, params.pseudo))
            # background: если нет cache, то по-хорошему тоже кешировать,
            # но оставим совместимо (можно позже утащить в ctx)
            pr_ns = ns_index.pr_by_chr.get(chrom)
            ns_total = float(ns_index.ns_total_by_chr.get(chrom, 0))
            if pr_ns is None or ns_total <= 0:
                df["bkg"] = 1.0 / n_bins
            else:
                pr_bins = as_pyranges_intervals(
                    df.rename(columns={"dna_chr":"chr"})[["chr","start","end","bin_index"]],
                    chrom_col="chr", start_col="start", end_col="end", extra_cols=["bin_index"]
                )
                ns_counts = overlaps_to_array(pr_bins, pr_ns, id_col="bin_index", n_bins=n_bins).astype(np.float64, copy=False)
                df["bkg"] = (ns_counts + float(params.pseudo)) / (ns_total + float(params.pseudo) * n_bins)
            df["n_contacts"] = 0.0

            # fill trans counts fast for this chrom
            g = rna_contacts[rna_contacts["chr"] == chrom]
            if not g.empty:
                counts = _counts_uniform_bins_single_chr(
                    g["start"].to_numpy(np.int64, copy=False),
                    g["end"].to_numpy(np.int64, copy=False),
                    bin_size,
                    n_bins,
                )
                df["n_contacts"] = counts
            parts.append(df[["rna","dna_chr","bin_index","start","end","center","sc","bkg","n_contacts"]])

        if parts:
            trans_df = pd.concat(parts, ignore_index=True)

    bins_df = pd.concat([cis_df, trans_df], ignore_index=True)

    # f, lambda, pseudocount
    sc = bins_df["sc"].to_numpy(np.float64, copy=False)
    bkg = bins_df["bkg"].to_numpy(np.float64, copy=False)
    bins_df["f"] = sc * bkg
    total_f = float(bins_df["f"].sum())

    if total_f == 0.0:
        bins_df["lambda"] = float(params.min_lambda)
    else:
        #bins_df["lambda"] = (float(n_cont) / total_f) * bins_df["f"] + float(params.min_lambda)
        bins_df["lambda"] = (float(n_cont) / total_f) * bins_df["f"]

    desired = ["rna","dna_chr","bin_index","start","end","center","sc","bkg","f","lambda","n_contacts"]
    return bins_df[desired]