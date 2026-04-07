from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Mapping, Any

import numpy as np
import pandas as pd

from .config import HmardicParams
from .cache import (
    PrecomputeContext,
    UniformBinCache,
    build_uniform_bin_cache,
    get_or_build_uniform_bin_cache,
)
from .nonspecific import build_nonspecific_index
from .optimization import optimize_trans_bin_size


def _counts_uniform_bins_single_chr(
    starts: np.ndarray,
    ends: np.ndarray,
    bin_size: int,
    n_bins: int,
) -> np.ndarray:
    """Count overlaps of intervals [start, end) with uniform bins."""
    if n_bins <= 0:
        return np.zeros(0, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(n_bins, dtype=np.int64)

    s = starts.astype(np.int64, copy=False)
    e = ends.astype(np.int64, copy=False)

    valid = e > s
    if not np.any(valid):
        return np.zeros(n_bins, dtype=np.int64)
    s = s[valid]
    e = e[valid]

    i0 = np.clip(s // bin_size, 0, n_bins - 1)
    i1 = np.clip((e - 1) // bin_size, 0, n_bins - 1)

    diff = np.zeros(n_bins + 1, dtype=np.int64)
    np.add.at(diff, i0, 1)
    np.add.at(diff, i1 + 1, -1)
    return np.cumsum(diff[:-1], dtype=np.int64)


def _scaling_by_chr(
    rna_contacts: pd.DataFrame,
    n_cont: int,
    pseudo: float,
    chrom_names: List[str],
) -> Dict[str, float]:
    if not chrom_names:
        return {}

    alpha = float(pseudo)
    if rna_contacts is None or rna_contacts.empty or n_cont <= 0:
        vc = pd.Series(0.0, index=chrom_names, dtype=np.float64)
    else:
        vc = (
            rna_contacts["chr"]
            .value_counts(sort=False)
            .reindex(chrom_names, fill_value=0)
            .astype(np.float64, copy=False)
        )

    k = int(len(chrom_names))
    denom = float(n_cont) + alpha * k
    if denom <= 0:
        return {}
    return ((vc + alpha) / denom).to_dict()


def _cis_scaling(
    starts: np.ndarray,
    ends: np.ndarray,
    centers: np.ndarray,
    *,
    rna_start: int,
    rna_end: int,
    cis_share: float,
    decay: float,
    bin_size: int,
) -> np.ndarray:
    start_f = starts.astype(np.float64, copy=False)
    end_f = ends.astype(np.float64, copy=False)
    center_f = centers.astype(np.float64, copy=False)

    d = np.where(
        end_f <= rna_start,
        rna_start - center_f,
        np.where(start_f >= rna_end, center_f - rna_end, 0.0),
    )
    raw = np.exp(-d / (decay * float(bin_size)))
    total = float(raw.sum())
    if total <= 0.0:
        n_bins = raw.shape[0]
        if n_bins == 0:
            return raw
        return np.full(n_bins, cis_share / n_bins, dtype=np.float64)
    return cis_share * (raw / total)


def _bins_from_cache_for_rna(
    rna: str,
    rna_chr: str,
    rna_start: int,
    rna_end: int,
    bin_cache: UniformBinCache,
    sc_by_chr: Dict[str, float],
    *,
    decay: float,
    pseudo: float,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]]]:
    cols = ["dna_chr", "bin_index", "start", "end", "center", "bkg"]

    parts_arrays: Dict[str, List[np.ndarray]] = {c: [] for c in cols}
    rna_col: List[np.ndarray] = []
    sc_col: List[np.ndarray] = []
    slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    for chrom, base in bin_cache.bins_by_chr.items():
        n = int(bin_cache.n_bins_by_chr.get(chrom, 0))
        if n <= 0:
            continue

        for c in cols:
            parts_arrays[c].append(base[c].to_numpy(copy=False))
        rna_col.append(np.full(n, rna, dtype=object))

        if chrom == rna_chr:
            sc_arr = _cis_scaling(
                base["start"].to_numpy(np.int64, copy=False),
                base["end"].to_numpy(np.int64, copy=False),
                base["center"].to_numpy(np.int64, copy=False),
                rna_start=rna_start,
                rna_end=rna_end,
                cis_share=float(sc_by_chr.get(rna_chr, pseudo)),
                decay=decay,
                bin_size=bin_cache.bin_size,
            )
        else:
            chrom_share = float(sc_by_chr.get(chrom, pseudo))
            sc_arr = np.full(n, chrom_share / n, dtype=np.float64)

        sc_col.append(sc_arr.astype(np.float64, copy=False))
        slices[chrom] = (cursor, cursor + n)
        cursor += n

    if cursor == 0:
        return pd.DataFrame(columns=["rna", "dna_chr", "bin_index", "start", "end", "center", "sc", "bkg", "n_contacts"]), {}

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
    return pd.DataFrame(out), slices


def _combine_sc_and_bkg_by_chr(bins_df: pd.DataFrame, slices: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Combine chromosome mass from sc with local sc/background profiles inside each chromosome."""
    n = len(bins_df)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    out = np.zeros(n, dtype=np.float64)
    sc = bins_df["sc"].to_numpy(np.float64, copy=False)
    bkg = bins_df["bkg"].to_numpy(np.float64, copy=False)

    for lo, hi in slices.values():
        sc_chr = sc[lo:hi]
        bkg_chr = bkg[lo:hi]
        n_chr = hi - lo
        if n_chr <= 0:
            continue

        chr_mass = float(sc_chr.sum())
        if chr_mass <= 0.0:
            continue

        sc_local = sc_chr / chr_mass
        bkg_sum = float(bkg_chr.sum())
        bkg_local = (bkg_chr / bkg_sum) if bkg_sum > 0.0 else np.full(n_chr, 1.0 / n_chr, dtype=np.float64)

        combined = sc_local * bkg_local
        combined_sum = float(combined.sum())
        out[lo:hi] = (
            chr_mass * (combined / combined_sum)
            if combined_sum > 0.0
            else np.full(n_chr, chr_mass / n_chr, dtype=np.float64)
        )

    return out


def _fill_counts_from_cache(
    bins_df: pd.DataFrame,
    slices: Dict[str, Tuple[int, int]],
    rna_contacts: pd.DataFrame,
    bin_cache: UniformBinCache,
) -> pd.DataFrame:
    if bins_df.empty or rna_contacts is None or rna_contacts.empty:
        bins_df["n_contacts"] = 0.0
        return bins_df

    n_contacts = bins_df["n_contacts"].to_numpy(np.float64, copy=False)

    for chrom, g in rna_contacts.groupby("chr", observed=True):
        chrom = str(chrom)
        sl = slices.get(chrom)
        if sl is None:
            continue
        lo, hi = sl
        n_bins = int(bin_cache.n_bins_by_chr.get(chrom, 0))
        if n_bins <= 0:
            continue

        counts = _counts_uniform_bins_single_chr(
            g["start"].to_numpy(np.int64, copy=False),
            g["end"].to_numpy(np.int64, copy=False),
            bin_cache.bin_size,
            n_bins,
        )
        n_contacts[lo:hi] = counts

    return bins_df


def preprocess_one_rna(
    rna_row: Mapping[str, Any],
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

    if rna_contacts is None or rna_contacts.empty:
        rna_contacts = pd.DataFrame(columns=["chr", "start", "end"])
        n_cont = 0
    else:
        rna_contacts = rna_contacts.loc[:, ["chr", "start", "end"]].astype(
            {"chr": str, "start": np.int64, "end": np.int64},
            copy=False,
        )
        n_cont = int(len(rna_contacts))

    if ctx is not None:
        chrom_dict = ctx.chrom_dict
    else:
        chrom_dict = dict(zip(chrom_sizes["chr"].astype(str), chrom_sizes["length"].astype(int)))
    chrom_names = list(chrom_dict.keys())

    bin_size = params.bin_size
    if bin_size is None:
        trans_contacts = rna_contacts[rna_contacts["chr"] != rna_chr] if n_cont else pd.DataFrame(columns=["chr", "start", "end"])
        bin_size, _ = optimize_trans_bin_size(
            trans_contacts,
            chrom_dict,
            gene_chrom=rna_chr,
            start=params.trans_start,
            end=params.trans_end,
            step=params.trans_step,
            tolerance=params.tolerance,
            w=params.w,
        )
    bin_size = int(bin_size)

    sc_by_chr = _scaling_by_chr(rna_contacts, n_cont, params.pseudo, chrom_names)

    if ctx is not None:
        bin_cache = get_or_build_uniform_bin_cache(ctx, chrom_sizes, bin_size, pseudo=float(params.pseudo))
    else:
        empty_ns = build_nonspecific_index(pd.DataFrame(columns=["chr", "start", "end"]))
        bin_cache = build_uniform_bin_cache(chrom_sizes, bin_size, empty_ns, pseudo=float(params.pseudo))

    bins_df, slices = _bins_from_cache_for_rna(
        rna_name,
        rna_chr,
        rna_start,
        rna_end,
        bin_cache,
        sc_by_chr,
        decay=float(params.decay),
        pseudo=float(params.pseudo),
    )
    bins_df = _fill_counts_from_cache(bins_df, slices, rna_contacts, bin_cache)

    bins_df["f"] = _combine_sc_and_bkg_by_chr(bins_df, slices)
    total_f = float(bins_df["f"].sum())

    if total_f == 0.0:
        bins_df["lambda"] = float(params.min_lambda)
    else:
        bins_df["lambda"] = float(n_cont) * (bins_df["f"] / total_f)

    desired = ["rna", "dna_chr", "bin_index", "start", "end", "center", "sc", "bkg", "f", "lambda", "n_contacts"]
    return bins_df[desired]
