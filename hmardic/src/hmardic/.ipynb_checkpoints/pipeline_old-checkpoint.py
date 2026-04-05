
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any, List

import os
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import HmardicParams
from .nonspecific import build_nonspecific_index
from .cache import PrecomputeContext, build_trans_bin_cache, build_context
from .preprocess import preprocess_one_rna
from .hmm import PoissonHMM, precompute_log_fact
from .merge import merge_adjacent_peaks


# def _bh_fdr(p: np.ndarray) -> np.ndarray:
#     m = p.size
#     if m == 0:
#         return p
#     order = np.argsort(p)
#     ranked = p[order]
#     q = ranked * (m / (np.arange(1, m + 1, dtype=np.float64)))
#     q = np.minimum.accumulate(q[::-1])[::-1]
#     q = np.clip(q, 0.0, 1.0)
#     out = np.empty_like(q)
#     out[order] = q
#     return out


# def _poisson_sf(k: np.ndarray, mu: np.ndarray) -> np.ndarray:
#     k = k.astype(np.int64, copy=False)
#     mu = mu.astype(np.float64, copy=False)
#     try:
#         from scipy.stats import poisson as _sp_poisson  # type: ignore
#         return _sp_poisson.sf(k - 1, mu)
#     except Exception:
#         pass

#     # Normal approximation with continuity correction
#     out = np.empty_like(mu, dtype=np.float64)
#     m0 = (mu <= 0.0)
#     out[m0] = np.where(k[m0] <= 0, 1.0, 0.0)

#     m1 = ~m0
#     if np.any(m1):
#         mmu = mu[m1]
#         kk = k[m1].astype(np.float64, copy=False)
#         sd = np.sqrt(mmu)
#         z = (kk - 0.5 - mmu) / np.where(sd > 0, sd, 1.0)
#         out[m1] = 0.5 * math.erfc(1.0 / math.sqrt(2.0) * z)
#         out[m1] = np.clip(out[m1], 0.0, 1.0)
#     return out


# ──────────────────────────────────────────────────────────────────────────────
# HMM calling
# ──────────────────────────────────────────────────────────────────────────────

def call_states(bins_df: pd.DataFrame, params: HmardicParams, *, cis_chr: str) -> pd.DataFrame:
    """
    FULL version ONLY.
    cis_chr определяется однозначно из rna_annot['chr'] и передаётся сюда.

    Обучаем:
      - cis модель на bins cis_chr
      - trans модель на всех остальных chr как независимые последовательности (multi-seq BW)

    Декодируем:
      - cis chr → cis модель
      - trans chr → trans модель
    """
    if bins_df is None or bins_df.empty:
        return bins_df.assign(
            n_contacts_int=np.array([], dtype=np.int64),
            state=np.array([], dtype=np.int8),
            p_peak=np.array([], dtype=np.float64),
            hmm_p_bg=np.array([], dtype=np.float64),
            hmm_alpha=np.array([], dtype=np.float64),
        )

    # logging knobs
    bw_verbose = os.environ.get("HMARDIC_BW_VERBOSE", "0") not in ("0", "", "false", "False")
    bw_profile = os.environ.get("HMARDIC_BW_PROFILE", "0") not in ("0", "", "false", "False")
    bw_log_every = int(os.environ.get("HMARDIC_BW_LOG_EVERY", "1"))

    # сортировка обязательна для правильной последовательности внутри chr
    sdf = bins_df.sort_values(["dna_chr", "start", "end"], kind="mergesort")
    rna_id = str(sdf["rna"].iloc[0]) if "rna" in sdf.columns and len(sdf) else "rna"

    chrom_groups: Dict[str, pd.DataFrame] = {}
    chrom_counts: Dict[str, np.ndarray] = {}
    chrom_lams: Dict[str, np.ndarray] = {}

    cis_counts: List[np.ndarray] = []
    cis_lams: List[np.ndarray] = []
    trans_counts: List[np.ndarray] = []
    trans_lams: List[np.ndarray] = []

    for chrom, g in sdf.groupby("dna_chr", observed=True, sort=False):
        ch = str(chrom)
        chrom_groups[ch] = g

        # pipeline guarantees dtype here; keep this conversion as the single source of truth
        counts_raw = g["n_contacts"].to_numpy(copy=False)
        # Fast path: if already integer dtype, avoid float conversion + floor
        if np.issubdtype(counts_raw.dtype, np.integer):
            counts_i = counts_raw.astype(np.int64, copy=False)
        else:
            counts_i = np.floor(counts_raw.astype(np.float64, copy=False)).astype(np.int64, copy=False)
            counts_i[counts_i < 0] = 0
        lams = g["lambda"].to_numpy(dtype=np.float64, copy=False)


        chrom_counts[ch] = counts_i
        chrom_lams[ch] = lams

        if counts_i.size == 0:
            continue

        if ch == cis_chr:
            cis_counts.append(counts_i)
            cis_lams.append(lams)
        else:
            trans_counts.append(counts_i)
            trans_lams.append(lams)

    model_cis: Optional[PoissonHMM] = None
    if cis_counts:
        model_cis = PoissonHMM.fit_multi(
            cis_counts,
            cis_lams,
            p_bg=params.p_bg,
            alpha=params.alpha,
            max_iter=params.max_iter,
            tol=params.tol,
            verbose=bw_verbose,
            log_every=bw_log_every,
            prefix=f"{rna_id}:cis",
            profile_time=bw_profile,
        )

    model_trans: Optional[PoissonHMM] = None
    if trans_counts:
        model_trans = PoissonHMM.fit_multi(
            trans_counts,
            trans_lams,
            p_bg=params.p_bg,
            alpha=params.alpha,
            max_iter=params.max_iter,
            tol=params.tol,
            verbose=bw_verbose,
            log_every=bw_log_every,
            prefix=f"{rna_id}:trans",
            profile_time=bw_profile,

        )

    if model_cis is None and model_trans is None:
        n = len(sdf)
        return sdf.assign(
            n_contacts_int=np.zeros(n, dtype=np.int64),
            state=np.zeros(n, dtype=np.int8),
            p_peak=np.zeros(n, dtype=np.float64),
            hmm_p_bg=np.full(n, np.nan, dtype=np.float64),
            hmm_alpha=np.full(n, np.nan, dtype=np.float64),
        ).reset_index(drop=True)

    def _make_decoder(model: PoissonHMM) -> PoissonHMM:
        return PoissonHMM(
            counts=np.array([0], dtype=np.int64),
            lambdas=np.array([1.0], dtype=np.float64),
            p_bg=model.p_bg,
            alpha=model.alpha,
            A=model.A,
            pi=model.pi,
            max_iter=0,
            tol=model.tol,
        )

    dec_cis = _make_decoder(model_cis) if model_cis is not None else None
    dec_trans = _make_decoder(model_trans) if model_trans is not None else None

    out_parts: List[pd.DataFrame] = []

    for ch, g in chrom_groups.items():
        counts_i = chrom_counts[ch]
        lams = chrom_lams[ch]

        if ch == cis_chr and dec_cis is not None:
            dec = dec_cis
            model = model_cis
        else:
            dec = dec_trans if dec_trans is not None else dec_cis
            model = model_trans if model_trans is not None else model_cis

        if dec is None or model is None:
            states = np.zeros(counts_i.size, dtype=np.int8)
            p_peak = np.zeros(counts_i.size, dtype=np.float64)
            p_bg = float("nan")
            alpha = float("nan")
        else:
            dec.counts_i = counts_i
            dec.lambdas = lams
            dec.T = int(counts_i.size)

            states, p_peak = dec.decode_peak()
            p_bg = float(model.p_bg)
            alpha = float(model.alpha)

        out_parts.append(
            g.assign(
                n_contacts_int=counts_i,
                state=states.astype(np.int8, copy=False),
                p_peak=p_peak,
                hmm_p_bg=p_bg,
                hmm_alpha=alpha,
            )
        )

    return pd.concat(out_parts, ignore_index=True)


def call_peaks(bins_with_states: pd.DataFrame) -> pd.DataFrame:
    if bins_with_states is None or bins_with_states.empty:
        return pd.DataFrame()

    needed = {
        "rna", "dna_chr", "start", "end", "state",
        "lambda", "n_contacts_int", "p_peak", "hmm_p_bg", "hmm_alpha"
    }
    miss = needed - set(bins_with_states.columns)
    if miss:
        raise ValueError(f"call_peaks: missing columns: {sorted(miss)}")

    rna0 = bins_with_states["rna"].iat[0]
    if (bins_with_states["rna"] != rna0).any():
        raise ValueError("call_peaks: expected bins for a SINGLE RNA, but multiple RNAs found")

    sdf = bins_with_states.sort_values(["dna_chr", "start", "end"], kind="mergesort")

    out_chr: List[Any] = []
    out_start: List[int] = []
    out_end: List[int] = []
    out_nbins: List[int] = []
    out_k: List[int] = []
    out_lamsum: List[float] = []
    out_mubg: List[float] = []
    out_fold: List[float] = []
    out_meanppk: List[float] = []
    out_maxppk: List[float] = []
    out_pbg: List[float] = []
    out_alpha: List[float] = []

    for chrom, g in sdf.groupby("dna_chr", observed=True, sort=False):
        st = g["state"].to_numpy(np.int8, copy=False)
        if st.size == 0:
            continue

        mask = (st == 1)
        if not mask.any():
            continue

        starts = g["start"].to_numpy(np.int64, copy=False)
        ends   = g["end"].to_numpy(np.int64, copy=False)
        cnt    = g["n_contacts_int"].to_numpy(np.int64, copy=False)
        lam    = g["lambda"].to_numpy(np.float64, copy=False)
        ppk    = g["p_peak"].to_numpy(np.float64, copy=False)

        p_bg = float(g["hmm_p_bg"].iat[0])
        alpha = float(g["hmm_alpha"].iat[0])

        prev = np.empty_like(mask)
        prev[0] = False
        prev[1:] = mask[:-1]
        nxt = np.empty_like(mask)
        nxt[-1] = False
        nxt[:-1] = mask[1:]

        run_starts = np.flatnonzero(mask & ~prev)
        run_ends   = np.flatnonzero(mask & ~nxt)

        c_cnt = np.cumsum(cnt, dtype=np.int64)
        c_lam = np.cumsum(lam, dtype=np.float64)
        c_ppk = np.cumsum(ppk, dtype=np.float64)

        for rs, re in zip(run_starts, run_ends):
            nbins = int(re - rs + 1)

            if rs > 0:
                k_obs   = int(c_cnt[re] - c_cnt[rs - 1])
                lam_sum = float(c_lam[re] - c_lam[rs - 1])
                sum_ppk = float(c_ppk[re] - c_ppk[rs - 1])
            else:
                k_obs   = int(c_cnt[re])
                lam_sum = float(c_lam[re])
                sum_ppk = float(c_ppk[re])

            mean_ppk = sum_ppk / nbins
            max_ppk  = float(ppk[rs:re+1].max())

            mu_bg = p_bg * lam_sum
            fold = (k_obs / mu_bg) if (mu_bg > 0.0) else (float("inf") if k_obs > 0 else 1.0)

            out_chr.append(chrom)
            out_start.append(int(starts[rs]))
            out_end.append(int(ends[re]))
            out_nbins.append(nbins)
            out_k.append(k_obs)
            out_lamsum.append(lam_sum)
            out_mubg.append(float(mu_bg))
            out_fold.append(float(fold))
            out_meanppk.append(float(mean_ppk))
            out_maxppk.append(float(max_ppk))
            out_pbg.append(float(p_bg))
            out_alpha.append(float(alpha))

    if not out_k:
        return pd.DataFrame()

    peaks = pd.DataFrame({
        "rna": rna0,
        "dna_chr": out_chr,
        "start": out_start,
        "end": out_end,
        "n_bins": out_nbins,
        "k_obs": out_k,
        "lambda_sum": out_lamsum,
        "mu_bg": out_mubg,
        "fold_enrichment": out_fold,
        "mean_p_peak": out_meanppk,
        "max_p_peak": out_maxppk,
        "hmm_p_bg": out_pbg,
        "hmm_alpha": out_alpha,
    })

    cols = [
        "rna","dna_chr","start","end","n_bins",
        "k_obs","mu_bg","lambda_sum","fold_enrichment",
        "mean_p_peak","max_p_peak",
        "hmm_p_bg","hmm_alpha",
    ]
    return peaks[cols].sort_values(["dna_chr","start","end"], kind="mergesort").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Worker context for multiprocessing
# ──────────────────────────────────────────────────────────────────────────────

_CTX: Optional[PrecomputeContext] = None
_PARAMS: Optional[HmardicParams] = None
_CHROM_SIZES: Optional[pd.DataFrame] = None


def _init_worker(chrom_sizes: pd.DataFrame, nonspecific_contacts: pd.DataFrame, params_dict: Dict[str, Any]):
    global _CTX, _PARAMS, _CHROM_SIZES
    _CHROM_SIZES = chrom_sizes
    _PARAMS = HmardicParams(**params_dict)

    # One-time precompute in each worker process
    precompute_log_fact(10000)

    ns_index = build_nonspecific_index(nonspecific_contacts)
    trans_cache = None
    if _PARAMS.fixed_bins:
        trans_cache = build_trans_bin_cache(chrom_sizes, _PARAMS.bin_size, ns_index, pseudo=_PARAMS.pseudo)
    _CTX = build_context(chrom_sizes, ns_index, trans_cache=trans_cache)


def _process_one_rna_worker(
    rna_row_dict: Dict[str, Any],
    cis_chr: str,
    rna_contacts: pd.DataFrame
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:

    global _CTX, _PARAMS, _CHROM_SIZES
    assert _CTX is not None and _PARAMS is not None and _CHROM_SIZES is not None

    rna_row = pd.Series(rna_row_dict)
    bins_df = preprocess_one_rna(rna_row, rna_contacts, _CHROM_SIZES, _PARAMS, ctx=_CTX)

    bins_states = call_states(bins_df, _PARAMS, cis_chr=cis_chr)
    peaks = call_peaks(bins_states)
    if _PARAMS.merge_peaks and not peaks.empty:
        peaks = merge_adjacent_peaks(peaks, state_col="state", peak_state=1)

    return str(rna_row_dict["rna"]), bins_states, peaks


def run_calling(
    chrom_sizes: pd.DataFrame,
    contacts: pd.DataFrame,
    rna_annot: pd.DataFrame,
    nonspecific_contacts: Optional[pd.DataFrame],
    params: HmardicParams,
    *,
    threads: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    nonspecific_contacts = nonspecific_contacts if nonspecific_contacts is not None else pd.DataFrame(columns=["chr","start","end"])

    # One-time precompute (main process)
    precompute_log_fact(1000)

    # ───── 1) contacts_by_rna ─────
    contacts_by_rna: Dict[str, pd.DataFrame] = {}
    if contacts is not None and not contacts.empty:
        for rna, g in contacts.groupby("rna", observed=True):
            contacts_by_rna[str(rna)] = g[["chr", "start", "end"]]

    # ───── 2) tasks ─────
    empty_cont = pd.DataFrame(columns=["chr", "start", "end"])
    tasks = []
    for _, row in rna_annot.iterrows():
        rna = str(row["rna"])
        cis_chr = str(row["chr"])
        tasks.append((row.to_dict(), cis_chr, contacts_by_rna.get(rna, empty_cont)))

    all_bins: List[pd.DataFrame] = []
    all_peaks: List[pd.DataFrame] = []

    if threads <= 1:
        ns_index = build_nonspecific_index(nonspecific_contacts)
        trans_cache = build_trans_bin_cache(chrom_sizes, params.bin_size, ns_index, pseudo=params.pseudo) if params.fixed_bins else None
        ctx = build_context(chrom_sizes, ns_index, trans_cache=trans_cache)

        for row_dict, cis_chr, rna_cont in tasks:
            rna_row = pd.Series(row_dict)
            bins_df = preprocess_one_rna(rna_row, rna_cont, chrom_sizes, params, ctx=ctx)

            bins_states = call_states(bins_df, params, cis_chr=cis_chr)
            if params.merge_peaks and not bins_states.empty:
                peaks = call_peaks(bins_states)

            all_bins.append(bins_states)
            all_peaks.append(peaks)

    else:
        params_dict = asdict(params)
        with ProcessPoolExecutor(
            max_workers=threads,
            initializer=_init_worker,
            initargs=(chrom_sizes, nonspecific_contacts, params_dict),
        ) as ex:
            futs = [ex.submit(_process_one_rna_worker, row_dict, cis_chr, rna_cont) for row_dict, cis_chr, rna_cont in tasks]
            for fut in as_completed(futs):
                _, bins_states, peaks = fut.result()
                all_bins.append(bins_states)
                all_peaks.append(peaks)

    bins_all = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()
    peaks_all = pd.concat(all_peaks, ignore_index=True) if all_peaks else pd.DataFrame()
    return bins_all, peaks_all
