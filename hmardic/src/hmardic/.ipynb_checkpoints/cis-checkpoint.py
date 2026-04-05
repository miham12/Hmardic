from __future__ import annotations

from typing import List, Dict, Tuple
import numpy as np
from numba import jit

@jit(nopython=True)
def geometric_upstream_bins_numba(gene_start: int, trans_bin_size: int, cis_start_size: int, factor: float):
    bins = []
    current_end = gene_start
    bin_size = cis_start_size

    while current_end > 0:
        if bin_size > trans_bin_size:
            bin_size = trans_bin_size
        current_start = current_end - bin_size
        if current_start < 0:
            current_start = 0
        center = (current_start + current_end) // 2
        bins.append((current_start, current_end, center))
        current_end = current_start
        if bin_size < trans_bin_size:
            bin_size = int(bin_size * factor)
            if bin_size > trans_bin_size:
                bin_size = trans_bin_size

    bins.reverse()
    return bins

@jit(nopython=True)
def geometric_downstream_bins_numba(gene_end: int, chrom_length: int, trans_bin_size: int, cis_start_size: int, factor: float):
    bins = []
    current_start = gene_end
    bin_size = cis_start_size

    while current_start < chrom_length:
        if bin_size > trans_bin_size:
            bin_size = trans_bin_size
        current_end = current_start + bin_size
        if current_end > chrom_length:
            current_end = chrom_length
        center = (current_start + current_end) // 2
        bins.append((current_start, current_end, center))
        current_start = current_end
        if bin_size < trans_bin_size:
            bin_size = int(bin_size * factor)
            if bin_size > trans_bin_size:
                bin_size = trans_bin_size
    return bins

def geometric_upstream_bins_full(gene_start: int, trans_bin_size: int, cis_start_size: int, factor: float) -> List[Dict]:
    tup = geometric_upstream_bins_numba(gene_start, trans_bin_size, cis_start_size, factor)
    return [{"start": int(a), "end": int(b), "center": int(c)} for a, b, c in tup]

def geometric_downstream_bins_full(gene_end: int, chrom_length: int, trans_bin_size: int, cis_start_size: int, factor: float) -> List[Dict]:
    tup = geometric_downstream_bins_numba(gene_end, chrom_length, trans_bin_size, cis_start_size, factor)
    return [{"start": int(a), "end": int(b), "center": int(c)} for a, b, c in tup]

def preprocess_bins_for_rna(
    rna_start: int,
    rna_end: int,
    chrom_length: int,
    bin_size: int,
    cis_start_size: int,
    factor: float,
) -> Tuple[List[Dict], Dict, List[Dict]]:
    upstream_bins = geometric_upstream_bins_full(rna_start, bin_size, cis_start_size, factor)
    gene_bin = {"start": int(rna_start), "end": int(rna_end), "center": int((rna_start + rna_end) // 2)}
    downstream_bins = geometric_downstream_bins_full(rna_end, chrom_length, bin_size, cis_start_size, factor)
    return upstream_bins, gene_bin, downstream_bins
