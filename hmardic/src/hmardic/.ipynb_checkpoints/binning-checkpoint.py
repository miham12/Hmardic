from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

def uniform_bins(chrom_length: int, bin_size: int) -> pd.DataFrame:
    n_bins = (chrom_length + bin_size - 1) // bin_size
    starts = np.arange(n_bins, dtype=np.int64) * bin_size
    ends = np.minimum(starts + bin_size, chrom_length).astype(np.int64)
    centers = ((starts + ends) // 2).astype(np.int64)
    return pd.DataFrame(
        {"bin_index": np.arange(n_bins, dtype=np.int64),
         "start": starts.astype(np.int64),
         "end": ends,
         "center": centers}
    )

def build_trans_bins_by_chr(chrom_sizes: pd.DataFrame, bin_size: int) -> Dict[str, pd.DataFrame]:
    by_chr: Dict[str, pd.DataFrame] = {}
    for _, row in chrom_sizes.iterrows():
        chr_name = row["chr"]
        length = int(row["length"])
        df = uniform_bins(length, bin_size)
        df["dna_chr"] = chr_name
        by_chr[chr_name] = df
    return by_chr

def concat_bins_by_chr(by_chr: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(by_chr.values(), ignore_index=True)
