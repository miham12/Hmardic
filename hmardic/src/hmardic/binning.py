from __future__ import annotations

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
