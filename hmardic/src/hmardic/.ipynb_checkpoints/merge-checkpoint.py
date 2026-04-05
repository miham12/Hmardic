from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def merge_adjacent_peaks(
    df: pd.DataFrame,
    *,
    state_col: str = "state",
    peak_state: int = 1,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Merge adjacent/overlapping intervals per group (default: rna,dna_chr).

    Works with two common inputs:
      1) Per-bin rows (e.g. output of call_states) that contain `state_col`.
         Then we filter df[state_col] == peak_state and merge.
      2) Already-called peak intervals (e.g. output of call_peaks) WITHOUT `state_col`.
         Then we treat ALL rows as peak intervals and just merge overlaps/adjacency.

    Output columns: group_cols + ['start','end'].
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = ["rna", "dna_chr"]

    needed_base = set(group_cols) | {"start", "end"}
    missing_base = needed_base - set(df.columns)
    if missing_base:
        raise ValueError(f"merge_adjacent_peaks: missing columns: {sorted(missing_base)}")

    # If state_col exists -> filter to peaks; otherwise assume df is already peaks.
    if state_col in df.columns:
        peak_df = df[df[state_col].to_numpy(copy=False) == peak_state]
        if peak_df.empty:
            return pd.DataFrame(columns=[*group_cols, "start", "end"])
    else:
        peak_df = df

    # Sort once globally for deterministic merging
    sdf = peak_df.sort_values([*group_cols, "start", "end"], kind="mergesort")

    out_rows = []
    for keys, g in sdf.groupby(group_cols, observed=True, sort=False):
        starts = g["start"].to_numpy(np.int64, copy=False)
        ends = g["end"].to_numpy(np.int64, copy=False)
        if starts.size == 0:
            continue

        cur_s = int(starts[0])
        cur_e = int(ends[0])

        for s, e in zip(starts[1:], ends[1:]):
            s = int(s)
            e = int(e)

            # Merge if overlapping OR touching (adjacent)
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
                row["start"] = cur_s
                row["end"] = cur_e
                out_rows.append(row)
                cur_s, cur_e = s, e

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["start"] = cur_s
        row["end"] = cur_e
        out_rows.append(row)

    return pd.DataFrame(out_rows)