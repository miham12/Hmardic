from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import pyranges as pr

from .overlaps import as_pyranges_intervals


@dataclass(frozen=True)
class NonspecificIndex:
    """Reusable index for nonspecific contacts.
    - ns_total_by_chr: non-specific contacts per chromosome 
    - pr_by_chr: PyRanges per chromosome
    """
    ns_total_by_chr: Dict[str, int]
    pr_by_chr: Dict[str, pr.PyRanges]


def build_nonspecific_index(nonspecific_contacts: pd.DataFrame) -> NonspecificIndex:
    if nonspecific_contacts is None or nonspecific_contacts.empty:
        return NonspecificIndex(ns_total_by_chr={}, pr_by_chr={})

    df = nonspecific_contacts[["chr", "start", "end"]].copy()
    df["chr"] = df["chr"].astype(str)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    ns_total = df["chr"].value_counts().to_dict()

    pr_by_chr: Dict[str, pr.PyRanges] = {}
    for chrom, g in df.groupby("chr", observed=True):
        pr_by_chr[str(chrom)] = as_pyranges_intervals(g, chrom_col="chr", start_col="start", end_col="end")


    #ns_total_by_chr -- каунты по х
    return NonspecificIndex(ns_total_by_chr=ns_total, pr_by_chr=pr_by_chr) 
