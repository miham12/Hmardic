from __future__ import annotations

from typing import Tuple
import pandas as pd

def read_chrom_sizes(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", names=["chr", "length"])

def _read_bed_like(path: str, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=names)
    # keep only the columns we actually use
    keep = [c for c in ["chr", "start", "end", "rna"] if c in df.columns]
    return df[keep].copy()

def read_contacts(path: str) -> pd.DataFrame:
    # your current files have 6 columns; we keep first 4
    return _read_bed_like(path, ["chr", "start", "end", "rna", "idk", "idk2"])

def read_rna_annot(path: str) -> pd.DataFrame:
    return _read_bed_like(path, ["chr", "start", "end", "rna", "idk", "idk2"])

def read_nonspecific_contacts(path: str) -> pd.DataFrame:
    return _read_bed_like(path, ["chr", "start", "end", "rna", "idk", "idk2"])

def read_real_data(
    chrom_sizes: str,
    contacts: str,
    rna_annot: str,
    nonspecific_contacts: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        read_chrom_sizes(chrom_sizes),
        read_contacts(contacts),
        read_rna_annot(rna_annot),
        read_nonspecific_contacts(nonspecific_contacts),
    )
