from __future__ import annotations

from typing import Tuple
import pandas as pd

def read_chrom_sizes(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["chr", "length"],
        usecols=[0, 1],
        dtype={"chr": str, "length": "int64"},
    )

def _read_bed_like(path: str, has_rna: bool) -> pd.DataFrame:
    usecols = [0, 1, 2, 3] if has_rna else [0, 1, 2]
    names = ["chr", "start", "end", "rna"] if has_rna else ["chr", "start", "end"]
    return pd.read_csv(
        path,
        sep="\t",
        names=names,
        usecols=usecols,
        dtype={"chr": str, "start": "int64", "end": "int64", **({"rna": str} if has_rna else {})},
    )

def read_contacts(path: str) -> pd.DataFrame:
    return _read_bed_like(path, has_rna=True)

def read_rna_annot(path: str) -> pd.DataFrame:
    return _read_bed_like(path, has_rna=True)

def read_nonspecific_contacts(path: str) -> pd.DataFrame:
    return _read_bed_like(path, has_rna=False)

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
