from __future__ import annotations

import argparse
import pandas as pd

from .config import HmardicParams
from .io import read_real_data
from .pipeline import run_calling

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hmardic", description="Hmardic: HMM-based calling of significant RNA-DNA contacts")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--contacts", required=True)
    p.add_argument("--rna-annot", required=True)
    p.add_argument("--nonspecific-contacts", required=True)

    p.add_argument("--bin-size", type=int, default=None, help="Trans bin size. If set with --factor -> fixed-bin mode.")
    p.add_argument("--cis-start-size", type=int, default=1000)
    p.add_argument("--factor", type=float, default=None, help="Geometric factor for cis bins. If set with --bin-size -> fixed-bin mode.")

    p.add_argument("--threads", type=int, default=1)

    p.add_argument("--decay", type=float, default=1e6)
    p.add_argument("--pseudo", type=float, default=1e-10)

    # optimization knobs
    p.add_argument("--trans-start", type=int, default=1_000)
    p.add_argument("--trans-end", type=int, default=10_000)
    p.add_argument("--trans-step", type=int, default=1_000)

    p.add_argument("--cis-factor-min", type=float, default=1.1)
    p.add_argument("--cis-factor-max", type=float, default=2.0)
    p.add_argument("--cis-factor-step", type=float, default=0.1)

    p.add_argument("--tolerance", type=float, default=0.01) # Для оптимизации размеров бинов, если будет использоваться 
    p.add_argument("--w", type=int, default=5)

    # HMM
    p.add_argument("--p-bg", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-1) # для останвоки обучения баума вельча

    # HMM execution mode
    # p.add_argument(
    #     "--light-version",
    #     action="store_true",
    #     help=(
    #         "Light mode: fixed transition probs (A/pi) and train only emission scales (p_bg/alpha) on cis. "
    #         "Much faster and fully reproducible."
    #     ),
    # )

    p.add_argument("--merge-peaks", action="store_true")
    p.add_argument("--out-prefix", default="hmardic")
    return p

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Use env var so the flag also works inside worker processes.
    if args.light_version:
        import os
        os.environ["HMARDIC_LIGHT_VERSION"] = "1"

    chrom_sizes, contacts, rna_annot, nonspecific = read_real_data(
        args.chrom_sizes, args.contacts, args.rna_annot, args.nonspecific_contacts
    )

    params = HmardicParams(
        bin_size=args.bin_size,
        cis_start_size=args.cis_start_size,
        factor=args.factor,
        decay=args.decay,
        pseudo=args.pseudo,
        trans_start=args.trans_start,
        trans_end=args.trans_end,
        trans_step=args.trans_step,
        cis_factor_min=args.cis_factor_min,
        cis_factor_max=args.cis_factor_max,
        cis_factor_step=args.cis_factor_step,
        tolerance=args.tolerance,
        w=args.w,
        p_bg=args.p_bg,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        merge_peaks=args.merge_peaks,
    )

    bins_all, peaks_all = run_calling(
        chrom_sizes=chrom_sizes,
        contacts=contacts,
        rna_annot=rna_annot,
        nonspecific_contacts=nonspecific,
        params=params,
        threads=args.threads,
    )

    bins_path = f"{args.out_prefix}.bins.tsv"
    peaks_path = f"{args.out_prefix}.peaks.tsv"
    bins_all.to_csv(bins_path, sep="\t", index=False)
    peaks_all.to_csv(peaks_path, sep="\t", index=False)

    print(f"Saved: {bins_path}")
    print(f"Saved: {peaks_path}")

if __name__ == "__main__":
    main()
