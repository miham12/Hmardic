from __future__ import annotations

import argparse

from .config import HmardicParams
from .io import read_real_data
from .pipeline import run_calling


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hmardic", description="Hmardic: HMM-based calling of significant RNA-DNA contacts")
    p.add_argument("--chrom-sizes", required=True)
    p.add_argument("--contacts", required=True)
    p.add_argument("--rna-annot", required=True)
    p.add_argument("--nonspecific-contacts", required=True)

    p.add_argument("--bin-size", type=int, default=None, help="Uniform genome-wide bin size. If omitted, it is optimized separately for each RNA.")
    p.add_argument("--threads", type=int, default=1)

    p.add_argument("--decay", type=float, default=1e6)
    p.add_argument("--pseudo", type=float, default=1e-10)

    p.add_argument("--trans-start", type=int, default=1_000)
    p.add_argument("--trans-end", type=int, default=10_000)
    p.add_argument("--trans-step", type=int, default=1_000)

    p.add_argument("--tolerance", type=float, default=0.01, help="Tolerance for bin-size optimization plateau detection.")
    p.add_argument("--w", type=int, default=5)

    p.add_argument("--p-bg", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--max-iter", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-1, help="Tolerance for HMM training convergence.")

    p.add_argument("--return_all_bins", action="store_true")
    p.add_argument("--out-prefix", default="hmardic")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    chrom_sizes, contacts, rna_annot, nonspecific = read_real_data(
        args.chrom_sizes,
        args.contacts,
        args.rna_annot,
        args.nonspecific_contacts,
    )

    params = HmardicParams(
        bin_size=args.bin_size,
        decay=args.decay,
        pseudo=args.pseudo,
        trans_start=args.trans_start,
        trans_end=args.trans_end,
        trans_step=args.trans_step,
        tolerance=args.tolerance,
        w=args.w,
        p_bg=args.p_bg,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        return_all_bins=args.return_all_bins,
    )

    bins_all, peaks_all = run_calling(
        chrom_sizes=chrom_sizes,
        contacts=contacts,
        rna_annot=rna_annot,
        nonspecific_contacts=nonspecific,
        params=params,
        threads=args.threads,
    )

    peaks_path = f"{args.out_prefix}.peaks.tsv"
    peaks_all.to_csv(peaks_path, sep="\t", index=False)
    print(f"Saved: {peaks_path}")

    if args.return_all_bins:
        bins_path = f"{args.out_prefix}.bins.tsv"
        bins_all.to_csv(bins_path, sep="\t", index=False)
        print(f"Saved: {bins_path}")


if __name__ == "__main__":
    main()
