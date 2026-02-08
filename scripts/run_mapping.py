"""
Mapping pipeline: get mapping result, compute mapping error, optionally store to CSV.

Usage:
  python3 run_mapping.py --genotype {diploid|haploid} --regime {fixation|loss|unstable|stable|auto} --method {grid|analytic} [--s_gd S] [--c_gd C] --h_gd H [--save/--no-save]

- genotype: diploid or haploid (haploid only supports regime=fixation).
- regime: fixation, loss, unstable, stable, or auto (auto requires specific --s_gd --c_gd; uses analytical partition to compute regime).
- method: grid or analytic.
- save: default True; write CSV to result/{method}_mapping/{genotype}_{regime}_h{h_gd}_s{s_gd}_c{c_gd}.csv or _all.csv.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.mapping import (
    diploid_grid_mapping,
    analytic_mapping,
    hap_grid_mapping,
    hap_analytic_mapping,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run mapping: get mapping result, compute error, optionally save to CSV."
    )
    parser.add_argument("--genotype", type=str, required=True, choices=["diploid", "haploid"],
        help="diploid or haploid mapping")
    parser.add_argument("--regime", type=str, required=True,
        choices=["fixation", "loss", "unstable", "stable", "auto"],
        help="Regime: fixation, loss, unstable, stable, or auto (auto needs --s_gd --c_gd)")
    parser.add_argument("--method", type=str, required=True, choices=["grid", "analytic"],
        help="grid or analytic")
    parser.add_argument("--s_gd", type=float, default=None, help="GD s (optional; if set with --c_gd, single config)")
    parser.add_argument("--c_gd", type=float, default=None, help="GD c (optional)")
    parser.add_argument("--h_gd", type=float, required=True, help="GD h")
    parser.add_argument("--gd_file", type=str, default="001", help="GD results file label (default 001)")
    parser.add_argument("--save", dest="save", action="store_true", default=True,
        help="Save results to CSV (default)")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save CSV")

    args = parser.parse_args()

    if args.genotype == "haploid":
        if args.regime != "fixation":
            sys.exit("For genotype=haploid the only available regime is fixation. Exiting.")
        if args.regime == "auto":
            sys.exit("For genotype=haploid regime cannot be auto. Use regime=fixation. Exiting.")

    regime = args.regime
    if regime == "auto":
        if args.s_gd is None or args.c_gd is None:
            sys.exit("For regime=auto you must provide --s_gd and --c_gd (specific GD config). Exiting.")
        from analysis.regime import get_regime
        regime = get_regime("GD", args.s_gd, args.h_gd, c=args.c_gd)
        if regime is None:
            sys.exit("Could not determine regime for the given GD config. Exiting.")
        print(f"Auto detected regime for (s={args.s_gd}, c={args.c_gd}, h={args.h_gd}): {regime}")

    s_gd = args.s_gd
    c_gd = args.c_gd
    h_gd = args.h_gd
    save = args.save
    gd_file = args.gd_file

    if args.genotype == "haploid":
        if args.method == "grid":
            rows = hap_grid_mapping(h_gd, s_gd=s_gd, c_gd=c_gd, gd_file=gd_file, save=save)
        else:
            rows = hap_analytic_mapping(h_gd, s_gd=s_gd, c_gd=c_gd, gd_file=gd_file, save=save)
    else:
        if args.method == "grid":
            rows = diploid_grid_mapping(regime, h_gd, s_gd=s_gd, c_gd=c_gd, gd_file=gd_file, save=save)
        else:
            rows = analytic_mapping(regime, h_gd, s_gd=s_gd, c_gd=c_gd, gd_file=gd_file, save=save)

    print(f"Mapping completed: {len(rows)} result(s).")
    for r in rows[:5]:
        print(r)
    if len(rows) > 5:
        print("...")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
