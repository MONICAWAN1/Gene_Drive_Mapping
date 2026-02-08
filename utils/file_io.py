import pickle
import os
import sys
import csv
from pathlib import Path

# Resolve repo root regardless of where script is called
ROOT = Path(__file__).resolve().parents[1]

PICKLE_DIR = ROOT / "pickle"
RESULT_DIR = ROOT / "result"

# Pickle path conventions (must match between getcurves/regime writers and loaders)
GD_RES_SUBDIR = "gd_res"
NGD_RES_SUBDIR = "ngd_res"
HAPLOID_RES_SUBDIR = "allhapresults"
STABILITY_SUBDIR = "OLD"

def gd_res_filename(h_val, gd_file="001"):
    """Filename for GD simulation results; matches getcurves.py save."""
    return f"allres{gd_file}G_GD_h{h_val}.pickle"

def ngd_res_filename():
    """Filename for NGD simulation results; matches getcurves.py save."""
    return "allres001G_NGD.pickle"

def haploid_res_filename():
    """Filename for haploid simulation results; matches getcurves.py save."""
    return "allhapres0001G.pickle"

def stability_filename(h_val):
    """Filename for regime partition; matches regime.py save."""
    return f"h{h_val}_gametic_stability_res.pickle"

# Default hint when no help string is provided (pipeline order from README)
_DEFAULT_PICKLE_HELP = (
    "1) python analysis/getcurves.py  2) python -m analysis/regime --model GD  "
    "3) python scripts/run_mapping.py <map_function> <h> <gdFile> [-s]  4) python scripts/run_plot.py"
)


def try_load_pickle(subdir, filename):
    """
    Load a pickle file if it exists; otherwise return None.
    Use for optional/cached data (e.g. regime partition).
    """
    if subdir:
        file_path = Path(PICKLE_DIR) / subdir / filename
    else:
        file_path = Path(PICKLE_DIR) / filename
    if not file_path.exists():
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_pickle(subdir, filename, help_msg=""):
    """
    Load a pickle file from the repo's pickle/ directory.

    subdir: subdirectory under pickle/ (use "" for files at root of pickle/).
    filename: name of the pickle file.
    help_msg: optional instructions to show when the file is missing.
    """
    if subdir:
        file_path = Path(PICKLE_DIR) / subdir / filename
    else:
        file_path = Path(PICKLE_DIR) / filename
    if not file_path.exists():
        hint = help_msg.strip() or _DEFAULT_PICKLE_HELP
        sys.exit(
            f"\nMissing required pickle file:\n  {file_path}\n\n"
            f"To generate it, run (in order):\n  {hint}\n"
        )
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(subdir, filename, res):
    """
    Save a pickle file under the repo's pickle/ directory.

    subdir: subdirectory under pickle/ (use "" for files at root of pickle/).
    filename: name of the pickle file.
    res: object to pickle (cannot be None).
    """
    if subdir:
        subdir_path = Path(PICKLE_DIR) / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        file_path = subdir_path / filename
    else:
        file_path = Path(PICKLE_DIR) / filename
    with open(file_path, "wb") as f:
        pickle.dump(res, f)


def write_mapping_csv(method, genotype, regime, h_gd, s_gd=None, c_gd=None, rows=None):
    """
    Write mapping results to CSV under result/{method}_mapping/.
    Creates all subdirectories. Filename: {genotype}_{regime}_h{h_gd}_s{s_gd}_c{c_gd}.csv
    or {genotype}_{regime}_h{h_gd}_all.csv when s_gd/c_gd not specified (all configs).
    rows: list of dicts with keys s_gd, c_gd, h_gd, s_ngd, h_ngd (optional for haploid), MSE.
    """
    out_dir = ROOT / "result" / f"{method}_mapping"
    out_dir.mkdir(parents=True, exist_ok=True)
    if s_gd is not None and c_gd is not None:
        fname = f"{genotype}_{regime}_h{h_gd}_s{s_gd}_c{c_gd}.csv"
    else:
        fname = f"{genotype}_{regime}_h{h_gd}_all.csv"
    path = out_dir / fname
    if not rows:
        return str(path)
    fieldnames = ["s_gd", "c_gd", "h_gd", "s_ngd", "h_ngd", "MSE"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return str(path)