import pickle
import os
import sys
from pathlib import Path

# Resolve repo root regardless of where script is called
ROOT = Path(__file__).resolve().parents[1]

PICKLE_DIR = ROOT / "pickle"
RESULT_DIR = ROOT / "result"

# Default hint when no help string is provided (pipeline order from README)
_DEFAULT_PICKLE_HELP = (
    "1) python analysis/getcurves.py  2) python analysis/stability.py  "
    "3) python scripts/run_mapping.py <map_function> <h> <gdFile> [-s]  4) python scripts/run_plot.py"
)


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