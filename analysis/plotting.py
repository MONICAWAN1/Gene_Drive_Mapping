import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from models import run_model, wm, haploid
from utils import PICKLE_DIR
from utils.file_io import RESULT_DIR


REPO_ROOT = Path(PICKLE_DIR).parent
FIGURE_DIR = REPO_ROOT / "figure"


def _set_plot_style():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _figure_path(*parts):
    path = FIGURE_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _truncate_colormap(cmap, minval=0.1, maxval=0.7, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"truncated({cmap.name},{minval:.2f},{maxval:.2f})", new_colors
    )


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "s_gd": _to_float(row.get("s_gd")),
                "c_gd": _to_float(row.get("c_gd")),
                "h_gd": _to_float(row.get("h_gd")),
                "s_ngd": _to_float(row.get("s_ngd")),
                "h_ngd": _to_float(row.get("h_ngd")),
                "MSE": _to_float(row.get("MSE")),
            }
            rows.append(parsed)
    return rows


def _collect_rows(method, genotype, h_gd, regime=None):
    method_dir = RESULT_DIR / f"{method}_mapping"
    if not method_dir.exists():
        raise FileNotFoundError(f"Missing mapping results directory: {method_dir}")

    if regime:
        pattern = f"{genotype}_{regime}_h{h_gd}_*.csv"
    else:
        pattern = f"{genotype}_*_h{h_gd}_*.csv"

    files = sorted(method_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No mapping CSV found for method={method}, genotype={genotype}, h={h_gd}, regime={regime} under {method_dir}"
        )

    merged = {}
    for p in files:
        for row in _load_rows(p):
            key = (row["s_gd"], row["c_gd"], row["h_gd"])
            if None in key:
                continue
            merged[key] = row
    return list(merged.values()), files


def _find_mapping_row(method, genotype, s_gd, c_gd, h_gd):
    rows, files = _collect_rows(method, genotype, h_gd, regime=None)
    matched = [
        r
        for r in rows
        if math.isclose(r["s_gd"], float(s_gd), rel_tol=0, abs_tol=1e-9)
        and math.isclose(r["c_gd"], float(c_gd), rel_tol=0, abs_tol=1e-9)
        and math.isclose(r["h_gd"], float(h_gd), rel_tol=0, abs_tol=1e-9)
    ]
    if not matched:
        raise ValueError(
            f"Config (s={s_gd}, c={c_gd}, h={h_gd}) not found in {method} mapping CSV(s): {[str(p) for p in files]}"
        )
    matched.sort(key=lambda r: float("inf") if r["MSE"] is None else r["MSE"])
    return matched[0]


def _gd_curve(s_gd, c_gd, h_gd, q0=0.001, target_steps=40000):
    params = {
        "s": float(s_gd),
        "c": float(c_gd),
        "h": float(h_gd),
        "q0": float(q0),
        "target_steps": int(target_steps),
    }
    return run_model(params)["q"]


def _mapped_curve(genotype, row, q0=0.001, target_steps=40000):
    s_ngd = row["s_ngd"]
    h_ngd = row["h_ngd"]
    if s_ngd is None:
        return None
    if genotype == "haploid":
        return haploid({"s": s_ngd, "q0": float(q0), "target_steps": int(target_steps)})["q"]
    if h_ngd is None:
        return None
    return wm({"s": s_ngd, "h": h_ngd, "q0": float(q0), "target_steps": int(target_steps)})["q"]


def plot_mapping(genotype, s, c, h, analytic=True, grid=True, q0=0.001, target_steps=40000):
    """
    Trajectory plot: original GD curve with mapped analytic/grid trajectories.
    """
    _set_plot_style()
    genotype = str(genotype).lower()
    if genotype not in {"diploid", "haploid"}:
        raise ValueError("genotype must be 'diploid' or 'haploid'")
    if not analytic and not grid:
        raise ValueError("At least one of analytic/grid must be enabled")

    gd_curve = _gd_curve(s, c, h, q0=q0, target_steps=target_steps)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, len(gd_curve)), gd_curve, color="#ca1f8e", label=f"GD (s={s}, c={c}, h={h})")

    if grid:
        row_grid = _find_mapping_row("grid", genotype, s, c, h)
        curve_grid = _mapped_curve(genotype, row_grid, q0=q0, target_steps=target_steps)
        if curve_grid is not None:
            if genotype == "haploid":
                label = f"Grid {genotype} NGD (s={row_grid['s_ngd']:.3f})"
            else:
                label = f"Grid {genotype} NGD (s={row_grid['s_ngd']:.3f}, h={row_grid['h_ngd']:.3f})"
            plt.plot(np.arange(0, len(curve_grid)), curve_grid, color="#0e169e", linestyle="-", marker="o", markersize=2, label=label)

    if analytic:
        row_ana = _find_mapping_row("analytic", genotype, s, c, h)
        curve_ana = _mapped_curve(genotype, row_ana, q0=q0, target_steps=target_steps)
        if curve_ana is not None:
            mse_text = "" if row_ana["MSE"] is None else f", mse={row_ana['MSE']:.4f}"
            if genotype == "haploid":
                label = f"Analytic {genotype} NGD (s={row_ana['s_ngd']:.3f}{mse_text})"
            else:
                label = f"Analytic {genotype} NGD (s={row_ana['s_ngd']:.3f}, h={row_ana['h_ngd']:.3f}{mse_text})"
            plt.plot(np.arange(0, len(curve_ana)), curve_ana, color="#0e169e", linestyle="--", marker="o", markersize=2, label=label)

    plt.ylabel("Gene Drive/Mutant Allele Frequency", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.title(f"Comparison of GD and mapping trajectories at h = {h}, q_init = {q0}")
    plt.grid(False)
    plt.legend(title="population condition", loc="lower right", bbox_to_anchor=(1, 0))

    outpath = _figure_path("trajectory", f"{genotype}_h{h}_s{s}_c{c}_q{q0}.pdf")
    plt.savefig(outpath, dpi=600, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {outpath}")


def _plot_mse_heatmap(rows, title, outpath):
    diffmap = {}
    for row in rows:
        if row["MSE"] is None:
            continue
        diffmap[(row["s_gd"], row["c_gd"], row["h_gd"])] = row["MSE"]

    if not diffmap:
        raise ValueError("No valid MSE values found in mapping CSV")

    all_s = sorted(set(conf[0] for conf in diffmap.keys()))
    all_c = sorted(set(conf[1] for conf in diffmap.keys()))
    errors = np.full((len(all_s), len(all_c)), np.nan)
    s_index = {v: i for i, v in enumerate(all_s)}
    c_index = {v: i for i, v in enumerate(all_c)}

    for (s, c, _), err in diffmap.items():
        errors[s_index[s], c_index[c]] = err

    cmin = np.nanmin(errors)
    cmax = 0.01
    cmap = _truncate_colormap(plt.get_cmap("Reds"), 0.1, 0.7)

    plt.figure(figsize=(9, 7))
    plt.imshow(
        errors,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[0, 1, 0, 1],
        vmin=cmin,
        vmax=cmax,
    )
    plt.colorbar(label="Mean Squared Error (Error of mapping)")
    tick_vals = np.round(np.arange(0, 1.01, 0.1), 2)
    plt.xticks(tick_vals)
    plt.yticks(tick_vals)
    plt.xlabel("c in gene drive", fontsize=15)
    plt.ylabel("s in gene drive", fontsize=15)
    plt.title(title)
    plt.savefig(outpath, dpi=600, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {outpath}")


def plot_hap_diff(h, method="grid"):
    """
    Haploid error heatmap from saved mapping CSV.
    method=analytic compares Se approximation vs GD (saved by mapping.py analytic output).
    method=grid uses saved haploid grid-search mapping output.
    """
    _set_plot_style()
    method = str(method).lower()
    if method not in {"grid", "analytic"}:
        raise ValueError("method must be 'grid' or 'analytic'")

    rows, files = _collect_rows(method, "haploid", h, regime="fixation")
    title = f"Mapping Error from GD to NGD at h = {h} with haploid {method} for fixation"
    outpath = _figure_path("error_heatmap", f"haploid_{method}_fixation_h{h}.pdf")
    _plot_mse_heatmap(rows, title, outpath)


def plot_diff(h, regime, method):
    """
    Diploid error heatmap from saved mapping CSV.
    """
    _set_plot_style()
    if not regime:
        raise ValueError("--regime is required for diploid plot_diff")

    method = str(method).lower()
    regime = str(regime).lower()
    if method not in {"grid", "analytic"}:
        raise ValueError("method must be 'grid' or 'analytic'")
    if regime not in {"fixation", "loss", "stable", "unstable"}:
        raise ValueError("regime must be one of: fixation, loss, stable, unstable")

    rows, files = _collect_rows(method, "diploid", h, regime=regime)
    title = f"Mapping Error from GD to NGD at h = {h} with {method} for {regime} Regime"
    outpath = _figure_path("error_heatmap", f"diploid_{method}_{regime}_h{h}.pdf")
    _plot_mse_heatmap(rows, title, outpath)
