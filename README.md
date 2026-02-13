# Gene Drive project

Repository for simulation models

This repository implements an end-to-end pipeline for gene-drive (GD) and non-gene-drive (NGD) Mendelian model analysis: it precomputes simulation trajectories, computes analytic regime partitions (fixation/loss/stable/unstable), maps GD parameter configurations `(s, c, h)` to NGD configurations, quantifies mapping error (MSE), and visualizes trajectory fits and error heatmaps from saved CSV outputs. For each of the regimes, users can choose to use either analytic or simulated-based grid search mapping method. 

## Top-Level Directories

### [`analysis/`](/analysis)
Core analysis modules for simulation preprocessing (`getcurves.py`), analytic regime computation (`regime.py`), mapping (`mapping.py`), and plotting (`plotting.py`).

### [`models/`](/models)
GD and NGD recurrence models used by the analysis pipeline.

### [`scripts/`](/scripts)
CLI entry points for mapping and plotting (`run_mapping.py`, `run_plot.py`).

### [`utils/`](/utils)
Shared helpers for file I/O, path conventions, and numeric utilities.

## 🧪 Typical Workflow

Run these steps in order from repo root.

1. **Precompute trajectories (run twice: GD and NGD)**

```bash
python3 analysis/getcurves.py --model GD --genotype diploid --h 0.8 --target_steps 2000 --q0 0.001
python3 analysis/getcurves.py --model NGD --genotype diploid --target_steps 2000 --q0 0.001
```

For haploid trajectory precompute:

```bash
python3 analysis/getcurves.py --model NGD --genotype haploid --target_steps 2000 --q0 0.001
```

Arguments:
- `--model`: `GD` or `NGD`.
- `--genotype`: `haploid` or `diploid`.
  - If `haploid`, `gethaploid` is used (NGD-style haploid simulation path).
  - If `diploid`, `getcurves` is used.
- `--h`: GD `h` value (used for `--model GD`; ignored for diploid NGD precompute).
- `--target_steps`: max simulation steps (default `40000`).
- `--q0`: initial allele frequency (default `0.001`).

2. **Compute and cache regime data**

```bash
python3 analysis/regime.py --model GD
python3 analysis/regime.py --model NGD
```

`analysis/regime.py` contains helper functions used to analytically classify configurations.

Arguments:
- `--model`: `GD` or `NGD`.
  - For `GD`, partition pickles are saved per `h` value over the internal `h` grid.

3. **Run mapping and store CSV outputs**

```bash
python3 scripts/run_mapping.py --genotype diploid --regime fixation --method grid --h_gd 0.8 --save
```

Arguments:
- `--genotype`: `diploid` or `haploid`.
- `--regime`: `fixation | loss | unstable | stable | auto`.
  - For `haploid`, only `fixation` is valid.
  - For `auto`, provide a specific config (`--s_gd` and `--c_gd`) so regime can be inferred analytically.
- `--method`: `analytic` or `grid`.
- `--s_gd`, `--c_gd`: optional; if provided, map one specific GD config.
- `--h_gd`: required GD `h` value.
- `--save`: default `True`; writes mapping rows to CSV.

Output CSV format (used by all mapping types):

```python
result/{method}_mapping/{genotype}_{regime}_h{h_gd}_s{s_gd}_c{c_gd}.csv
# if s_gd/c_gd are omitted (map all configs at h_gd):
result/{method}_mapping/{genotype}_{regime}_h{h_gd}_all.csv
```

4. **Plot results from saved mapping CSVs**

Trajectory plot (GD + mapped analytic/grid curves):

```bash
python3 scripts/run_plot.py plot_mapping --genotype diploid --s 0.2 --c 0.9 --h 0.8 --analytic --grid --regime fixation
```

Haploid error heatmap:

```bash
python3 scripts/run_plot.py plot_hap_diff --h 0.8 --method analytic
python3 scripts/run_plot.py plot_hap_diff --h 0.8 --method grid
```

Diploid error heatmap:

```bash
python3 scripts/run_plot.py plot_diff --h 0.7 --regime fixation --method analytic
python3 scripts/run_plot.py plot_diff --h 0.7 --regime fixation --method grid
```

Notes:
- `plot_diff` requires `--regime`.
- Plotting reads `MSE` and mapped parameters from mapping CSVs under `result/`.
- Figures are saved under `figure/`.

---

## ⚙️ Mapping & Comparison Pipeline

- **Haploid mapping**
  - `hap_grid_mapping`: grid-based mapping to `s_ngd` (no `h_ngd`), saves CSV rows with `MSE`.
  - `hap_analytic_mapping`: analytic `Se`-based mapping, saves CSV rows with `MSE`.

- **Diploid mapping**
  - `diploid_grid_mapping`: unified grid mapping for `fixation/loss/stable/unstable`, writes CSV rows (`s_gd, c_gd, h_gd, s_ngd, h_ngd, MSE`).
  - `analytic_mapping`: analytic mapping by regime (uses solver helpers in `regime.py`), writes CSV rows in the same schema.

- **Comparison metrics**
  - Mapping quality is tracked by trajectory `MSE` and consumed directly by plotting heatmaps.

---

## 🛠️ Requirements

- Python >= 3.7
- Install dependencies:

```bash
pip install -r requirements.txt
```

Core libraries:

| Library | Purpose |
|---------|---------|
| NumPy | numerical routines |
| pandas | CSV/data handling |
| matplotlib | plotting |
| sympy | analytic regime/solver expressions |

---

## 📊 Outputs

- Precomputed trajectory pickles under `pickle/`
- Mapping CSVs under `result/{grid_mapping|analytic_mapping}/`
- Figures under `figure/trajectory/` and `figure/error_heatmap/`

---

## 🧬 About Gene Drives

Gene drives are genetic mechanisms that bias inheritance, enabling a chosen allele to spread through a population faster than Mendelian rules predict. They hold promise for controlling vectors (e.g. malaria-carrying mosquitoes), invasive species, and agricultural pests, while also raising ecological and ethical questions.

---

## 📫 Contact

Maintained by **[Monica Wan](https://github.com/MONICAWAN1)**.
