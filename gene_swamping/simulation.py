# compare the gene drive and mapped non-gene drive simulations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import math
import csv
import argparse
import pandas as pd
from pathlib import Path

from gd_model import gd_simulation
from ngd_model import ngd_simulation
from matrix import find_mapped
from gd_partition import *


def _savefig_safe(fig_or_plt, outpath, **kwargs):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig_or_plt.savefig(outpath, **kwargs)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def compare_sim(params, q2_state, analytic=True):
    """
    Compare single-configuration trajectories between GD and mapped NGD.

    This function runs one GD simulation and one mapped NGD simulation, then
    plots them side-by-side for deme 1 and deme 2.

    When to use:
    - Called when CLI flag `--traj True`.
    - Intended for inspecting trajectory behavior for one specific parameter set.

    Parameters
    ----------
    params : dict
        Simulation settings dictionary. Users should edit the values in `main()`
        to test a specific setup, especially:
        - `s`, `c`, `h` : GD biological parameters
        - `m`           : migration rate
        - `alpha`,`beta`: deme-2 scaling factor
        Other fields (`q1`, `q2`, `target_steps`, `type`) are also used and could 
        be modified when needed.
    q2_state : str
        Regime label used when retrieving mapped parameters for deme 2
        (currently expected to be `"unstable"` in `main()`).
    analytic : bool, default=True
        If True, use analytic mapped NGD parameters; otherwise use grid-based
        mapped results loaded from saved mapping outputs.
    """
    # Run the gene drive simulation
    gd = False
    params['q1'] = 0.1
    params['q2'] = 0.0
    gd_params = params.copy()
    ngd_params = params.copy()
    genotype = params['type']
    # set s1 s2 in gene drive params and find se1 (he1)
    s1, s2 = gd_params['s'], round(gd_params['s']*gd_params['alpha'], 2)
    c2 = round(gd_params['c']*gd_params['beta'],2)
    print("s1, s2:", s1, s2)
    c, h = gd_params['c'], gd_params['h']
    if q2_state == 'unstable':
        q_eq = get_unstable_qeq(gd_params)
        print("Unstable q_eq:", q_eq)
    gd_result = gd_simulation(gd_params)

    tl = 400

    for key in ['q1', 'q2']:
        if len(gd_result[key]) < tl:
            gd_result[key] = np.pad(gd_result[key], (0, tl - len(gd_result[key])), 'edge')
        elif len(gd_result[key]) > tl:
            gd_result[key] = gd_result[key][:tl]
    ### PLOT GD ONLY
    if gd: 
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(gd_result['q1'], label=f"Gene Drive Allele (q1) (m={gd_params['m']}, alpha={gd_params['alpha']})")
        plt.plot(gd_result['q2'], label='Wild-type Allele (q2)')
        plt.title(f"Gene Drive Simulation (s1={gd_params['s']}, s2={gd_params['s'] * gd_params['alpha']:.3f}, c={gd_params['c']}, h={gd_params['h']})")
        plt.xlabel('Generations')
        plt.ylabel('Allele Frequency')
        plt.legend()
        plt.tight_layout()
        _savefig_safe(
            plt,
            f"gd_unstable_traj_fig/q{params['q1']}_{params['type']}_s{gd_params['s']}_c{gd_params['c']}_h{gd_params['h']}_m{gd_params['m']}_alpha{gd_params['alpha']}.png",
            dpi=600,
        )
        plt.show()
        return 
    # find_mapped only works if s1 is in fixation regime
    mapped_config = find_mapped(s1, c, h, genotype, analytic)
    if mapped_config == None:
        print(f"Error in s1 fixation mapping: no mapping found for s1={s1}")
        # This means that s1 is in other regimes like unstable
        if not analytic:
            mapped_config = read_loss_unstable_res(s1, c, h, q2_state) # diploid loss mapping res
        else:
            mapped_config = find_unstable_mapped_analytic(s1, c, h)
    if genotype == "diploid":
        se1, he1 = mapped_config
        ngd_params['h'] = he1
    else:
        se1 = mapped_config
    ngd_params['s'] = se1
    print("deme 1 mapped config:", se1, he1)
    
    ### PLOT NGD FROM MAPPING 
    # find se2 (he2) from loss/unstable mapping
    if analytic:
        ngd_d2_config = find_unstable_mapped_analytic(s2, c2, h) 
    else:
        ngd_d2_config = read_loss_unstable_res(s2, c2, h, q2_state)
    se2, he2 = ngd_d2_config

    ngd_params['alpha'] = se2/se1
    ngd_params['type'] = "diploid"
    ngd_params['h'] = (he1, he2)
    print(gd_params, ngd_params)

    ngd_result = ngd_simulation(ngd_params)
    for key in ['q1', 'q2']:
        if len(ngd_result[key]) < tl:
            ngd_result[key] = np.pad(ngd_result[key], (0, tl - len(ngd_result[key])), 'edge')
        elif len(ngd_result[key]) > tl:
            ngd_result[key] = ngd_result[key][:tl]
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(gd_result['q1'], label=f"Gene Drive Allele D1 (q1) (m={gd_params['m']}, alpha={gd_params['alpha']:.3f})")
    plt.plot(gd_result['q2'], label='Gene Drive Allele D2 (q2)')
    plt.title(f"Gene Drive Simulation (s1={gd_params['s']:.2f}, s2={gd_params['s'] * gd_params['alpha']:.3f}, c={gd_params['c']}, h={gd_params['h']})")
    plt.xlabel('Generations')
    plt.ylabel('Allele Frequency')
    plt.legend(loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(ngd_result['q1'], label=f"Mutant Allele D1(q1) (m={ngd_params['m']}, alpha={ngd_params['alpha']:.3f})")
    plt.plot(ngd_result['q2'], label='Mutant Allele D2(q2)')
    paramstr = (
        f"s1={ngd_params['s']:.2f}, s2={(ngd_params['s'] * ngd_params['alpha']):.2f}, h1={he1:.2f}, h2={he2:.2f}"
        if params['type'] == "diploid"
        else f"s1={ngd_params['s']}, s2={(ngd_params['s'] * ngd_params['alpha']):.3f}"
    )
    plt.title(f"NGD {params['type']} simulation ({paramstr})")
    plt.xlabel('Generations')
    plt.ylabel('Allele Frequency')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    _savefig_safe(
        plt,
        f"/Users/wanbo/Desktop/fig5_traj/q{params['q1']}_{params['type']}_gd_vs_ngd_s{gd_params['s']}_c{gd_params['c']}_h{gd_params['h']}_m{gd_params['m']}_alpha{gd_params['alpha']}.pdf",
        dpi=600,
    )
    plt.show()


### START HELPER FUNCTIONS FOR GETTING GENE SWAMPING TABLE ###########################
def _eval_candidates(funcs, cval, hval):
    vals = []
    for fn in funcs:
        try:
            v = float(fn(cval, hval))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return vals

def _internal_qstar_exists_and_unstable(s2, cval, hval):
    try:
        qstar = float(q3_func(s2, cval, hval))
        # print("Checking qstar in range:", qstar, "for s2:", s2, "c:", cval, "h:", hval)
        if not (0.0 < qstar < 1.0) or not np.isfinite(qstar):
            return False
        der = float(df_at_q3_func(s2, cval, hval))
        return np.isfinite(der) and abs(der) > 1.0
    except Exception:
        return False

def find_beta_range(s, c, h, eps=1e-6):
    beta_min = eps
    beta_max = 1.0
    step = int((beta_max-beta_min)/0.01) 
    beta_grid = np.linspace(beta_min, beta_max, step) 
    
    c2_cands0 = _eval_candidates(c_q3_0_funcs, s, h)
    c2_cands1 = _eval_candidates(c_q3_1_funcs, s, h)
    if not c2_cands0 or not c2_cands1:
        raise ValueError(f"Cannot bracket internal root with q3=0/1 collisions for these ({s},{h}).")
    c2_min = max(min(c2_cands0 + c2_cands1), 0.0)
    c2_max = min(max(c2_cands0 + c2_cands1), 1.0)

    mask = []
    for a in beta_grid:
        c2 = a * c
        in_span = (c2_min <= c2 <= c2_max) or (c2_max <= c2 <= c2_min)
        mask.append(in_span and _internal_qstar_exists_and_unstable(s, c2, h))
    mask = np.array(mask, dtype=bool)

    if not mask.any():
        raise ValueError("No beta range produces an internal unstable fixed point at these (s1,c,h).")

    idx = np.where(mask)[0]
    lo = float(beta_grid[idx[0]])
    hi = float(beta_grid[idx[-1]])

    return lo, hi


def find_alpha_range(s1, cval, hval, case="t1", state="loss", alpha_grid=None):
    # sensible alpha scan if none provided
    if alpha_grid is None:
        alpha_min = 0.01
        alpha_max = min(1.4, 1.0/s1) if case=="t3" else 1.0/s1
        step = int((alpha_max-alpha_min)/0.01)
        alpha_grid = np.linspace(alpha_min, alpha_max, step)   # adjust if you need wider

    if state == "loss":
        s2_bounds = _eval_candidates(s_q3_1_funcs, cval, hval)
        print(s2_bounds)
        if not s2_bounds:
            raise ValueError("No q3=1 bound found for these (c,h).")
        s2_bound = s2_bounds[0]  
        alpha_lo = s2_bound / s1    # note: s2 negative if alpha>0, so signs matter
        alpha_hi = min(1.2, 1.0 / s1) if case == "s3" else 1.0/s1   
        lo, hi = alpha_lo, alpha_hi

    elif state == "unstable":
        # 1) Find the s2-span where an internal root can exist (between q3=0 and q3=1 collisions)
        s2_cands0 = _eval_candidates(s_q3_0_funcs, cval, hval)
        s2_cands1 = _eval_candidates(s_q3_1_funcs, cval, hval)
        if not s2_cands0 or not s2_cands1:
            raise ValueError("Cannot bracket internal root with q3=0/1 collisions for these (c,h).")
        s2_min = max(min(s2_cands0 + s2_cands1), 0.0)
        s2_max = min(max(s2_cands0 + s2_cands1), 1.0)

        # 2) Translate to alpha via s2 = alpha*s1
        mask = []
        for a in alpha_grid:
            s2 = a * s1
            in_span = (s2_min <= s2 <= s2_max) or (s2_max <= s2 <= s2_min)
            mask.append(in_span and _internal_qstar_exists_and_unstable(s2, cval, hval))
        mask = np.array(mask, dtype=bool)

        if not mask.any():
            raise ValueError("No alpha range produces an internal unstable fixed point at these (s1,c,h).")

        # 3) Return the first contiguous interval where condition holds
        idx = np.where(mask)[0]
        lo = float(alpha_grid[idx[0]])
        hi = float(alpha_grid[idx[-1]])

    else:
        raise ValueError("state must be 'loss' or 'unstable'")

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"No valid alpha: lower={lo} ≥ upper={hi}")
    print(lo, hi)
    return lo, hi


def read_loss_unstable_res(s, c, h, state):
    '''
    Load diploid grid mapping CSV (generated by analysis/mapping.py) and
    return mapped (s_ngd, h_ngd) for the input GD config (s, c, h).

    Current naming convention:
      result/grid_mapping/diploid_{state}_h{h}_all.csv
    '''
    state = str(state).lower()
    if state not in {"loss", "unstable", "stable", "fixation"}:
        print(f"Error: unsupported state '{state}'")
        return None

    repo_root = Path(__file__).resolve().parents[1]
    grid_dir = repo_root / "result" / "grid_mapping"
    h = float(h)


    csv_path = grid_dir / f"diploid_{state}_h{h}_all.csv"
    if not csv_path.exists():
        print(
            f"Error: mapping CSV not found for state={state}, h={h}. "
            f"Expected file: {csv_path}"
        )
        return None

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s_gd = float(row["s_gd"])
                c_gd = float(row["c_gd"])
                h_gd = float(row["h_gd"])
            except (KeyError, TypeError, ValueError):
                continue

            if (
                math.isclose(s_gd, float(s), abs_tol=1e-6)
                and math.isclose(c_gd, float(c), abs_tol=1e-6)
                and math.isclose(h_gd, float(h), abs_tol=1e-6)
            ):
                s_ngd_raw = row.get("s_ngd", "")
                h_ngd_raw = row.get("h_ngd", "")
                if s_ngd_raw in ("", None) or h_ngd_raw in ("", None):
                    return None
                try:
                    return (float(s_ngd_raw), float(h_ngd_raw))
                except ValueError:
                    return None

    print(f"Error: GD config ({s}, {c}, {h}) not found in {csv_path}")
    return None

def is_swamping(q, tol=1e-3, decay_thresh=0.1):
    """Check if a single deme is swamping toward 0."""
    if len(q) <= 1 and not abs(q[-1]) < tol:
        print(f"Warning: Trajectory {q} too short to determine swamping.")
        return False
    # Already essentially zero
    if abs(q[-1]) < tol:
        return True
    # Still decreasing and below a small threshold
    if (q[-1] < q[-2] and not math.isclose(q[-1], q[-2]) 
            and q[-1] < decay_thresh):
        return True
    return False

def check_gs_traj(traj, tol=1e-3, decay_thresh=0.1):
    """Check gene swamping: both demes trending to 0 or below tolerance."""
    return is_swamping(traj['q1'], tol, decay_thresh) and \
           is_swamping(traj['q2'], tol, decay_thresh)
    

def get_unstable_qeq(p):
        """Resolve q* if qeq_func not provided."""
        s, c, h, alpha = p['s'], p['c'], p['h'], p['alpha']
        s2 = s*alpha
        if q3_func is not None:
            return float(q3_func(s2, c, h))
    
        raise ValueError("No unstable equilibrium resolver found. ")

def find_unstable_mapped_analytic(s, c, h):
    denom = 2*c*h*s-2*c+s
    if math.isclose(denom, 0.0):
        print("zero denom in analytic")
        return None
    h_ngd = (c*h*s-c+h*s)/(2*c*h*s-2*c+s)
    s_ngd = 2*c*h*s-2*c+s
    return (s_ngd, h_ngd)

### END HELPER FUNCTIONS FOR GETTING GENE SWAMPING TABLE #####################################

# START GENE SWAMPING TABLE GENERATION AND FIGURES FUNCTIONS #################################
def generate_swamping_table(params,
                            d2_state,
                            analytic,
                            case="t1",
                            s2=False,
                            n_m=50,
                            n_alpha=50, 
                            tol=1e-3):
    """
    Generate a swamping-partition table over a 2D sweep and return it as a DataFrame.

    This is the expensive table-construction step used by partition plotting.
    In `main()`, it is called only when a matching table file is not already on disk.
    After first run, the resulting DataFrame is saved to text and reused on
    subsequent runs for the same parameter set to avoid unnecessary recomputation.

    Sweep behavior:
    - If `params['vary_mode'] == 'alpha'`:
      fix `(s1, c1, h)` and sweep `alpha` with `s2 = s1 * alpha`.
    - If `params['vary_mode'] == 'beta'`:
      fix `(s, c1, h)` and sweep `beta` with `c2 = c1 * beta`.

    Parameters
    ----------
    params : dict
        Base simulation/mapping settings (uses keys such as `s`, `c`, `h`,
        `type`, and `vary_mode`).
    d2_state : str
        Regime label used when loading/solving mapped parameters for deme 2
        (current workflow uses `"unstable"`).
    analytic : bool
        True -> use analytic mapping helper; False -> use grid-search mapping.
    case : str, default="t1"
        Scenario label used in alpha-range logic and output folder naming.
    s2 : bool, optional
        Legacy argument retained for compatibility (not actively used).
    n_m : int, default=50
        Number of migration (`m`) grid points for the sweep.
    n_alpha : int, default=50
        Number of ratio grid points (`alpha` or `beta`, depending on vary_mode).
    tol : float, default=1e-3
        Swamping tolerance parameter (reserved for classification behavior).
    """
    vary_mode = params.get('vary_mode', 'alpha').lower()
    s1 = params['s']
    c1 = params['c']
    h  = params.get('h', None)
    n_m = int(50)
    n_alpha = int(50)

    m_vals = np.linspace(0.01, 0.5, n_m)

    # Build the sweep grid depending on mode
    if vary_mode == 'alpha':
        lo, hi = find_alpha_range(s1, c1, h, case, d2_state)
        sweep_vals = np.linspace(lo, hi, n_alpha)
    else:
        # beta sweep: pick a reasonable range
        # e.g., let c2 in (0,1]; if c1 in (0,1], then beta in (eps, 1/c1]
        eps = 1e-3
        lo, hi = find_beta_range(s1, c1, h, eps) 
        sweep_vals = np.linspace(lo, hi, n_alpha)

    rows = []

    # Map the fixed deme 1 GD config once (this is (s1, c1, h) in both modes)
    s1_mapped = find_mapped(s1, c1, h, params['type'], analytic)
    if s1_mapped is None:
        print(f"s1 configuration ({s1}, {c}, {h}) is not in fixation regime")
        if not analytic:
            s1_mapped = read_loss_unstable_res(s1, c1, h, d2_state)
        else:
            s1_mapped = find_unstable_mapped_analytic(s1, c1, h)
    if s1_mapped is None:
        raise RuntimeError("Failed to map deme 1; cannot proceed.")
    se1, he1 = s1_mapped

    # Precompute the mapped (se2, he2) per sweep value
    mapped_d2 = {}  # key: sweep value -> (se2, he2, alpha_e)
    for v in sweep_vals:
        if vary_mode == 'alpha':
            alpha = float(v)
            s2 = round(float(s1 * alpha), 2)
            c2 = round(c1, 2)
        else:
            beta = float(v)
            s2 = float(s1)
            c2 = float(c1 * beta)

        # Map (s2, c2, h) via analytic computation or reading the previously saved mapping results
        if not analytic:
            mapped = read_loss_unstable_res(s2, c2, h, d2_state)
        else:
            mapped = find_unstable_mapped_analytic(s2, c2, h)
        if mapped is None:
            continue
        se2, he2 = mapped

        # Effective alpha_e stays defined as se2/se1
        alpha_e = se2 / se1 if se1 != 0 else np.nan

        key = alpha if vary_mode == 'alpha' else beta
        mapped_d2[key] = (se2, he2, alpha_e)

    # Main sweep over (m, sweep_val)
    for m_val in m_vals:
        for v, (se2, he2, alpha_e) in mapped_d2.items():
            # he2 = he1 # only for supplementary
            if vary_mode == 'alpha':
                alpha = float(v)
                s2    = float(s1 * alpha)
                c2    = c1
            else:
                beta  = float(v)
                s2    = float(s1)
                c2    = float(c1 * beta)

            # --- GD simulation ---
            gd_p = params.copy()
            gd_p.update({
                'm':     round(m_val, 5),
                'alpha': round(alpha if vary_mode=='alpha' else (s2/s1 if s1!=0 else np.nan), 6),  # keep 'alpha' column for plotting if desired
                'c':     c1,      # GD deme-1 c is c1
            })
            # Deme-2 GD parameter is either s2 (alpha mode) or c2 (beta mode).s
            gd_p['beta'] = c2/c1   
            gd_out = gd_simulation(gd_p)
            gs_gd = check_gs_traj(gd_out)

            # --- NGD simulation ---
            ngd_p = params.copy()
            ngd_p.update({
                's':     se1,
                'h':     (he1, he2),  
                'm':     m_val,
                'alpha': alpha_e,    
                'type':  "diploid"
            })
            ngd_out = ngd_simulation(ngd_p)
            gs_ngd = check_gs_traj(ngd_out)

            # Rounding & bookkeeping
            s1_r = round(s1, 6)
            s2_r = round(s2, 6)
            c1_r = round(c1, 6)
            c2_r = round(c2, 6)
            val_r = round(float(v), 6)
            se1_r = round(se1, 6)
            se2_r = round(se2, 6)
            he1_r = round(he1, 6) if he1 is not None else None
            he2_r = round(he2, 6) if he2 is not None else None
            alpha_e_r = round(alpha_e, 6) if alpha_e is not None else None
            if vary_mode == 'alpha':
                alpha_r = val_r
                beta_r  = 1.0
            else:
                beta_r  = val_r
                alpha_r = 1.0

            row = {
                'm':                  round(m_val, 6),
                's1':                 s1_r,
                's2':                 s2_r,
                'c1':                 c1_r,
                'c2':                 c2_r,
                'h':                  round(h, 6) if h is not None else None,
                'alpha':              alpha_r,
                'beta':               beta_r,
                'gene_swamping_gd':   gs_gd,
                'se1':                se1_r,
                'se2':                se2_r,
                'he1':                he1_r,
                'he2':                he2_r,
                'alpha_e':            alpha_e_r,
                'gene_swamping_ngd':  gs_ngd,
                'vary_mode':          vary_mode,
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    return df

### START HELPERS FOR PLOT OVERLAP PARTITION ###################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def _pivot_on_grid(df, value_col, m_vals, alpha_vals):
    # Pivot then reindex to common axes so both GD/NGD align
    pv = df.pivot(index='m', columns='ratio', values=value_col)
    pv = pv.reindex(index=m_vals, columns=alpha_vals)
    return pv.values

import matplotlib

def plot_gs_partition_overlap(df, state="unstable", analytic=False, case="t1"):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'Avenir'
    matplotlib.rcParams['font.weight'] = 'book'
    vary_mode = df['vary_mode'].iloc[0] if 'vary_mode' in df.columns else 'alpha'
    xcol = 'alpha' if vary_mode == 'alpha' else 'beta'
    s_fixed = float(df['s1'].iloc[0])
    c = float(df['c'].iloc[0]) if 'c' in df.columns else float(df['c1'].iloc[0])
    h = float(df['h'].iloc[0])

    df = df[df['m'] <= 0.5].copy()

    # Build grids
    m_vals = np.sort(df['m'].unique())
    x_vals = np.sort(df[xcol].unique())

    gd_df  = df[['m', xcol, 'gene_swamping_gd']].dropna()
    Z_gd  = _pivot_on_grid(gd_df.rename(columns={xcol: 'ratio'}),  'gene_swamping_gd',  m_vals, x_vals)
    Z_gd = (Z_gd > 0.5).astype(int)

    col_name = 'gene_swamping_ngd'
    ngd_df = df[['m', xcol, col_name]].dropna()
    
    Z_ngd = _pivot_on_grid(ngd_df.rename(columns={xcol: 'ratio'}), col_name, m_vals, x_vals)
    Z_ngd = (Z_ngd > 0.5).astype(int)

    labels = (1 * (Z_ngd == 1) + 2 * (Z_gd == 1))
    labels_masked = np.ma.masked_where(labels == 0, labels)

    colors = {
        1: "#32B7DC",  # NGD
        2: "#E84ADE",  # GD
        3: "#e3d2ef",   # Overlap (mix color)
    }
    i = 1
    cmap = ListedColormap(["#ffffff", colors[i], colors[2], colors[3]])
    bounds = np.arange(-0.5, 4.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6.5, 6))

    extent = [x_vals.min(), x_vals.max(), m_vals.min(), m_vals.max()]
    cf = ax.contourf(
        labels_masked, levels=bounds, cmap=cmap, norm=norm,
        antialiased=True, extent=extent
    )

    a = 'Analytic' if analytic else 'Simulation'
    
    xlabel = "alpha" if vary_mode == 'alpha' else "beta (c2 / c1)"
    ax.set_title(f"Gene Swamping Partition (GD vs NGD {a})\nGD: s={s_fixed}, c={c}, h={h}")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("m (migration rate)", fontsize=14)
    ax.set_xlim(x_vals.min(), x_vals.max())
    ax.set_ylim(m_vals.min(), m_vals.max())
    ax.grid(False)

    legend_elements = [
        Patch(facecolor=colors[i], edgecolor='k', label='NGD: swamping'),
        Patch(facecolor=colors[2], edgecolor='k', label='GD: swamping'),
        Patch(facecolor=colors[3], edgecolor='k', label='Overlap'),
    ]
    ax.legend(handles=legend_elements, title="Outcome", loc='best', frameon=True)

    fig.tight_layout()
    ana = "analytic" if analytic else "sim"
    foldername = f"{case}_sim_partition_fig"

    out = f"{foldername}/{state}_{ana}_overlap_s{s_fixed}_c{c}_h{h}_{vary_mode}.pdf"
    _savefig_safe(fig, out)
    plt.show()
    return ax

### END GS PARTITION OVERLAP FIGURE ###################################
def main():
    parser = argparse.ArgumentParser(
        description="Gene swamping trajectory/partition runner."
    )
    parser.add_argument("--traj", type=_str2bool, default=True,
                        help="True: run compare_sim; False: generate/load swamping table and plot partition.")
    parser.add_argument("--analytic", type=_str2bool, default=True,
                        help="Use analytic mapping helper where applicable (default True).")
    parser.add_argument("--case", type=str, default="t1",
                        help="Case label t1, t2, t3 correspondes to the three gene swamping settings described in the paper.")
    args = parser.parse_args()

    params = {
        'q1': 0.1,          # Initial frequency in population 1
        'q2': 0.0,          # Initial frequency in population 2
        'target_steps': 1000,
        's': 0.45,           # Selection coefficient in deme 1 (fixation)
        'c': 0.95,           # Conversion rate
        'h': 1.0,           # Dominance coefficient
        'm': 0.5,          # Migration rate
        'alpha': 1.0,        # Relative fitness of the favoured allele
        'beta': 1.0,
        'type': "diploid", 
        'vary_mode':'alpha'
    }
    traj = args.traj
    analytic = args.analytic
    case = args.case
    case_key = str(case).strip().lower()
    params["vary_mode"] = "alpha" if case_key in {"t1", "t3"} else "beta"
    d2_state = "unstable"
    if traj: 
        compare_sim(params, d2_state, analytic)
    else: 
        # explore_values(params)
        s, c, h, m = params['s'], params['c'], params['h'], params['m'] 
        ana = "analytic" if analytic else "sim"
        folder = f"{case}_swamping_table"
        Path(folder).mkdir(parents=True, exist_ok=True)
        txtname = f"{folder}/gd_{d2_state}_{ana}_s{s}_c{c}_h{h}.txt"
        
        # Plot swamping parititon
        try:
            swamping_df = pd.read_csv(txtname, delim_whitespace=True)
            print("File loaded successfully:")
        except FileNotFoundError:
            print('ERROR: need to generate table first')
            table = generate_swamping_table(params, d2_state, analytic, case)
            # Save to CSV 
            table.to_string(txtname, index=False, justify='left')
            swamping_df = pd.read_csv(txtname, delim_whitespace=True)
        plot_gs_partition_overlap(swamping_df, d2_state, analytic, case)

if __name__ == "__main__":
    main()
