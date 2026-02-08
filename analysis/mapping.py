import numpy as np
import math
import pickle
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sympy as sp
import concurrent.futures
from itertools import repeat

from .regime import derivative, compute_lambda_gd, get_stability_table, get_regime, get_regime_and_eq
# from .plotting import plot_lambda_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_pickle, save_pickle, euclidean, write_mapping_csv, GD_RES_SUBDIR, NGD_RES_SUBDIR, HAPLOID_RES_SUBDIR, gd_res_filename, ngd_res_filename, haploid_res_filename
from models import haploid, run_model, wm, haploid_se
from .solve import solve_sngd, solve_sngd_unstable

def get_eq(params):
    s, c, h = params['s'], params['c'], params['h']
    sn = 0.5*(1-c)*(1-h*s)
    if params['conversion'] == 'zygotic':
        sc = c*(1-s)
    else:
        sc = c*(1-h*s)

    eqs = {'q1':0, 'q2':1}

    if 4*sn+2*sc+s-2 != 0:
        q3 = (2*sn+2*sc-1)/(4*sn+2*sc+s-2)
    else:
        q3 = 'NA'
        print(f"no eq result, config = {params['config']}")
    eqs['q3'] = q3
    return eqs

'''
Given the NGD parameter (s, h), return the stable eq
'''
def get_eq_ngd(s, h):
    if not math.isclose(h, 0.5):
        return (h*s)/(2*h*s - s)
    return 'NA'

# def get_ngd_stability(s, h, q):
#     num = 2*q**2*(1-q)*(1-s)*(1-h*s) - q**2*(1-2*q)*(1-s)*(1-h*s) + 2*q*(1-q)**2*(2-s-h*s) + (1-2*q)*(1-q)**2*(1-h*s)

#     denom = (q**2*(1-s) + 2*(1-q)*q*(1-h*s) + (1-q)**2)**2

#     if denom != 0:
#         slope = num/denom
#     else:
#         slope = 'NA'
#         print(f"no slope result, config = {(s, h, q)}")
    
#     return slope
q, s, h = sp.symbols('q s h')

# Define the symbolic expression for q(t+1)
numerator = q**2 * (1 - s) + q * (1 - q) * (1 - h * s)
wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (1 - h * s) + (1 - q)**2
q_next = numerator / wbar

# Derivative of q(t+1) with respect to q(t)
dq_next_dq = sp.simplify(sp.diff(q_next, q))

# Convert the symbolic derivative into a Python function
dq_dq_func = sp.lambdify((s, h, q), dq_next_dq, "numpy")



def get_ngd_stability(s, h, q):
    try:
        slope = dq_dq_func(s, h, q)
    except ZeroDivisionError:
        slope = 'NA'
        print(f"Division by zero: config = {(s, h, q)}")
    return slope


"""
Check if NGD config (s, h) has the same regime as gdState ("stable" or "unstable").
Uses regime.get_regime_and_eq("NGD", s, h) for analytic NGD regime.
"""
def check_state_eq(s, h, q, gdState):
    r, _ = get_regime_and_eq("NGD", s, h)
    return gdState == r

def loadStability(currH):
    """
    GD regime partition for given h. Returns {state: [(config, eq), ...]}
    with state in 'stable', 'unstable', 'fixation', 'loss'; config = (s, c, h).
    Uses cached pickle if present, else computes analytically.
    """
    return get_stability_table(currH)

def loadGres(currH, gdFile="001"):
    """Load GD simulation results; path matches getcurves.py save."""
    return load_pickle(GD_RES_SUBDIR, gd_res_filename(currH, gdFile))


def _configs_for_h(gd_results, stability_res, h_gd, regime, s_gd=None, c_gd=None):
    """Return list of (s, c, h) configs to map: either single config or all in regime for h_gd."""
    gd_configs, gd_res = gd_results[0], gd_results[1]
    if s_gd is not None and c_gd is not None:
        return [(float(s_gd), float(c_gd), float(h_gd))]
    h_f = float(h_gd)
    if regime == "fixation":
        configs = [cfg for (cfg, eq) in stability_res.get("fixation", []) if math.isclose(cfg[2], h_f)]
    elif regime == "loss":
        configs = [cfg for (cfg, eq) in stability_res.get("loss", []) if math.isclose(cfg[2], h_f)]
    else:
        key = regime.lower()
        configs = [cfg for (cfg, eq) in stability_res.get(key, []) if math.isclose(cfg[2], h_f)]
    return configs if configs else [(s, c, h) for (s, c, h) in gd_configs if math.isclose(h, h_f)]

def hap_grid_mapping(h_gd, s_gd=None, c_gd=None, gd_file="001", save=True, target_steps=40000, q0=0.001):
    """
    Haploid grid mapping (fixation only). Returns list of dicts with s_gd, c_gd, h_gd, s_ngd, h_ngd (empty), MSE.
    If save=True, writes CSV to result/grid_mapping/.
    """
    params = {"h": h_gd, "target_steps": target_steps, "q0": q0}
    gd_results = loadGres(h_gd, gd_file)
    stability_res = loadStability(h_gd)
    configs = _configs_for_h(gd_results, stability_res, h_gd, "fixation", s_gd, c_gd)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    hap_results = load_pickle(HAPLOID_RES_SUBDIR, haploid_res_filename())
    rows = []
    for (s, c, h) in configs:
        if ((s, c, h), 1.0) not in stability_res.get("fixation", []):
            continue
        gd_curve = gd_res[(s, c, h)]["q"]
        best_diff = np.inf
        best_s = None
        for ngd_key, ngd_curve in hap_results.items():
            diff = euclidean(ngd_curve["q"], gd_curve)
            if diff < best_diff:
                best_diff = diff
                best_s = ngd_key
        if best_s is None:
            continue
        rows.append({
            "s_gd": round(s, 4), "c_gd": round(c, 4), "h_gd": round(h, 4),
            "s_ngd": round(float(best_s), 4), "h_ngd": "", "MSE": round(best_diff, 6)
        })
    if save and rows:
        write_mapping_csv("grid", "haploid", "fixation", h_gd, s_gd, c_gd, rows)
    return rows

def hap_analytic_mapping(h_gd, s_gd=None, c_gd=None, gd_file="001", save=True, target_steps=40000, q0=0.001):
    """
    Haploid analytic mapping (fixation): Se = h*s - c + c*h*s, trajectory from haploid_se. Same output format as hap_grid_mapping.
    """
    params = {"h": h_gd, "target_steps": target_steps, "q0": q0}
    gd_results = loadGres(h_gd, gd_file)
    stability_res = loadStability(h_gd)
    configs = _configs_for_h(gd_results, stability_res, h_gd, "fixation", s_gd, c_gd)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    rows = []
    for (s, c, h) in configs:
        if ((s, c, h), 1.0) not in stability_res.get("fixation", []):
            continue
        gd_curve = gd_res[(s, c, h)]["q"]
        se = h * s - c + c * h * s  # analytic s_effective
        param_se = {"s": s, "c": c, "h": h, "target_steps": target_steps, "q0": q0}
        hap_curve = haploid_se(param_se)["q"]
        mse = euclidean(hap_curve, gd_curve)
        rows.append({
            "s_gd": round(s, 4), "c_gd": round(c, 4), "h_gd": round(h, 4),
            "s_ngd": round(se, 4), "h_ngd": "", "MSE": round(mse, 6)
        })
    if save and rows:
        write_mapping_csv("analytic", "haploid", "fixation", h_gd, s_gd, c_gd, rows)
    return rows

def get_delta_lambda(gd_config, ngd_config, eq):
    params = {'config':gd_config, 'currq':eq, 'conversion': "gametic"}
    gd_lambda = derivative(params)
    ngd_lambda = get_ngd_stability(ngd_config[0], ngd_config[1], eq)
    return (gd_lambda-ngd_lambda, ngd_lambda)

def same_eq(s_ngd, h_ngd, eq):
    """
    Check if the NGD eq is the same as the GD eq.
    """
    ngd_eq = get_eq_ngd(s_ngd, h_ngd)
    return math.isclose(ngd_eq, eq)
    
def find_candidate(
        s_ngd:float, 
        h_ngd:float, 
        q_ngd:float,
        state:str,  
        gd_curve:list, 
        eq:float) -> tuple:
    """
    Compute MSE and traj for a given (s_ngd, h_ngd) if it matches the target state and eq.
    Returns mse or np.inf if not matching.
    """
    # check if NGD has same state and eq (within tolerance)
    if not check_state_eq(s_ngd, h_ngd, eq, state):
        # print("NOT SAME STATE", s_ngd, h_ngd, eq)
        return None, np.inf
    if not same_eq(s_ngd, h_ngd, eq):
        # print("NOT SAME EQ", s_ngd, h_ngd, eq)
        return None, np.inf

    # compute trajectory and metrics

    traj = wm({'s': s_ngd, 'h': h_ngd, 'target_steps': len(gd_curve), 'q0': q_ngd})['q']
    if len(traj) == 1:
        return None, np.inf
    mse = euclidean(traj, gd_curve)  # MSE
    return (traj, mse)

def evaluate_sngd(s_ngd, h_ngd, params, q_init_list, eq, state, gd_config):
    s_ngd = round(s_ngd, 3)
    s, c, h = gd_config
    all_q_MSE = 0
    ### boundary for s_ngd and h_ngd
    if s_ngd * h_ngd > 1:
        return s_ngd, np.inf, None
    new_params = params.copy()
    new_params['s'] = round(s, 3)
    new_params['c'] = round(c, 3)
    new_params['h'] = round(h, 3)
    for q_ngd in q_init_list:
        curr_ngd_curve = None
        if q_ngd == eq:
            print("q_ngd == eq", q_ngd)
            continue
        new_params['q0'] = q_ngd
        gd_curve = run_model(new_params)['q']

        ### check candidate and get MSE
        res = find_candidate(s_ngd, h_ngd, q_ngd, state, gd_curve, eq)
        if res[1] != np.inf:
            curr_ngd_curve, curr_MSE = res[0], res[1]
            all_q_MSE += curr_MSE
        
    return s_ngd, all_q_MSE, curr_ngd_curve
        
def diploid_grid_mapping(regime, h_gd, s_gd=None, c_gd=None, gd_file="001", save=True, target_steps=40000, q0=0.001):
    """
    Combined grid mapping for diploid: fixation, stable, unstable, loss. Returns list of dicts
    with s_gd, c_gd, h_gd, s_ngd, h_ngd, MSE. If save=True, writes CSV to result/grid_mapping/.

    - Fixation: best-matching NGD from precomputed diploid grid (NGD_RES_SUBDIR/ngd_res_filename() from getcurves).
    - Stable/Unstable: equilibrium-based; h_ngd from eq; s_range and q_init_list as in attached spec.
    - Loss: grid search over q_init_list [0.001, 0.2, 0.5, 0.8], then take the s_ngd with minimum average MSE.
    """
    params = {"h": h_gd, "target_steps": target_steps, "q0": q0, "n": 500}
    gd_results = loadGres(h_gd, gd_file)
    stability_res = loadStability(h_gd)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    configs = _configs_for_h(gd_results, stability_res, h_gd, regime, s_gd, c_gd)
    rows = []

    if regime == "fixation":
        ngd_results = load_pickle(NGD_RES_SUBDIR, ngd_res_filename())
        for (s, c, h) in configs:
            if ((s, c, h), 1.0) not in stability_res.get("fixation", []):
                continue
            run_params = {**params, "s": s, "c": c, "h": h}
            gd_curve = run_model(run_params)["q"]
            best_diff = np.inf
            best_ngd = None
            for ngd_key, ngd_curve in ngd_results.items():
                diff = euclidean(ngd_curve, gd_curve)
                if diff < best_diff:
                    best_diff = diff
                    best_ngd = ngd_key
            if best_ngd is not None:
                rows.append({"s_gd": round(s, 4), "c_gd": round(c, 4), "h_gd": round(h, 4),
                    "s_ngd": round(best_ngd[0], 4), "h_ngd": round(best_ngd[1], 4), "MSE": round(best_diff, 6)})

    elif regime == "loss":
        # Haploid grid over q_init_list; average mapped s_ngd and MSE across q_inits
        q_init_list = [0.001, 0.2, 0.5, 0.8]
        for (s, c, h) in configs:
            if get_regime("GD", s, h, c=c) != "loss":
                continue
            best_ngd_config = {}
            run_params_base = {**params, "s": s, "c": c, "h": h}
            for q_init in q_init_list:
                best_ngd_config_q = None
                best_diff = np.inf
                run_params = {**run_params_base, "q0": q_init}
                gd_curve = run_model(run_params)["q"]
                for ngd_s in np.arange(0.01, 1.01, 0.01):
                    ngd_s = round(float(ngd_s), 4)
                    ngd_params = {"s": ngd_s, "target_steps": target_steps, "q0": q_init}
                    ngd_curve = haploid(ngd_params)["q"]
                    diff = euclidean(ngd_curve, gd_curve)
                    if diff < best_diff:
                        best_diff = diff
                        best_ngd_config_q = ngd_s
                best_ngd_config[q_init] = (best_ngd_config_q, best_diff)
            all_mapped_se = [res[0] for res in best_ngd_config.values()]
            all_mapped_diff = [res[1] for res in best_ngd_config.values()]
            avg_se = sum(all_mapped_se) / len(q_init_list)
            avg_diff = sum(all_mapped_diff) / len(q_init_list)
            # Loss mapping is haploid: no h_ngd; store empty or N/A for CSV
            rows.append({"s_gd": round(s, 4), "c_gd": round(c, 4), "h_gd": round(h, 4),
                "s_ngd": round(avg_se, 4), "h_ngd": "", "MSE": round(avg_diff, 6)})

    elif regime in ("stable", "unstable"):
        state = regime
        for (s, c, h) in configs:
            params_eq = {"s": s, "c": c, "h": h, "conversion": "gametic", "config": (s, c, h)}
            eq = get_eq(params_eq)["q3"]
            if eq == "NA" or math.isclose(eq, 0.5) or ((s, c, h), eq) not in stability_res.get(regime, []):
                continue
            h_ngd = eq / (2 * eq - 1)
            # q_init_list and s_values exactly as in attached code
            if regime == "unstable":
                q_init_list = np.arange(0, 1.0, 0.05)
                if h_ngd < 0:
                    s_values = np.arange(1 / h_ngd, 0.0, 0.01)
                else:
                    s_values = np.arange(0.0, min(1 / h_ngd, 1), 0.01)
            else:  # stable
                q_init_list = np.arange(0.1, 1.0, 0.05)
                if h_ngd < 0:
                    s_values = np.arange(0.0, 1.0, 0.01)
                else:
                    s_values = np.arange(-10.0, min(1 / h_ngd, 0), 0.01)
            best = {"mse": np.inf, "s_ngd": None}
            gd_config = (s, c, h)
            run_params = {**params, "s": s, "c": c, "h": h}
            for s_ngd in s_values:
                s_ngd = round(float(s_ngd), 4)
                _, all_q_MSE, _ = evaluate_sngd(s_ngd, h_ngd, run_params, q_init_list, eq, state, gd_config)
                if 0 < all_q_MSE < best["mse"]:
                    best["mse"] = all_q_MSE
                    best["s_ngd"] = s_ngd
            if best["s_ngd"] is not None:
                rows.append({"s_gd": round(s, 4), "c_gd": round(c, 4), "h_gd": round(h, 4),
                    "s_ngd": best["s_ngd"], "h_ngd": round(h_ngd, 4), "MSE": round(best["mse"], 6)})

    else:
        raise ValueError(f"regime must be fixation, loss, stable, or unstable; got {regime}")

    if save and rows:
        write_mapping_csv("grid", "diploid", regime, h_gd, s_gd, c_gd, rows)
    return rows


def analytic_mapping(regime, h_gd, s_gd=None, c_gd=None, gd_file="001", save=True, target_steps=40000, q0=0.001):
    """
    Diploid analytic mapping: uses solve_ngd (solve_sngd_unstable) for stable/unstable/loss,
    solve_ngd_fix (solve_sngd) for fixation. Returns same row format as diploid_grid_mapping.
    """
    gd_results = loadGres(h_gd, gd_file)
    stability_res = loadStability(h_gd)
    configs = _configs_for_h(gd_results, stability_res, h_gd, regime, s_gd, c_gd)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    rows = []
    for (s, c, h) in configs:
        if regime == "fixation":
            if ((s, c, h), 1.0) not in stability_res.get("fixation", []):
                continue
            mapped_s, mapped_h = solve_sngd(s, c, h)
        else:
            mapped_s, mapped_h = solve_sngd_unstable(s, c, h)
        gd_curve = gd_res[(s, c, h)]["q"]
        q_range = np.arange(0.01, 1.0, 0.1) if regime == "unstable" else [0.01] if regime == "fixation" else [0.8]
        total_mse = 0
        for q_init in q_range:
            gd_params = {"s": s, "c": c, "h": h, "q0": q_init, "target_steps": target_steps}
            mapped_curve = wm({'s': mapped_s, 'h': mapped_h, 'target_steps': target_steps, 'q0': q_init})["q"]
            total_mse += euclidean(run_model(gd_params)["q"], mapped_curve)
        mse = total_mse / len(q_range) if q_range else 0
        rows.append({"s_gd": round(s, 4), "c_gd": round(c, 4), "h_gd": round(h, 4),
            "s_ngd": round(mapped_s, 4), "h_ngd": round(mapped_h, 4), "MSE": round(mse, 6)})
    if save and rows:
        write_mapping_csv("analytic", "diploid", regime, h_gd, s_gd, c_gd, rows)
    return rows

