# %%
import sympy
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import csv
import math
from pathlib import Path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gd_model import gd_simulation, one_step

model = "ngd"
ROOT = Path(__file__).resolve().parents[1]
PICKLE_DIR = ROOT / "pickle"
RESULT_DIR = ROOT / "result"


def _find_mapping_csv(method, genotype, regime, h):
    method_dir = RESULT_DIR / f"{method}_mapping"
    for h_try in (h, round(float(h), 3), round(float(h), 2), round(float(h), 1)):
        path = method_dir / f"{genotype}_{regime}_h{h_try}_all.csv"
        if path.exists():
            return path
    return None


def _read_mapping_row(csv_path, s, c, h):
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
                return row
    return None

#%%
def get_analytics():
    #%%
    model = "ngd"
    type = "Diploid"
    q1 = sympy.Symbol('q1')
    q2 = sympy.Symbol('q2')
    s = sympy.Symbol('s')
    c = sympy.Symbol('c')
    h = sympy.Symbol('h')
    h2 = sympy.Symbol('h2')
    m = sympy.Symbol('m')
    alpha = sympy.Symbol('alpha')

    s2 = s * alpha ## selection coefficient in deme 2
    
    #%%
    # Migration
    q1_m = q1*(1-m) + q2*m
    q2_m = q2*(1-m) + q1*m
    # Selection
    if model == "ngd":
        if type == "Haploid":
            print("hap")
            wA1, wa1 = 1-s, 1
            wA2, wa2 = 1-s2, 1
            expr_d1 = q1_m*wA1/(q1_m*wA1+(1-q1_m)*wa1)
            expr_d2 = q2_m*wA2/(q2_m*wA2+(1-q2_m)*wa2)    
        if type == "Diploid":
        # Fitness parameters (a favored in deme 2, A favored in deme 1):
            wAA1, wAa1, waa1 = 1 - s,   1 - h*s,       1
            wAA2, wAa2, waa2 = 1 - s2,  1 - h2*s2,      1
            # wAA1, wAa1, waa1 = sympy.Symbol('wAA1'), sympy.Symbol('wAa1'), sympy.Symbol('waa1')
            # wAA2, wAa2, waa2 = sympy.Symbol('wAA2'), sympy.Symbol('wAa2'), sympy.Symbol('waa2')


            # Selection
            expr_d1 = (q1_m**2 * wAA1 + q1_m * (1-q1_m) * wAa1) / (q1_m**2 * wAA1 + 2 * q1_m * (1-q1_m) * wAa1 + (1-q1_m)**2 * waa1)
            expr_d2 = (q2_m**2 * wAA2 + q2_m * (1-q2_m) * wAa2) / (q2_m**2 * wAA2 + 2 * q2_m * (1-q2_m) * wAa2 + (1-q2_m)**2 * waa2)
    # Gene Drive selection
    else: 
        # Gene Drive fitness: A is gene drive allele and is favored in deme 1
        wAA1, wAa1, waa1 = 1 - s,   1 - h*s,       1
        wAA2, wAa2, waa2 = 1 - s2,  1 - h*s2,      1
        # wAA1, wAa1, waa1 = sympy.Symbol('wAA1'), sympy.Symbol('wAa1'), sympy.Symbol('waa1')
        # wAA2, wAa2, waa2 = sympy.Symbol('wAA2'), sympy.Symbol('wAa2'), sympy.Symbol('waa2')

        # Selection
        s_n1, s_c1 = 0.5 * (1-c) * wAa1, c * wAA1
        s_n2, s_c2 = 0.5 * (1-c) * wAa2, c * wAA2
        expr_d1 = (q1_m**2 * wAA1 + 2 * q1_m * (1-q1_m) * (s_n1 + s_c1)) / (q1_m**2 * wAA1 + 2 * q1_m * (1-q1_m) * (2 * s_n1 + s_c1) + (1-q1_m)**2 * waa1)
        expr_d2 = (q2_m**2 * wAA2 + 2 * q2_m * (1-q2_m) * (s_n2 + s_c2)) / (q2_m**2 * wAA2 + 2 * q2_m * (1-q2_m) * (2 * s_n2 + s_c2) + (1-q2_m)**2 * waa2)

    # q_11, q_12, eq1 = sympy.solvers.solve(expr_d1 - q1, q1, dict=True)
    # q_21, q_22, eq2 = sympy.solvers.solve(expr_d2 - q2, q2, dict=True)
    eq1 = eq2 = 0

    #%%
    ################################################################################
    # 1) FULL SYMBOLIC JACOBIAN 
    ################################################################################
    J_sym = sympy.Matrix([[sympy.diff(expr_d1, q1),
                        sympy.diff(expr_d1, q2)],
                        [sympy.diff(expr_d2, q1),
                        sympy.diff(expr_d2, q2)]])

    ################################################################################
    # 2) SUBSTITUTE q1 = q2 = 0 
    ################################################################################
    subs_eq  = {q1: eq1, q2: eq2}
    J_eq_sym = sympy.simplify(J_sym.subs(subs_eq))

    print("\n--- Jacobian at the internal equilibrium (symbolic) ---")
    sympy.pretty_print(J_eq_sym)

    #%%
    tr  = sympy.trace(J_eq_sym)
    det = J_eq_sym.det()
    lambda_max = sympy.simplify((tr + sympy.sqrt(tr**2 - 4*det)) / 2)

    print("Leading eigenvalue (lambda)")
    sympy.pretty_print(lambda_max)

    mcrit_solutions = sympy.solve(lambda_max - 1, m)

    if model == "ngd":
        if type == "Diploid":
            m_free_syms = (s, h, alpha, h2)
        else:
            m_free_syms = (s, alpha)

    mcrit_exprs = [sympy.simplify(sol) for sol in mcrit_solutions]
    for me in mcrit_exprs:
        print("mcrit_exprs ##############")
        sympy.pretty_print(me)
    mcrit_funcs = [sympy.lambdify(m_free_syms, expr, modules="numpy") for expr in mcrit_exprs]


    if model == "ngd":
        if type == "Diploid":
            free_syms = (s, h, m, alpha, h2)
        else:
            free_syms = (s, m, alpha)
    else:
        free_syms = (s, h, m, c, alpha, h2)

    eig_func = sympy.lambdify(free_syms, lambda_max, modules="numpy")
    #%%
    def m_threshold(s_val, h_val, alpha_val, h2_val=None, *, prefer="min"):
        """
        Evaluate all symbolic solutions for mcrit and pick a physically valid one.
        Returns np.nan if none are valid.
        prefer:
          - "min": pick smallest valid root in [0,1]
          - "max": pick largest valid root in [0,1]
        """
        #%%
        # evaluate candidates
        vals = []
        for f in mcrit_funcs:
            try:
                if type == "Diploid":
                    v = f(s_val, h_val, alpha_val, h2_val)
                else:
                    v = f(s_val, alpha_val)
            except Exception:
                continue

            # Handle numpy arrays / scalars uniformly
            v = np.array(v, dtype=np.complex128)

            # keep real roots only
            real_mask = np.isfinite(v.real) & (np.abs(v.imag) < 1e-8)
            v_real = v.real[real_mask]

            # physical range
            v_real = v_real[(v_real >= 0.0) & (v_real <= 1.0)]
            vals.extend(v_real.tolist())

        if len(vals) == 0:
            lam_m0 = eig_func(s_val, h_val, 0.0, alpha_val, h2_val)

            if lam_m0 > 1:
                print(f"[m_threshold] No valid mcrit in [0,1]: A invades for all m (λ(m=0)={lam_m0:.4g})")
                return np.inf
            else:
                print(f"[m_threshold] No valid mcrit in [0,1]: A is swamped for all m (λ(m=0)={lam_m0:.4g})")
                return 0.0

        if prefer == "max":
            return float(np.max(vals))
        return float(np.min(vals))

    return eig_func, m_threshold, mcrit_exprs

eig_func, m_threshold, mcrit_exprs = get_analytics()

#######################################
# Read mapped ngd parameters from stored mapping results
# OR find analytic ngd
#######################################
def find_mapped(s, c, h, gtype, analytic = False):
    '''
    Old version:
    Read the old mapping result pickle for gd fixation regime 
    and find the mapped ngd
    '''
    mapped = None
    type = gtype
    if analytic: 
        denom = (
        -2 * (c**2) * (h**2) * (s**2)
        + 4 * (c**2) * h * s
        - 2 * (c**2)
        - 2 * c * (h**2) * (s**2)
        + 4 * c * h * s
        - 2 * c
        + s
        )
    
        s_ngd = denom  # from your formula, s_ngd equals this polynomial

        num = c * h * s - c + h * s
        h_ngd = num / denom if denom != 0 else float("nan")

        return s_ngd, h_ngd
    
    csv_path = _find_mapping_csv("grid", type, "fixation", h)
    if csv_path is None:
        return None
    row = _read_mapping_row(csv_path, s, c, h)
    if row is None:
        return None

    s_ngd = row.get("s_ngd")
    h_ngd = row.get("h_ngd")
    if s_ngd in ("", None):
        return None
    if type == "haploid":
        mapped = float(s_ngd)
    else:
        if h_ngd in ("", None):
            return None
        mapped = (float(s_ngd), float(h_ngd))
    return mapped

if __name__ == "__main__":
    params = {"s": -0.5, "h":0.8, "m": 0.01, "alpha": 2.5, "c": 0.8}


# %%
