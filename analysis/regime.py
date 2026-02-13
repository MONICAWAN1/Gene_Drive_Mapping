"""
Analytic GD regime partition and stability helpers.

This is the single module for regime classification and lookup. Use this module directly for:
  get_partition, get_configs_in_regime, get_regime, get_stability_table,
  get_regime_and_eq, run_analytic_partition, q3_lambda, derivative_q3_lambda,
  derivative, get_eq, compute_lambda_gd, get_ngd_stability, get_eq_ngd.

CLI to precompute and save partition: python -m analysis.regime --model GD|NGD
"""
import os
import sys
import math
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import try_load_pickle, save_pickle, STABILITY_SUBDIR, stability_filename

# Default grid: same as getcurves (analysis/getcurves.py)
DEFAULT_S_RANGE = np.arange(0, 1.01, 0.01)
DEFAULT_C_RANGE = np.arange(0, 1.01, 0.01)

# ---------------------------------------------------------------------------
# GD analytic recurrence: q3, derivative at q3, expr at 0.5 (same math as partition)
# ---------------------------------------------------------------------------
_q = sp.Symbol("q")
_s = sp.Symbol("s")
_c = sp.Symbol("c")
_h = sp.Symbol("h")
_expr_gd = (_q**2 * (1 - _s) + _q * (1 - _q) * (1 + _c) * (1 - _h * _s)) / (
    _q**2 * (1 - _s) + 2 * _q * (1 - _q) * (1 - _h * _s) + (1 - _q)**2
)
_gd_q1, _gd_q2, _gd_q3 = sp.solvers.solve(_expr_gd - _q, _q)
_expr_gd_derivative = sp.diff(_expr_gd, _q)
_gd_derivative_q3 = sp.simplify(_expr_gd_derivative.subs(_q, _gd_q3))
_gd_expr_0_5 = _expr_gd.subs(_q, 0.5)

q3_lambda = sp.lambdify((_s, _c, _h), _gd_q3, "numpy")
derivative_q3_lambda = sp.lambdify((_s, _c, _h), _gd_derivative_q3, "numpy")
expr_gd_0_5_lambda = sp.lambdify((_s, _c, _h), _gd_expr_0_5, "numpy")

# ---------------------------------------------------------------------------
# NGD analytic: q3, derivative at q3, expr at 0.5 (no c)
# ---------------------------------------------------------------------------
_expr_ngd = (_q**2 * (1 - _s) + _q * (1 - _q) * (1 - _h * _s)) / (
    _q**2 * (1 - _s) + 2 * _q * (1 - _q) * (1 - _h * _s) + (1 - _q)**2
)
_nq1, _nq2, _nq3 = sp.solvers.solve(_expr_ngd - _q, _q)
_expr_ngd_derivative = sp.diff(_expr_ngd, _q)
_ngd_derivative_at_q3 = sp.simplify(_expr_ngd_derivative.subs(_q, _nq3))
_ngd_expr_0_5 = _expr_ngd.subs(_q, 0.5)

ngd_q3_lambda = sp.lambdify((_s, _h), _nq3, "numpy")
ngd_derivative_q3_lambda = sp.lambdify((_s, _h), _ngd_derivative_at_q3, "numpy")
ngd_expr_0_5_lambda = sp.lambdify((_s, _h), _ngd_expr_0_5, "numpy")

_partition_cache = {}


def _safe_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def get_partition(h_gd, s_range=None, c_range=None):
    """
    Analytic partition for GD at given h_gd over (s, c) grid.
    Ranges default to getcurves: s, c in [0, 1.01] step 0.01.

    Returns dict:
      by_regime: {regime: [{"config": (s,c,h), "eq": eq}, ...]}
      by_config: {(s,c,h): {"regime": str, "eq": float}}
    Loads from pickle/OLD/h{h_gd}_gametic_stability_res.pickle if present.
    """
    h_gd = round(float(h_gd), 3)
    if h_gd in _partition_cache:
        return _partition_cache[h_gd]
    raw = try_load_pickle(STABILITY_SUBDIR, stability_filename(h_gd))
    if raw is not None:
        by_regime = {r: [{"config": c, "eq": e} for (c, e) in raw.get(r, [])] for r in ("stable", "unstable", "fixation", "loss")}
        by_config = {}
        for r, pairs in by_regime.items():
            for d in pairs:
                by_config[d["config"]] = {"regime": r, "eq": d["eq"]}
        out = {"by_regime": by_regime, "by_config": by_config}
        _partition_cache[h_gd] = out
        return out
    s_range = s_range if s_range is not None else DEFAULT_S_RANGE
    c_range = c_range if c_range is not None else DEFAULT_C_RANGE
    by_regime = {"stable": [], "unstable": [], "fixation": [], "loss": []}
    by_config = {}
    for s_val in s_range:
        for c_val in c_range:
            try:
                q3 = q3_lambda(s_val, c_val, h_gd)
                dq3 = derivative_q3_lambda(s_val, c_val, h_gd)
                dq05 = expr_gd_0_5_lambda(s_val, c_val, h_gd)
            except Exception:
                continue
            q3 = _safe_float(q3)
            dq3 = _safe_float(dq3)
            dq05 = _safe_float(dq05)
            if q3 is None or dq3 is None or dq05 is None:
                continue
            config = (round(float(s_val), 3), round(float(c_val), 3), h_gd)
            if 0 < q3 < 1 and dq3 < 1:
                r, eq = "stable", q3
            elif 0 < q3 < 1 and dq3 > 1:
                r, eq = "unstable", q3
            elif (q3 >= 1 or q3 <= 0) and dq05 > 0.5:
                r, eq = "fixation", 1.0
            elif (q3 >= 1 or q3 <= 0) and dq05 < 0.5:
                r, eq = "loss", 0.0
            else:
                continue
            by_regime[r].append({"config": config, "eq": eq})
            by_config[config] = {"regime": r, "eq": eq}
    out = {"by_regime": by_regime, "by_config": by_config}
    _partition_cache[h_gd] = out
    return out


def get_configs_in_regime(h_gd, regime):
    """All configs with eq values for the given regime at h_gd. Returns list of {"config": (s,c,h), "eq": eq}."""
    p = get_partition(h_gd)
    return p["by_regime"].get(regime, [])


def get_regime(model, s, h, c=None):
    """
    Regime for a single config. Returns "stable"|"unstable"|"fixation"|"loss" or None.

    - model: "GD" or "NGD"
    - GD: get_regime("GD", s_gd, h_gd, c=c_gd) — c required.
    - NGD: get_regime("NGD", s_ngd, h_ngd) — c ignored.
    """
    r, _ = get_regime_and_eq(model, s, h, c=c)
    return r


def get_stability_table(h_gd):
    """Backward compat: {regime: [(config, eq), ...]} for use by mapping/getDiff/plotting."""
    p = get_partition(h_gd)
    return {
        r: [(d["config"], d["eq"]) for d in p["by_regime"].get(r, [])]
        for r in ("stable", "unstable", "fixation", "loss")
    }


def get_regime_and_eq(model, s, h, c=None):
    """
    Single-config analytic regime and equilibrium.

    - model: "GD" or "NGD"
    - GD: get_regime_and_eq("GD", s_gd, h_gd, c=c_gd) — c required.
    - NGD: get_regime_and_eq("NGD", s_ngd, h_ngd) — c optional and ignored.

    Returns (regime, eq) with regime in "stable"|"unstable"|"fixation"|"loss", or (None, None).
    """
    if model.upper() == "GD":
        if c is None:
            return (None, None)
        try:
            q3 = q3_lambda(s, c, h)
            dq3 = derivative_q3_lambda(s, c, h)
            dq05 = expr_gd_0_5_lambda(s, c, h)
        except Exception:
            return (None, None)
    elif model.upper() == "NGD":
        try:
            q3 = ngd_q3_lambda(s, h)
            dq3 = ngd_derivative_q3_lambda(s, h)
            dq05 = ngd_expr_0_5_lambda(s, h)
        except Exception:
            return (None, None)
    else:
        return (None, None)
    q3 = _safe_float(q3)
    dq3 = _safe_float(dq3)
    dq05 = _safe_float(dq05)
    if q3 is None or dq3 is None or dq05 is None:
        return (None, None)
    if 0 < q3 < 1 and dq3 < 1:
        return ("stable", q3)
    if 0 < q3 < 1 and dq3 > 1:
        return ("unstable", q3)
    if (q3 >= 1 or q3 <= 0) and dq05 > 0.5:
        return ("fixation", 1.0)
    if (q3 >= 1 or q3 <= 0) and dq05 < 0.5:
        return ("loss", 0.0)
    return (None, None)


def run_analytic_partition(model):
    """
    Precompute and save partition for GD (per h) or NGD. 
    CLI: python -m analysis.regime --model GD.
    """
    h_grid = np.arange(0.0, 1.05, 0.05)
    if model == "GD":
        for h_val in h_grid:
            h_val = round(float(h_val), 3)
            table = get_stability_table(h_val)
            save_pickle(STABILITY_SUBDIR, stability_filename(h_val), table)
            print(f"GD: saved h{h_val} partition of {sum(len(v) for v in table.values())} configs")
    else:
        s_grid = np.arange(0.01, 1.01, 0.01)
        table = _compute_regime_table_ngd(s_grid, h_grid)
        save_pickle(STABILITY_SUBDIR, "ngd_gametic_stability_res.pickle", table)
        print(f"NGD: saved partition of {sum(len(v) for v in table.values())} configs")


def _compute_regime_table_ngd(s_grid, h_grid):
    table = {"stable": [], "unstable": [], "fixation": [], "loss": []}
    for s_val in s_grid:
        for h_val in h_grid:
            try:
                q3 = ngd_q3_lambda(s_val, h_val)
                dq3 = ngd_derivative_q3_lambda(s_val, h_val)
                dq05 = ngd_expr_0_5_lambda(s_val, h_val)
            except Exception:
                continue
            q3, dq3, dq05 = _safe_float(q3), _safe_float(dq3), _safe_float(dq05)
            if q3 is None or dq3 is None or dq05 is None:
                continue
            config = (round(float(s_val), 3), round(float(h_val), 3))
            if 0 < q3 < 1 and dq3 < 1:
                table["stable"].append((config, q3))
            elif 0 < q3 < 1 and dq3 > 1:
                table["unstable"].append((config, q3))
            elif (q3 >= 1 or q3 <= 0) and dq05 > 0.5:
                table["fixation"].append((config, 1.0))
            elif (q3 >= 1 or q3 <= 0) and dq05 < 0.5:
                table["loss"].append((config, 0.0))
    return table


# ---------------------------------------------------------------------------
# Stability helpers used by mapping and others (from old stability.py)
# ---------------------------------------------------------------------------

def compute_lambda_gd():
    s, c, h, q = sp.symbols("s c h q")
    sc = (1 - h * s) * c
    sn = 0.5 * (1 - h * s) * (1 - c)
    numerator = q**2 * (1 - s) + 2 * q * (1 - q) * c * (sc + sn)
    wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (sc + 2 * sn) + (1 - q)**2
    q_next = numerator / wbar
    dq_next_dq = sp.simplify(sp.diff(q_next, q))
    return sp.lambdify((s, c, h, q), dq_next_dq, "numpy")


def derivative(params):
    s, c, h = params["config"]
    q = params["currq"]
    sc = c * (1 - h * s) if params.get("conversion") != "zygotic" else c * (1 - s)
    sn = 0.5 * (1 - c) * (1 - h * s)
    num = 2 * q**2 * (1 - q) * (1 - s) * (2 * sn + 1) - 2 * q**2 * (1 - 2 * q) * (1 - s) * sn + 2 * q * (q - 1)**2 * (1 - s + 2 * sn + 2 * sc) + 2 * (1 - 2 * q) * (1 - q)**2 * (sn + sc)
    denom = (q**2 * (1 - s) + 2 * (1 - q) * q * (2 * sn + sc) + (1 - q)**2)**2
    if denom != 0:
        return num / denom
    return float("nan")


def get_eq(params):
    s, c, h = params["config"]
    sn = 0.5 * (1 - c) * (1 - h * s)
    sc = c * (1 - h * s) if params.get("conversion") != "zygotic" else c * (1 - s)
    eqs = {"q1": 0, "q2": 1}
    if 4 * sn + 2 * sc + s - 2 != 0:
        eqs["q3"] = (2 * sn + 2 * sc - 1) / (4 * sn + 2 * sc + s - 2)
    else:
        eqs["q3"] = "NA"
    return eqs


def compute_lambda():
    q, s, h = sp.symbols("q s h")
    numer = q**2 * (1 - s) + q * (1 - q) * (1 - h * s)
    wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (1 - h * s) + (1 - q)**2
    q_next = numer / wbar
    f1 = sp.simplify(sp.diff(q_next, q))
    f2 = sp.simplify(sp.diff(f1, q))
    poly_eq = sp.simplify(sp.expand(numer - q * wbar))
    poly_q = sp.Poly(poly_eq, q)
    coeffs = poly_q.all_coeffs()
    f1_num = sp.lambdify((q, s, h), f1, "numpy")
    f2_num = sp.lambdify((q, s, h), f2, "numpy")
    coef_funcs = [sp.lambdify((s, h), c, "numpy") for c in coeffs]
    return coef_funcs, f1_num, f2_num


_coef_funcs, _f1_num, _f2_num = compute_lambda()


def get_ngd_stability(s, h, q, f1_num=_f1_num):
    try:
        return f1_num(q, s, h)
    except Exception:
        return float("nan")


def get_eq_ngd(s, h):
    if math.isclose(h, 0.5):
        return None
    if math.isclose(2 * h * s - s, 0.0):
        return None
    eq = (h * s) / (2 * h * s - s)
    if eq < 0 or eq > 1 or math.isclose(eq, 0.0) or math.isclose(eq, 1.0):
        return None
    return [eq]


def _gd_next_q(q, s, c, h):
    num = q * q * (1 - s) + q * (1 - q) * (1 + c) * (1 - h * s)
    den = q * q * (1 - s) + 2 * q * (1 - q) * (1 - h * s) + (1 - q) * (1 - q)
    if math.isclose(den, 0.0):
        return float("nan")
    return num / den


def _se_equivalent(s_gd, c_gd, h_gd):
    return h_gd * s_gd - c_gd + c_gd * h_gd * s_gd


def solve_sngd(s_gd, c_gd, h_gd, h_ngd=None, q=None):
    """
    Analytic GD->NGD mapping helper.

    - solve_sngd(s_gd, c_gd, h_gd) -> (s_ngd, h_ngd)
    - solve_sngd(s_gd, c_gd, h_gd, h_ngd=..., q=...) -> s_ngd at a fixed (h_ngd, q)
      by matching one-step update q(t+1) between GD and NGD at q(t)=q.
    """
    s_gd = float(s_gd)
    c_gd = float(c_gd)
    h_gd = float(h_gd)

    if h_ngd is not None and q is not None:
        h_ngd = float(h_ngd)
        q = float(q)
        tgt = _gd_next_q(q, s_gd, c_gd, h_gd)
        if not math.isfinite(tgt):
            return float("nan")
        a = q * q + h_ngd * q * (1 - q)
        b = q * q + 2 * h_ngd * q * (1 - q)
        denom = tgt * b - a
        if math.isclose(denom, 0.0):
            return float("nan")
        return (tgt - q) / denom

    regime, eq = get_regime_and_eq("GD", s_gd, h_gd, c=c_gd)
    se = _se_equivalent(s_gd, c_gd, h_gd)
    if regime in ("stable", "unstable") and eq is not None and not math.isclose(2 * eq - 1, 0.0):
        h_ngd = eq / (2 * eq - 1)
    else:
        h_ngd = h_gd if not math.isclose(h_gd, 0.0) else 1.0
    if math.isclose(h_ngd, 0.0):
        return (0.0, h_ngd)
    return (se / h_ngd, h_ngd)


def solve_sngd_unstable(s_gd, c_gd, h_gd):
    """
    Stable/unstable mapping using GD interior equilibrium to match NGD h.
    """
    s_gd = float(s_gd)
    c_gd = float(c_gd)
    h_gd = float(h_gd)
    regime, eq = get_regime_and_eq("GD", s_gd, h_gd, c=c_gd)
    if regime not in ("stable", "unstable") or eq is None or math.isclose(2 * eq - 1, 0.0):
        return solve_sngd(s_gd, c_gd, h_gd)
    h_ngd = eq / (2 * eq - 1)
    se = _se_equivalent(s_gd, c_gd, h_gd)
    if math.isclose(h_ngd, 0.0):
        return (0.0, h_ngd)
    return (se / h_ngd, h_ngd)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute and save GD or NGD analytic partition.")
    parser.add_argument("--model", choices=["GD", "NGD"], required=True, help="GD or NGD model.")
    args = parser.parse_args()
    run_analytic_partition(args.model)
