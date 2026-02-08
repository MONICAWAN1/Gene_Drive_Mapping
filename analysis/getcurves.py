import argparse
import numpy as np
import math
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import run_model, wm, haploid
from utils import save_pickle, GD_RES_SUBDIR, NGD_RES_SUBDIR, HAPLOID_RES_SUBDIR, gd_res_filename, ngd_res_filename, haploid_res_filename

### run simulation for a list of configurations, returns a list of configs and a dictionary {config: curve} #############
def gd_simulation(params, h_gd):
    '''
    decides what configurations the current simulation set includes
    '''
    configs = []
    s_vals = np.arange(0.01, 1.01, 0.01)
    c_vals = np.arange(0.01, 1.01, 0.01)

    # get a list of configurations for simulations 
    for s in s_vals:
        for c in c_vals:
            configs.append((round(float(s), 3), round(float(c), 3), round(float(h_gd), 3)))

    gd_results = dict()
    for (s, c, h) in configs:
        params['s'], params['c'], params['h'] = s, c, h
        gd_res = run_model(params)
        gd_results[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = gd_res

    return configs, gd_results

def getcurves(params, model, h_gd=None):
    '''
    Pickle dump NGD results: {(s,h): curve} or GD results [configlist, gd_res].
    For GD, h_gd is the single h value to run (required for GD); params['h'] must match when model is GD.
    '''
    minVal, maxVal, step = 0.01, 5.0, 0.01
    s_range = np.arange(minVal, maxVal, step)
    hmax = 5.0
    h_range = np.arange(0, hmax, step)
    wm_results = dict()

    if model == "NGD":
        for s_nat in s_range:
            for h_nat in h_range:
                wm_curve = wm({'s': s_nat, 'h': h_nat, 'target_steps': params['target_steps'], 'q0': params['q0']})['q']
                if (not math.isclose(wm_curve[-1], 1.0) and (not math.isclose(wm_curve[-1], 0.0))):
                    wm_results[(round(s_nat, 3), round(h_nat, 3))] = wm_curve
        save_pickle(NGD_RES_SUBDIR, ngd_res_filename(), wm_results)
    elif model == "GD":
        assert h_gd is not None, "h_gd is required for GD"
        gd_configs, gd_res = gd_simulation(params, h_gd)
        gd_results = [gd_configs, gd_res]
        h_val = params['h'] if h_gd is None else h_gd
        save_pickle(GD_RES_SUBDIR, gd_res_filename(h_val), gd_results)
    


def gethaploid(params):
    '''
    Run haploid (NGD) model over s range; save trajectories. Uses params['h'], params['target_steps'], params['q0'].
    '''
    hapRes = dict()
    minVal, maxVal, step = -10, 1, 0.001
    s_range = np.arange(minVal, maxVal, step)
    for s in s_range:
        print(s)
        params['s'] = s
        hapC = haploid(params)
        hapRes[round(s, 3)] = hapC

    save_pickle(HAPLOID_RES_SUBDIR, haploid_res_filename(), hapRes)


def main():
    parser = argparse.ArgumentParser(
        description="Run simulation trajectories for GD or NGD. Use --h only for diploid GD (single h or omit for all h)."
    )
    parser.add_argument("--model", type=str, required=True, choices=["GD", "NGD"],
        help="GD or NGD (diploid only; haploid always uses NGD)")
    parser.add_argument("--genotype", type=str, required=True, choices=["haploid", "diploid"],
        help="haploid: run gethaploid (NGD); diploid: run getcurves with --model")
    parser.add_argument("--h", type=float, default=None, dest="h",
        help="For diploid GD only: single h value. Omit to run GD simulation for all h (0, 0.1, ..., 1.0). Ignored for NGD/haploid.")
    parser.add_argument("--target_steps", type=int, default=40000,
        help="Maximum simulation steps (default: 40000)")
    parser.add_argument("--q0", type=float, default=0.001,
        help="Initial allele frequency (default: 0.001)")
    args = parser.parse_args()

    params = {
        "target_steps": args.target_steps,
        "q0": args.q0,
    }

    if args.genotype == "haploid":
        params["h"] = args.h
        gethaploid(params)
    elif args.model == "NGD":
        getcurves(params, args.model)
    else:
        # diploid GD: -h optional; if given run for that h only, else run for all h
        h_vals = np.arange(0, 1.01, 0.1)
        if args.h is not None:
            h_vals = [round(float(args.h), 3)]
        for h_val in h_vals:
            h_val = round(float(h_val), 3)
            params["h"] = h_val
            getcurves(params, args.model, h_gd=h_val)
            print(f"GD: saved h{h_val} -> {GD_RES_SUBDIR}/{gd_res_filename(h_val)}")


if __name__ == "__main__":
    main()