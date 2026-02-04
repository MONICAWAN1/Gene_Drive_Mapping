import numpy as np
import math
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import run_model, wm, haploid
from utils import save_pickle

### run simulation for a list of configurations, returns a list of configs and a dictionary {config: curve} #############
def gd_simulation(params):
    '''
    decides what configurations the current simulation set includes
    '''
    minVal, maxVal = 0, 1.1 # gd params all in range [0, 1]
    configs = []
    s_vals = np.arange(0, 1.01, 0.01)
    c_vals = np.arange(0, 1.01, 0.01)
    h_vals = np.arange(0, maxVal, 0.1)

    # get a list of configurations for simulations 
    for s in s_vals:
        if s>0:
            for c in c_vals:
                if c>0:
                    for h in h_vals:
                        if math.isclose(h, params['h']):
                            configs.append((round(float(s), 3), round(float(c), 3), round(float(h), 3)))

    gd_results = dict()
    for (s, c, h) in configs:
        params['s'], params['c'], params['h'] = s, c, h
        gd_res = run_model(params)
        gd_results[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = gd_res

    return configs, gd_results

def getcurves(params):
    '''
    pickle dump ngd results: {(s,h): curve} and gd results [configlist, gd_res] 
    where gd_res: (s,c,h): {'q':, 'state':}
    '''
    ### RUN NON-GD DIPLOID MODEL AND SAVE
    minVal, maxVal, step = 0.01, 5, 0.01
    s_range = np.arange(minVal, maxVal, step)
    hmax = 5
    h_range = np.arange(0, hmax, step)
    wm_results = dict()
    params['q0'] = 0.1

    for s_nat in s_range: 
        for h_nat in h_range: 
            wm_curve = wm(s_nat, h_nat, params['target_steps'], params['q0'])['q']
            if (not math.isclose(wm_curve[-1], 1.0) and (not math.isclose(wm_curve[-1], 0.0))):
                print(s_nat, h_nat)
                wm_results[(round(s_nat, 3), round(h_nat, 3))] = wm_curve
    
    pickle_subdir = "ngd_res"
    save_pickle(pickle_subdir, 'new_allngdres001G_gspos.pickle', wm_results)

    ## RUN GD MODEL AND SAVE
    # gd_configs, gd_res = gd_simulation(params)
    # gd_results = [gd_configs, gd_res]  # gd_configs: list of tuples, gd_res: map {config: curve}
    # print('length of configs', len(gd_configs))
    # save_pickle(f"gd_simulation_results/h{params['h']}_allgdres001G.pickle", gd_results)  ### allgdres001: step = 0.01

def gethaploid(params):
    '''
    run haploid model to get all result in the param space
    '''
    hapRes = dict()
    minVal, maxVal, step = -10, 1, 0.001
    s_range = np.arange(minVal, maxVal, step)
    for s in s_range:
        print(s)
        params['s'] = s
        hapC = haploid(params)
        hapRes[round(s, 3)] = hapC

    subdir = "allhapresults"
    save_pickle(subdir, "allhapres0001G.pickle", hapRes)

colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']

def main():
    params = {'h': 0.7, 'target_steps': 40000, 'q0': 0.001}
    getcurves(params)
    # gethaploid(params)

if __name__ == '__main__':
    main()