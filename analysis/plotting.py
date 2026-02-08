import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib import cm
import seaborn as sns
import math
import pickle
import os, sys
import pandas as pd, statsmodels.formula.api as smf
import sympy as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import run_model, wm, haploid_se, haploid
from pathlib import Path
from utils import export_legend, euclidean, load_pickle, save_pickle, PICKLE_DIR, GD_RES_SUBDIR, gd_res_filename
from .mapping import get_eq
from .regime import q3_lambda, derivative_q3_lambda, get_stability_table, get_regime

def loadGrad():
    return load_pickle("", "h0.0_hap_gradient_G_fix.pickle")

colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']

diffmap5 = load_pickle("mapping_diff", "h0.0_mappingdiff_hap_grid.pickle")
diffmap6 = load_pickle("mapping_diff", "h0.0_mappingdiff_gdhapse01.pickle")
# print(diffmap1, '\n', diffmap2)

# BuPu Color Scale
all_values = list(diffmap5.values()) + list(diffmap6.values()) + [0.011150530195789305]
min_val = min(all_values)
max_val = max(all_values)
# print(min_val, max_val)
allmax = 0.1

'''
Plot error vs h for a specific configuration
'''
def plot_errorh(h):
    hapse_errors = []
    hapgrid_errors = []
    s = 0.2
    c = 0.9
    h_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
    for h in h_range:
        hapseDiff = load_pickle("mapping_diff", f"h{h}_mappingdiff_gdhapse001.pickle")
        hapgridDiff = load_pickle("mapping_diff", f"h{h}_mappingdiff_hap_grid.pickle")
        # print(list(hapseDiff.keys()))
        hapse_errors.append(hapseDiff[(s, c, h)])
        hapgrid_errors.append(hapgridDiff[(s, c, h)])

    
    plt.figure(figsize=(8, 6))
    plt.plot(h_range, hapse_errors, label = "Haploid_Se Model")
    plt.plot(h_range, hapgrid_errors, label = "Grid Search Haploid")

    plt.xlabel('h', fontsize = 14)
    plt.ylabel('MSE between Gene-Drive curve and Mapped curve', fontsize = 14)
    plt.title(f'Change in Mapping Error over H at S = {s}, C = {c}')
    plt.legend(title='Mapping Method', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.tight_layout()

    plt.show()
        

### plot simple dynamics for gd and non gd ################
def plot_ngd(s, h):
    # with open('pickle/allngdres.pickle', 'rb') as f1:
    #     ngd_res = pickle.load(f1)
    # params = {'s': 0.4, 'c': 0.4, 'h': 0.5, 'target_steps': 100, 'q0': 0.1}
    # res = run_model(params)
    res = wm({'s': s, 'h': h, 'target_steps': 40000, 'q0': 0.2})
    mut = res['q']
    regime = get_regime("NGD", s, h)
    print(h, mut, regime)
    # plt.plot(wt, color = 'orange', label = 'wild-type')
    plt.plot(np.arange(0, len(mut)), mut, color = 'blue', label = 'mutant')
    # plt.ylabel('Allele Frequency')
    # plt.xlabel('Time')
    # plt.title('Allele frequency dynamics in non-gene drive population')
    # plt.grid(True)
    # plt.legend(title='Allele', bbox_to_anchor=(0.8, 0.5), loc='center left')
    # plt.show()

def plot_ngds(s:float):
    # h_range = np.arange(0, 1, 0.1)
    h_range = [2.375]
    for h in h_range:
        plot_ngd(s, h)
    plt.ylabel('Allele Frequency')
    plt.xlabel('Time')
    plt.title('Allele frequency dynamics in non-gene drive population')
    plt.grid(True)
    plt.legend(title='Allele', bbox_to_anchor=(0.8, 0.5), loc='center left')
    plt.show()

# plot_ngds(-2.02)

def plot_gd(ts, tc):
    q_init = 0.2
    h = 0.8
    colormaps = ['Greys', 'Reds', 'YlOrBr', 'Oranges', 'PuRd', 'BuPu',
                      'GnBu', 'YlGnBu', 'PuBuGn', 'Greens']
    gd_result = load_pickle(GD_RES_SUBDIR, gd_res_filename(h))
    configs, res = gd_result[0], gd_result[1]
    # ngd_results1 = load_pickle(f"new_allngdres001G_h5.pickle")
    # ngd_results2 = load_pickle(f"new_allngdres001G_h.pickle")
    # ngd_results = ngd_results1 | ngd_results2
    # print(ngd_results)
    # print(configs)
    # print(res)

    plt.figure(figsize = (15, 7))
    for i in range(len(configs)):
        s, c, h = configs[i]
    
        # if len(res[(s, c, h)]['q']) < 1500 and not math.isclose(res[(s, c, h)]['q'][-1], 1.0) and res[(s, c, h)]['q'][-1] < 0.1:
        if math.isclose(s, ts) and math.isclose(c, tc):
            gd_curve = run_model({'s': s, 'c': c, 'h': h, 'target_steps': 40000, 'q0': q_init})['q']
            cmap1 = plt.get_cmap(colormaps[int(s*10-1)])
            gd_color = cmap1(c)
            gd_color = 'b'
            # time = np.arange(len(res[(s, c, h)]['q']))
            time = np.arange(len(gd_curve))

            # print((s, c, h))
            # print(res[(s, c, h)]['q'])
            # s_range = np.arange(-10, 0, 1)
            # h_range = np.arange(0, 20, 1)
            # for ngds in s_range:
            #     for ngdh in h_range:
            #         ngd_curve = wm(ngds, ngdh, 40000, q_init)['q']
            #         s_norm = (ngds + 10) /len(s_range)  # Normalize s: -5 -> 0, -1 -> 1
            #         h_norm = (ngdh) /len(h_range)        # Normalize h: 0 -> 0, 9 -> 1
            #         color = (s_norm, h_norm, 0.5)  # fixed blue component (or vary as desired)
            #         plt.plot(np.arange(1, len(ngd_curve)+1), ngd_curve, color = color, label = f"NGD s = {ngds}, h = {ngdh}")
        # gd_color = cmap(1.*i/len(configs))

            plt.plot(time, gd_curve, color = gd_color, label = f"s = {s}, c = {c}, h = {h}")
    plt.ylabel('Gene Drive Allele Frequency')
    plt.xlabel('Time')
    plt.title("Change in Mutant Allele Frequency")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.grid(True)
    plt.legend(title='population condition', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.show()

# plot_gd(0.4, 0.4)

def delta_curve(curve):
    first_d = []
    for t in range(len(curve)-1):
        delta = curve[t+1] - curve[t]
        first_d.append(delta)
    return first_d

def derivative_plot(params, gd_configs, fitting_res):
    plt.figure(figsize=(10,7))
    for (s, c, h) in gd_configs:
        params['s'] = s
        params['c'] = c
        params['h'] = h
        res = run_model(params)
        gd_allele, wt_allele = res['q'], res['p']
    
        d1 = delta_curve(gd_allele)

        second_d1 = delta_curve(d1)
        cmap1 = plt.get_cmap('BuPu')
        gd_color = cmap1(c)
        time = list(range(params['target_steps']-2))
        plt.plot(time, second_d1, color = gd_color, label = f'gene drive s={round(s, 3)}, c={round(c, 3)}, h={round(h, 3)}')

        fitting_s = fitting_res[(s, c, h)][0]
        for h in np.arange(0, 1, 0.1):
            ngd_curve = wm({'s': fitting_s, 'h': h, 'target_steps': params['target_steps'], 'q0': params['q0']})['q']
            ngd_d1 = delta_curve(ngd_curve)
            ngd_d2 = delta_curve(ngd_d1)

            cmap2 = plt.get_cmap('Oranges')
            ngd_color = cmap2(h)
            time2 = time[:len(ngd_d2)]
        # plt.plot(time, ngd_d2, color = ngd_color, label = f'c={round(c, 3)}, s_eff={fitting_s}')
            plt.plot(time2, ngd_d2, color = ngd_color, label = f'no gene drive s={round(float(fitting_s), 3)}, h={round(h, 3)}')
    
    # plt.plot(time, wt_allele, color = 'orange')

    plt.legend(title='Parameters', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.xlabel('time', fontsize = 15)
    plt.ylabel('second derivative of gene-drive allele curve', fontsize = 15)
    plt.show()


def getHapseMapDiff(currH):
    '''
    Given current h value, plot an Error heatmap for haploid_se vs GD
    Store results as .txt under analysis/mapping_diff_txt/ and as .pickle under mapping_diff/
    '''
    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH))
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = get_stability_table(currH)

    # Name output files (txt + pickle)
    is001 = "001" if '001' in loadFile_name else '01'
    outTXTFile = f"haploid_mapping_diff_txt/h{currH}_mappingdiff_gdhapse{is001}_fix.txt"
    savedPickle_subdir = "mapping_diff"
    savedPickle_name = f"h{currH}_mappingdiff_gdhapse{is001}_fix.pickle"

    gds = [res['q'] for res in gd_res.values()]

    valid_configs = []
    for (s, c, h) in gd_configs:
        if ((s, c, h), 1.0) in stabilityRes['fixation']: ### NEED TO USE ANALYTICAL BOUNDS
        # if gd_res[(s, c, h)]['state'] == 'loss':
            valid_configs.append((s, c, h))

    saved = dict()

    config_index = {conf: i for i, conf in enumerate(gd_res.keys())}

    for (s, c, h) in valid_configs:

        gd_curve = gd_res[(s,c,h)]['q']

        # plot haploid using Se
        se = h*s-c+c*h*s
        paramSe = {'s':s, 'c':c, 'n': 500, 'h':h, 'target_steps': 40000, 'q0': 0.001}
        hapSe = haploid_se(paramSe)['q']

        curveDiff = euclidean(gd_curve, hapSe)
        saved[(s, c, h)] = curveDiff
        ### DEBUGGING
        # if (math.isclose(s, 0.8) and math.isclose(c,0.9)):
        #     with open("s0.8_c0.9_hapse_diff.txt", 'w') as fout:
        #         fout.write(f"gd: {gd_curve}\nhapse: {hapSe}\nDiff:{curveDiff}")
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(np.arange(len(hapSe)), hapSe, label = 'hapse')
        #     plt.plot(np.arange(len(gd_curve)), gd_curve,label = 'gd')
        #     plt.legend(title='Model', bbox_to_anchor=(1, 1.05), loc='upper left')
        #     plt.show()

    #write the diff to a txt file
    with open(outTXTFile, 'w') as fout:
        fout.write(f"Configuration\tHapse Mapping Diff\n")
        for key, diff in saved.items():
            fout.write(f"{key}\t{diff}\n")   

    save_pickle(savedPickle_subdir, savedPickle_name, saved)

def plotMapDiff(currH):
    '''
    Given current h value, plot an Error heatmap for haploid_se vs GD
    '''
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text remains text
    plt.rcParams['ps.fonttype'] = 42
    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH))
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = get_stability_table(currH)
    hapseDiffMap = load_pickle("mapping_diff", f"h{currH}_mappingdiff_gdhapse001_fix.pickle")

    valid_configs = []
    for (s, c, h) in gd_configs:
        if ((s, c, h), 1.0) in stabilityRes['fixation']: ### NEED TO USE ANALYTICAL BOUNDS
        # if gd_res[(s, c, h)]['state'] == 'fix':
            valid_configs.append((s, c, h))

    all_s = sorted(set(s for (s,c,h) in valid_configs))
    all_c = sorted(set(c for (s,c,h) in valid_configs))

    diffmap_name = f"h{currH}_mappingdiff_gdhapse001_fix.pickle"
    if '001' in diffmap_name:
        step = 0.01
    else:
        step = 0.1
    axis_len = int(1/step)

    diff_map = np.full((axis_len, axis_len), np.nan)

    # diff_map = np.full((len(all_s), len(all_c)), np.nan)

    for (s, c, h) in valid_configs:
        # The index of this config in wms
        row_idx = all_s.index(s)
        col_idx = all_c.index(c)
        diff_map[row_idx, col_idx] = hapseDiffMap[(s,c,h)]

    X, Y = np.meshgrid(all_c, all_s)
    import matplotlib.colors as mcolors

    def truncate_colormap(cmap, minval=0.1, maxval=0.7, n=256):
        new_colors = cmap(np.linspace(minval, maxval, n))
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            f'truncated({cmap.name},{minval:.2f},{maxval:.2f})', new_colors)
        return new_cmap

    base_cmap = plt.get_cmap('Reds')
    cmap = truncate_colormap(base_cmap, 0.1, 0.7)
    cmap.set_bad('white')

    # Define min and max values for color normalization
    vmin, vmax = min_val, max_val
    vmin2, vmax2 = np.nanmin(diff_map), np.nanmax(diff_map)
    print(vmin2, vmax2)
    vmax2 = 0.01
    norm = mcolors.Normalize(vmin=vmin2, vmax=vmax2)

    plt.figure(figsize=(9, 7))
    plt.imshow(
        diff_map,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        extent=[0, 1, 0, 1],  # fixes axes to 0–1
        vmin=vmin2,
        vmax=vmax2
    )

    plt.colorbar(label='Mean Squared Error (Error of mapping)')

    tick_vals = np.round(np.arange(0, 1.01, 0.1), 2)
    plt.xticks(tick_vals)
    plt.yticks(tick_vals)

    plt.xlabel('c in gene drive', fontsize=15)
    plt.ylabel('s in gene drive', fontsize=15)
    plt.title(f'Difference Heatmap: GD vs. Haploid Se at h = {currH}')
    plt.savefig(f"mapping_diff_figures/h{currH}_gdhapse001_fix_md_001.pdf", format="pdf", dpi=600, bbox_inches='tight')
    plt.show()


'''
Plot Diploid Analytic mapping difference
'''
def plot_error_da(currH):
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text remains text
    plt.rcParams['ps.fonttype'] = 42


    # plot error of fixation regime mapping 
    # get gd config and then get mapdiff errror
    # Get gd config in fixation
    s_ngd, h_ngd = solve_sngd(s, c, h)
    solved_curve = wm({'s': s_ngd, 'h': h_ngd, 'target_steps': 40000, 'q0': q_init})['q']
    time_solved = np.arange(0, len(solved_curve))
    mse = euclidean(solved_curve, gd_curve)
    plt.figure(figsize=(9, 7))
    plt.imshow(
        diff_map,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        extent=[0, 1, 0, 1],  # fixes axes to 0–1
        vmin=min_val,
        vmax=max_val
    )
    

'''
Partition plot of final state based on simulation
partition graph: fixed h, x is c, y is s, color indicates s-eff/final state in gd simulation
'''
def partition():
    # loading
    # with open('pickle/sch_to_s_results.pickle', 'rb') as f:
    #     map_results = pickle.load(f)
    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(0.5))
    gd_configs, gd_res = gd_results[0], gd_results[1]

    # seffmap, ngdcurves = map_results['map'], map_results['ngC']
    slist = sorted(set([conf[0] for conf in gd_configs]))
    clist = sorted(set([conf[1] for conf in gd_configs]))
    s_effs = []
    configurations = []
    states = []
    finals = []
    statemap = {'fix': 2.5, 'loss': 0.5, 'stable': 1.5, 'unstable': 1.0}
    h = 0.5 # need to change this!
    for s in slist:
        for c in clist:
            configurations.append((s, c))
            res = gd_res[(s, c, h)]
            state = get_regime("GD", s, h, c=c)
            state = 'fix' if state == 'fixation' else (state or 'stable')
            states.append(statemap.get(state, 1.5))
            finals.append(res['q'][-1])

    print(states)
    s_values = [conf[0] for conf in configurations]
    c_values = [conf[1] for conf in configurations]

    # print(len(s_effs))
    # print(len(configurations))
    
    # Normalize S-effective values to map to colors
    norm = mcolors.Normalize(min(finals), max(finals))
    bounds = [0, 1, 2, 3]
    # cmap = plt.cm.get_cmap('Purples', len(bounds) - 1)
    cmap = plt.cm.get_cmap('viridis')
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    scatter = ax.scatter(c_values, s_values, c=finals, cmap=cmap, s=50, norm=norm) 

    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[0, 1, 2, 3])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Final Gene Drive Allele Frequency')
    # cbar.set_ticklabels(['', 'Loss', 'Equilibrium', 'Fixation'])

    ax.set_ylabel('Selection Coefficient (s)')
    ax.set_xlabel('Conversion Factor (c)')
    ax.set_title(f'Partition of Gene Drive Configurations by final gene drive allele frequency with H = {h}')

    # Show plot
    plt.show()

def get_solved_curve(s, c, h, currH):
    solved_res = load_pickle("anasim_mapdiff", f"solved_res_h{currH}.pickle")
    solved_params, solved_curve = solved_res[(s, c, h)]
    return solved_params, solved_curve


'''
mapping curves of ngd and gd populations for a single configuration
Need to change the loaded files according to plotted curves
'''
def plot_mapping(currH):
    # loading results
    diploid = True
    gradient = False
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text remains text
    plt.rcParams['ps.fonttype'] = 42
    mapfunction = 'grid' if diploid else 'hap_grid'
    if gradient: 
        gradResult = load_pickle("", f"h{currH}_hap_gradient_G_fix.pickle") # all gradient results
        sMap_grad, wms_grad = gradResult['map'], gradResult['ngC']
    
    if not gradient: 
        if diploid: 
            # gridResult_diploid = load_pickle(...) # all hap grid results
            # sMap_grid_diploid, wms_grid_diploid = gridResult_diploid['map'], gridResult_diploid['ngC']

            sMap_grid_diploid = load_pickle("stable_mapping_result", f"h{currH}_grid_stable001_G.pickle")
            print(sMap_grid_diploid)
        else: 
            gridResult = load_pickle("mapping_result", f"h{currH}_hap_grid001_G_fix.pickle")
            sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']
    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH))

    gd_configs, gd_res = gd_results[0], gd_results[1]


    plt.figure(figsize = (8, 6))
    ts = 0.85
    tc = 0.57
    q_init = 0.01

    for (s, c, h) in gd_configs:
        if (math.isclose(s, ts) and math.isclose(c, tc)):
            gd_param = {'s': s, 'h':h, 'c': c, 'q0': q_init, 'target_steps': 40000}
            gd_curve = run_model(gd_param)['q']
            time = np.arange(0, len(gd_curve))
        # if s < c and math.isclose(s, 0.1):
            ### PLOT SOLVED DIPLOID NGD
            # solved_params, solved_curve = get_solved_curve(s, c, h, currH)
            # solved_s, solved_h = solved_params 
            # plt.plot(solved_curve, color = '#d16817', marker = 's', markersize = '2', label = f'solved curve ngd_s={solved_s:.4f}, ngd_h={solved_h:.4f}')
            # haploid trajectory
            if not diploid and (s, c, h) in sMap_grid and type(sMap_grid[(s,c,h)]) != tuple:
                best_s_grid = sMap_grid[(s, c, h)]
                i = list(sMap_grid.keys()).index((s, c, h))
                wm_curve_grid = wms_grid[i]
                time0 = np.arange(0, len(wm_curve_grid))
                w_color1 = 'y'
                plt.plot(time0, wm_curve_grid, marker = 'o', color = w_color1, markersize=3, linestyle = '-', label = f'grid search haploid NGD (s = {best_s_grid})')
    
            # diploid trajectory
            if diploid and (s, c, h) in sMap_grid_diploid:
                # print("TRUEE")
                if "unstable" not in gridres_filename:
                    best_s_grid, best_h_grid = sMap_grid_diploid[(s, c, h)][0], sMap_grid_diploid[(s, c, h)][1]
                else:
                    best_s_grid, best_h_grid = sMap_grid_diploid[0], sMap_grid_diploid[1]

                cmap = plt.get_cmap('GnBu')
                # w_color = cmap(1.*i/len(h_range))
                w_color0 = '#0e169e'
                # i = list(sMap_grid_diploid.keys()).index((s, c, h))
                wm_curve_grid = wm({'s': best_s_grid, 'h': best_h_grid, 'target_steps': 40000, 'q0': q_init})['q']
                time1 = np.arange(0, len(wm_curve_grid))
                plt.plot(time1, wm_curve_grid, marker = 'o', color = w_color0, markersize=2, linestyle = '-', label = f'grid search diploid NGD (s = {best_s_grid:.3f}, h = {best_h_grid:.3f})')

                ### plot analytics
                s_ngd, h_ngd = solve_sngd(s, c, h)
                solved_curve = wm({'s': s_ngd, 'h': h_ngd, 'target_steps': 40000, 'q0': q_init})['q']
                time_solved = np.arange(0, len(solved_curve))
                mse = euclidean(solved_curve, gd_curve)
                plt.plot(time_solved, solved_curve, color = w_color0, markersize=2, linestyle = '--', label = f'Analytical diploid NGD (s = {s_ngd:.3f}, h = {h_ngd:.3f}), mse={mse:.4f}')


            # PLOT GRADIENT (HAP/DIPLOID)
            if gradient:
                print("Plotting gradient descent curves")
                cmap2 = plt.get_cmap('GnBu') # grid mapping curves
                w_color2 = cmap2(abs(best_s_grad))
                w_color2 = "b"
                if not diploid: 
                    if(s, c, h) in sMap_grad and type(sMap_grad[(s,c,h)]) != tuple:
                        best_s_grad = sMap_grad[(s, c, h)]
                        paramHap = {'s':best_s_grad, 'n': 500, 'target_steps': 40000, 'q0': 0.001}
                        wm_curve_hap_grad = haploid(paramHap)['q']
                        time4 = np.arange(0, len(wm_curve_hap_grad))
                        plt.plot(time4, wm_curve_hap_grad, marker = 's', color = w_color2, markersize=3, linestyle = '-', label = f'gradient descent haploid NGD s = {best_s_grad}')
                        # best_h_grid = sMap_grid[(s, c, h)][1]
                elif diploid:
                    best_s_grad, best_h_grad = sMap_grad[(s, c, h)][0], sMap_grad[(s, c, h)][1]
                    print("BEFORE", best_s_grad, best_h_grad)
                    # best_s_grad, best_h_grad = -1.33, 1
                    wm_curve_grad = wm({'s': best_s_grad, 'h': best_h_grad, 'target_steps': 40000, 'q0': 0.001})['q']
                    time2 = np.arange(0, len(wm_curve_grad))
                    error = euclidean(gd_res[(s, c, h)]['q'], wm_curve_grad)
                    print('ERROR', error)
                    plt.plot(time2, wm_curve_grad, marker = 's', color = w_color2, markersize=3, linestyle = '-', label = f'gradient descent diploid NGD s = {best_s_grad}, h = {best_h_grad}')
                print(best_s_grad, best_h_grad)
            
            # print("AFTER", best_s_grad, best_h_grad)
            # print("best_s_grad", best_s_grad, "best_h_grad", best_h_grad) 

            # PLOT GD CURVE
            cmap = plt.get_cmap("Reds")
            gd_color = "#ca1f8e" # original gene-drive curve

            # PLOT HAPSE
            paramSe = {'s':s, 'c':c, 'n': 500, 'h': h, 'target_steps': 40000, 'q0': 0.001}
            se = h*s-c+c*h*s
            hapSe = haploid_se(paramSe)['q']
            cmapSe = plt.get_cmap("Oranges")
            se_color = cmapSe(0.75) # original gene-drive curve

            # nearby = []
            # for hnew in np.arange(best_h_grid-0.2, best_h_grid+0.3, 0.1):
            #     newc = wm(best_s_grid, hnew, params['target_steps'], params['q0'])['q']
            #     nearby.append(newc)
            #     time = np.arange(0, len(newc))
            #     h_color = cmap2(abs(hnew-best_h_grid)*4)
            #     plt.plot(time, newc, marker = 'o', color = h_color, markersize=3, linestyle = '-', label = f"NGD s = {'%.3f' % best_s_grid}, h = {'%.3f' % hnew}")      

            # time = np.arange(0, len(gd_res[(s, c, h)]['q']))
            time_se = np.arange(0, len(hapSe))
            time3 = np.arange(0, len(gd_res[(s, c, h)]['q'])-1)

            print("plotting gd and hapse curves")
            plt.plot(time, gd_curve, color = gd_color, label = f"s = {s}, c = {c}, h = {h}")
            # plt.plot(time_se, hapSe, marker = 'o', color = se_color, markersize=3, label = f"haploid using Se (s = {'%.3f'%se})")
            # plt.plot(time3, gd_res[(s, c, h)]['w_bar'], color = 'b', label = f"wbar for s = {s}, c = {c}, h = {h}")


            # PLOTTING NGD CURVE FOR GRADIENT GRAPH
            # mapdiffFile = f"s{s}_c{c}_h{currH}_gd_to_ngd_diff.pickle"
            # gd_ngd_diff = load_pickle(mapdiffFile)
            # ngd_s, ngd_h = best_s_grad, best_h_grad
            # error = gd_ngd_diff[(ngd_s, ngd_h)]
            # ngd_curve = wm(ngd_s, ngd_h, 40000, 0.0001)['q']
            # time_ngd = np.arange(0, len(ngd_curve))
            # plt.plot(time_ngd, ngd_curve, color = '#6666FF', label = f"NGD with s={ngd_s}, h={ngd_h}, error={'%.3f'%error}")
    # print(euclidean(gd_res[(s, c, h)]['q'], wm_curve_grid))
    plt.ylabel('Gene Drive/Mutant Allele Frequency', fontsize=12)
    plt.xlabel('Time', fontsize=12)
    plt.title(f"Comparison of GD and different mapping results at h = {currH}, q_init = {q_init}")
    # plt.tight_layout(rect=[0, 0, 0.7, 1])
    plt.grid(False)
    # legend = plt.legend(title='population condition', loc='upper left', bbox_to_anchor=(0,1))
    legend = plt.legend(title='population condition', loc='lower right', bbox_to_anchor=(1, 0))
    export_legend(legend)
    plt.savefig(f"trajectory_plot/h{currH}_{mapfunction}_s{ts}_c{tc}_q{q_init}.jpg", dpi=600, bbox_inches='tight')
    # plt.savefig(f"trajectory_plot/h{currH}_{mapfunction}_s{ts}_c{tc}.pdf", format="pdf", dpi=600, bbox_inches='tight')
    plt.show()



def ploterror(currH):
    '''
    For fixation or unstable regime, read the grid mapping difference file
    Plot the error heatmap
    '''
    mfunc = 'hap_grid'  # 'hap_grid'/'grid'/'analytic'
    state = "fixation"  # or 'unstable'
    if state == 'unstable':
        diffmap_subdir = "unstable_mapping_results"
        diffmap_name = f"h{currH}_mappingdiff_grid_unstable.pickle"
        figure_file = f"unstable_mapping_v2/h{currH}_maperror_unstable.pdf"
    elif state == 'stable':
        diffmap_subdir = "stable_mapping_results"
        diffmap_name = f"h{currH}_mappingdiff_grid_stable.pickle"
        figure_file = f"stable_mapping_rf/h{currH}_maperror_stable.pdf"
    elif state == "fixation":
        if "grid" in mfunc:
            diffmap_subdir = "mapping_diff"
            diffmap_name = f"h{currH}_mappingdiff_{mfunc}_{state}001_G.pickle"
            figure_file = f"mapping_diff_figures/h{currH}_mappingdiff_{mfunc}_{state}_001.pdf"
        else:
            diffmap_subdir = "anasim_mapdiff"
            diffmap_name = f"h{currH}_fixation_analytical_mapdiff_q001.pickle"
            figure_file = f"mapping_diff_figures/h{currH}_mappingdiff_{mfunc}_fix_001.pdf"
    diffmap = load_pickle(diffmap_subdir, diffmap_name)
    print(len(diffmap))
    # stability = load_pickle(f"h{currH}_gametic_stability_res.pickle")

    ngd_s = sorted(set([conf[0] for conf in diffmap.keys()]))
    ngd_c = sorted(set([conf[1] for conf in diffmap.keys()]))
    print(len(ngd_s), len(ngd_c))

    if '001' in diffmap_name:
        step = 0.01
    else:
        step = 0.1
    axis_len = int(1/step)

    errors = np.full((axis_len, axis_len), np.nan)

    for (s, c, h), error in diffmap.items():
        i = ngd_s.index(s)  # row (y-axis)
        j = ngd_c.index(c)  # column (x-axis)
        errors[i, j] = error

    cmin, cmax = np.nanmin(errors), 0.01
    print(cmin, cmax)
    # cmax = allmax
    import matplotlib.colors as mcolors

    def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
        new_colors = cmap(np.linspace(minval, maxval, n))
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            f'truncated({cmap.name},{minval:.2f},{maxval:.2f})', new_colors)
        return new_cmap

    base_cmap = plt.get_cmap('Reds')
    cmap = truncate_colormap(base_cmap, 0.1, 0.7)

    plt.figure(figsize=(9, 7))
    plt.imshow(
        errors,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        extent=[0, 1, 0, 1], 
        vmin=cmin,
        vmax=cmax
    )

    plt.colorbar(label='Mean Squared Error (Error of mapping)')

    # Set ticks at intervals of 0.1 and label them with the actual values of s and c
    tick_vals = np.round(np.arange(0, 1.01, 0.1), 2)
    plt.xticks(tick_vals)
    plt.yticks(tick_vals)

    plt.xlabel('c in gene drive', fontsize=15)
    plt.ylabel('s in gene drive', fontsize=15)
    plt.title(f"Mapping Error from GD to NGD at h = {currH} with {mfunc} for {state} Regime")
    plt.savefig(figure_file, dpi=600, format="pdf", bbox_inches='tight')
    plt.show()

def gd_to_ngd_diff(currH):
    gd_s = 0.2
    gd_c = 0.9
    gd_h = 0.0
    mapdiff_subdir = ""
    mapdiff_name = f"s{gd_s}_c{gd_c}_h{currH}_gd_to_ngd_diff.pickle"
    # mappingdiff: (s,c,h): error of map

    ### CALUCLATING DIFFERENCES AND SAVING TO PICKLE ###
    # loadFile = f"gd_simulation_results/h{currH}_allgdresG.pickle"
    # gd_results = load_pickle(loadFile)
    # gd_configs, gd_res = gd_results[0], gd_results[1]
    # ngdRes = load_pickle("allngdres001G.pickle")

    # gd_curve = gd_res[(gd_s, gd_c, gd_h)]['q']

    # diffmap = dict()

    # for ngd_config, ngd_curve in ngdRes.items():
    #     diff = euclidean(ngd_curve, gd_curve)
    #     diffmap[ngd_config] = diff

    # save_pickle(mapdiffFile, diffmap)
    ### COMMENT OUT EXCEPT FOR FIRST RUN ###############

    diffmap = load_pickle(mapdiff_subdir, mapdiff_name)
    
    traceFile_name = f"s{gd_s}_c{gd_c}_h{currH}_gradient_trace_r.pickle"
    traces = load_pickle("gradient_traces", traceFile_name)
    s_traces = [trace[0] for trace in traces]
    h_traces = [trace[1] for trace in traces]
    # print(diffmap)
    ngd_s = sorted(set([conf[0] for conf in diffmap.keys()]))
    ngd_h = sorted(set([conf[1] for conf in diffmap.keys()]))

    # print(ngd_h, len(ngd_s), len(ngd_h))
    errors = np.zeros((len(ngd_h), len(ngd_s)))

    for (s, h), error in diffmap.items():
        i = ngd_h.index(h)  # Row index (h axis)
        j = ngd_s.index(s)  # Column index (s axis)
        errors[i, j] = error
    
    print(errors)

    gridResult_diploid = load_pickle("", f"h{currH}_grid_G_fix.pickle") # all hap grid results
    gridMap = gridResult_diploid['map']
    gridS, gridH = gridMap[(gd_s, gd_c, gd_h)][0], gridMap[(gd_s, gd_c, gd_h)][1]

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(errors, aspect='auto', cmap='BuPu', origin='lower',
               extent=[min(ngd_s), max(ngd_s), min(ngd_h), max(ngd_h)], vmax = 0.2, vmin = 0)
    plt.colorbar(label='Euclidean Distance (Error of mapping)')
    plt.plot(s_traces, h_traces, marker = 'o', color = '#cc99ff')
    plt.scatter([gridS], [gridH], marker = 'X', color = '#ff6666')
    plt.xlabel('s in non-gene-drive')
    plt.ylabel('h in non-gene-drive')
    plt.title(f"Gradient Map Trace at Gene-Drive config {(gd_s, gd_c, gd_h)} on GD-NGD Error Heatmap")
    plt.show()


def plot_gdtn(params):   # the dense mesh figure
    # mapping_result = mapping(params)
    with open('pickle/sch_to_s_results.pickle', 'rb') as f:
        map_results = pickle.load(f)
    
    with open('pickle/gd_simresults.pickle', 'rb') as f2:
        gd_results = pickle.load(f2)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    seffmap, ngdcurves = map_results['map'], map_results['ngC']
    print(seffmap)
    gd_configs = seffmap.keys()
    slist = sorted(set([conf[0] for conf in gd_configs]))
    clist = sorted(set([conf[1] for conf in gd_configs]))
    hlist = sorted(set([conf[2] for conf in gd_configs]))
    seff = [seffmap[config] for config in gd_configs]
    plt.figure(figsize = (15, 7))
    cmap = plt.get_cmap('BuPu')
    for s in slist:
        sefflist = [seffmap[(s,c,0.5)][0] for c in clist]
        # print(sefflist)
        cvcolor = cmap(s)
        plt.plot(clist, sefflist, marker = 'o', linestyle='-', color = cvcolor, label = f's = {s}')
    plt.ylabel('S_effective in non-gene-drive population')
    plt.xlabel('C in gene drive population')
    plt.legend(title='S in gene drive population', bbox_to_anchor=(1, 1.05), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.title("Map from gene drive configuration to effective S in non-gene-drive population (h = 0.5)")
    plt.show()


def plot_msedl(gd_config):
    output_folder = "gd_candidates"
    s_mse_map = load_pickle(output_folder, f"gd{gd_config}_stable_finds.pickle")

    s_vals = sorted(s_mse_map.keys())
    mse_vals = [s_mse_map[s]['MSE'] for s in s_vals]
    dl_vals = [s_mse_map[s]['dl'] for s in s_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(s_vals, mse_vals, label="MSE", marker='o')
    plt.plot(s_vals, dl_vals, label="Delta Lambda", marker='x')

    plt.xlabel("s_ngd")
    plt.ylabel("Metric Value")
    plt.title(f"MSE and Delta Lambda vs s_ngd at mapping for GD config: {gd_config}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


'''
Plot eq & lambda vs s_ngd
'''
def plot_eq_lambda(gd_config):
    output_folder = "gd_candidates"
    s_mse_map = load_pickle(output_folder, f"gd{gd_config}_stable_finds.pickle")

    s_vals = sorted(s_mse_map.keys())
    eq_vals = [s_mse_map[s]['eq'] for s in s_vals]
    lambda_vals = [s_mse_map[s]['lambda'] for s in s_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(s_vals, eq_vals, label="Simulation Eq", marker='o')
    plt.plot(s_vals, lambda_vals, label="Lambda", marker='x')

    plt.xlabel("s_ngd")
    plt.ylabel("Metric Value")
    plt.title(f"Equilibrium and Stability vs s_ngd at mapping for GD config: {gd_config}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lambda_curve(h, q, gd_config):
    # dfunc = compute_lambda()
    # lambdas= []
    # s_range = np.arange(-10, 10, 0.01)
    # for s in s_range:
    #     l = get_ngd_stability(s, h, q, dfunc)
    #     lambdas.append(l)
    
    # plt.figure(figsize=(8, 6))
    # plt.plot(s_range, lambdas, label=f"dq at equilibrium with h_ngd = {h:.3f}", marker='o')

    plt.xlabel("s_ngd")
    plt.ylabel("Lambda (dq)")
    plt.title(f"dq vs s_ngd in mapping for GD config {gd_config}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from collections import defaultdict

def load_mapping_unstable_results(filepath):
    curves = dict()
    mapped = dict()
    current_gd = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("Gene Drive Configuration") or line.startswith("(s, h)"):
                continue  # skip headers

            parts = line.split()
            print(parts)
            if line.startswith('s='):  # gene drive line
                # Extract gene drive parameters
                gd_s = float(parts[0].split('=')[1].strip(','))
                gd_c = float(parts[1].split('=')[1].strip(','))
                gd_h = float(parts[2].split('=')[1])

                # Extract NGD parameters
                q_init = float(parts[3].split('=')[1].strip(','))
                ngd_s = float(parts[4].split('=')[1].strip(','))
                ngd_h = float(parts[5].split('=')[1].strip(','))

                current_gd = (gd_s, gd_c, gd_h, q_init)
                curves[current_gd]=(q_init, ngd_s, ngd_h)
                mapped[q_init] = (ngd_s, ngd_h)
    return current_gd, curves, mapped

def load_mapping_unstable_2(filepath, gd_config):
    subdir, filename = os.path.split(filepath)
    mapres = load_pickle(subdir, filename)
    mapped_config = mapres[gd_config][:2]
    return mapped_config

def plot_sngd_q(mapped, currH, current_gd):
    # Plotting mapped s_ngd vs q_init
    gd_s, gd_c, gd_h = current_gd
    plt.figure(figsize=(10, 6))
    q_values = list(mapped.keys())
    s_ngd_values = [item[0] for item in mapped.values()]
    plt.plot(q_values, s_ngd_values, marker='o', linestyle='-', markersize=3, color='blue', label=f's_ngd vs q_init at gd_s={gd_s}, gd_c={gd_c}, gd_h={gd_h})')
    plt.xlabel('q_init')
    plt.ylabel('s_ngd')
    plt.title(f'Mapped s_ngd vs q_init for h = {currH}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unstable_plots/h{currH}_s{current_gd[0]}_c{current_gd[1]}_mapped_s_ngd.jpg", dpi=600)
    plt.show()

''''
Plot the mapped GD and NGD curves with based on a set of q_init for unstable regime
'''
def plot_qmaps(currH):
    curves = dict()
    ts = 0.6
    tc = 0.9
    current_gd = (ts, tc, float(currH))
    mapped = dict()
    version = 2  # version of the mapping results
    plot_solve = False
    solved = "analytical" if plot_solve else "grid"
    plt.figure(figsize=(10, 6))
    cmap1 = cm.get_cmap('viridis')
    gd_param = {'s': current_gd[0], 'c': current_gd[1], 'h': current_gd[2], 'conversion': "gametic", 'q0': 0.1, 'target_steps': 40000}

    # version 1 of unstable mapping results are stored in txt 
    if version == 1:
        filepath = f"unstable_mapping/h{currH}_s{ts}_c{tc}_grid_G_unstable.txt"
        current_gd, curves, mapped = load_mapping_unstable_results(filepath)
        q_values = [item[3] for item in curves.keys()]
        norm = Normalize(vmin=min(q_values), vmax=max(q_values))


        ngd_s = curves[current_gd][1]
        ngd_h = curves[current_gd][2]

        # Plotting ##########################
        for i, (gd_params, data_points) in enumerate(curves.items()):
            print(gd_params, data_points)
            gd_s, gd_c, gd_h, q_init = gd_params
            gd_param = {'s': gd_s, 'h':gd_h, 'c': gd_c, 'q0': q_init, 'target_steps': 40000}
            q_init, ngd_s, ngd_h = data_points
            gd_curve = run_model(gd_param)['q'][:201]
            ngd_curve = wm({'s': ngd_s, 'h': ngd_h, 'target_steps': 40000, 'q0': q_init})['q'][:201]
    
            color1 = cmap1(norm(q_init))
            # color2 = cmap2(norm(q_init))
            plt.plot(np.arange(0, len(gd_curve)), gd_curve, marker = 'o', linestyle = '-', markersize=3, color = color1, label = f"GD q_init={q_init}, s = {gd_s}, c = {gd_c}, h = {gd_h}")
            plt.plot(np.arange(0, len(ngd_curve)), ngd_curve, linestyle = '--', markersize=3, color = color1, label = f"NGD q_init={q_init}, s = {ngd_s}, h = {ngd_h}")
    elif version == 2:
        grid_res = load_pickle("unstable_mapping_result", "h0.8_grid_unstable001_G.pickle")
        mapped_config = grid_res[current_gd]
        # mapped_config = load_mapping_unstable_2(filepath, current_gd)
        q_values = np.arange(0.01, 1.0, 0.2)
        norm = Normalize(vmin=min(q_values), vmax=max(q_values))
        ngd_s, ngd_h = mapped_config[0], mapped_config[1]
        solved_s, solved_h = solve_sngd_unstable(current_gd[0], current_gd[1], current_gd[2])

        ### Plotting mapped curves
        steps = 101
        totalmapdiff = 0
        totalsolvediff = 0
        for q_init in q_values:
            gd_param['q0'] = q_init
            gd_curve = run_model(gd_param)['q'][:steps]
            ngd_curve = wm({'s': ngd_s, 'h': ngd_h, 'target_steps': 40000, 'q0': q_init})['q'][:steps]
            solved_curve = wm({'s': solved_s, 'h': solved_h, 'target_steps': 40000, 'q0': q_init})['q'][:steps]
            color1 = cmap1(norm(q_init))
            q_init = round(q_init, 4)
            totalmapdiff += euclidean(ngd_curve, gd_curve)
            totalsolvediff += euclidean(solved_curve, gd_curve)
            plt.plot(np.arange(0, len(gd_curve)), gd_curve, marker = 'o', linestyle = '-', markersize=3, color = color1, label = f"GD q_init={q_init}")
            plt.plot(np.arange(0, len(ngd_curve)), ngd_curve, linestyle = '--', markersize=3, color = color1, label = f"NGD grid q_init={q_init}, s = {ngd_s:.3f}, h = {ngd_h:.3f}")
            plt.plot(np.arange(0, len(solved_curve)), solved_curve, linestyle = '-.', markersize=3, color = color1, label = f"NGD analytical q_init={q_init}, s = {solved_s:.3f}, h = {solved_h:.3f}")
    # else:
    #     plot_sngd_q(mapped, currH, current_gd)
    print("totalmse", totalmapdiff)
    print("totalsolvedmse", totalsolvediff)
    ##### plot q_init = eq curve #######################################
    eq_q = get_eq(gd_param)['q3']
    gd_param['q0'] = eq_q
    eq_gdcurve = run_model(gd_param)['q']
    print("eq_gdcurve", eq_gdcurve)
    eq_gdcurve = np.concatenate([eq_gdcurve, np.full(steps - len(eq_gdcurve), eq_gdcurve[-1])])
    eq_ngdcurve = wm({'s': ngd_s, 'h': ngd_h, 'target_steps': 40000, 'q0': eq_q})['q']
    totaldiff = totalsolvediff if plot_solve else totalmapdiff
    plt.plot(np.arange(0, len(eq_gdcurve)), eq_gdcurve, marker = 'o', linestyle = '-', markersize=3, color = 'red', label = f"GD q_eq={eq_q:.4f}, s = {current_gd[0]}, c = {current_gd[1]}, h = {current_gd[2]}")
    # plt.plot(np.arange(0, len(eq_ngdcurve)), eq_ngdcurve, linestyle = '--', markersize=3, color = 'red', label = f"NGD q_init={eq_q}, s = {ngd_s}, h = {ngd_h}")

    plt.xlabel('Time Steps')
    plt.ylabel('Mapped GD and NGD curves over different q_init')
    plt.title(f'GD vs NGD with different q_init for Unstable Regime (s={current_gd[0]}, c={current_gd[1]}, h={current_gd[2]})')
    # plt.legend(fontsize=8)
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"unstable_plots/h{currH}_s{ts}_c{tc}_unstable_{solved}_q02.pdf", format = 'pdf', dpi=600)
    plt.show()

'''
Plot mapping results from fixation mapping
currH: the fixed value of one parameter (could be c or h)
Fix h, s, and c on x-axis, plot s_ngd vs c and h_ngd vs c
PARAM_V: fixed for each curve, but varied for different curves
Fixed_vars: PARAM_V + another fixed parameter
x_param: the varied parameter on x-axis
NGD_p: y-axis label
'''
def plot_fixation_res(currH):
    hv = currH
    state = "fixation"
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text remains text
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.weight'] = 'book'
    gdFile = "001"
    slope_flag = False
    se_flag = True
    regression_flag = False
    diploid = False
    hap = "diploid" if diploid else "haploid"
    # {s, c, h -> 0, 1, 2}
    fixed_vars = {1, 2}
    PARAM_V = 1
    FV = list(fixed_vars-{PARAM_V})[0]
    PARAM0 = sorted(fixed_vars)[0]
    print(f"fixed vars: {fixed_vars}, varied param: {PARAM_V}")
    NGD_p = r"$S_{NGD}$" # y-axis
    allvars = {0, 1, 2}
    x_param = list(allvars - fixed_vars)[0]
    print("x_param:", x_param)
    plt.figure(figsize=(8, 7))
    color_cycle = plt.cm.Set2.colors

    # for different C_GD
    for fv_val in [hv]:
        print("curr fixed variable", fv_val)
        var_vals = []
        coeffs = [] # all (slope, intercept) for different PARAM_V
        records = [] # all (c, s, s_ngd) for different PARAM_V
        se_records = [] # all (c, s, se) for different PARAM_V
        for idx, var in enumerate(np.arange(0.2, 1.1, 0.2)):
            # if fixed C_gd, then H_gd is the variable
            if FV == 1: currH = round(float(var), 3)
            print("+++++current H", currH)
        
            # get grid mapp results 
            gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH, gdFile))
            # gradResult = load_pickle(f"h{currH}_hap_gradient_G_fix.pickle") # all gradient results
            if diploid: 
                if state == "unstable":
                    sMap_grid_diploid = load_pickle("unstable_mapping_results", f"h{currH}_grid_unstable001_G.pickle")
                else:
                    sMap_grid_diploid = load_pickle("fixation_mapping_result", f"h{currH}_grid_fix001_G.pickle") # all diploid grid results
            else:
                gridResult = load_pickle("fixation_mapping_result_haploid", f"h{currH}_hap_grid001_G_fix.pickle")
                sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']

            grid_res = sMap_grid_diploid if diploid else sMap_grid
            gd_configs, gd_res = gd_results[0], gd_results[1]

            # Store c, s_ngd, h_ngd values
            x_vals = []
            s_ngd_vals = []
            se_vals = []
            he_vals = []

            h_ngd_vals = []
            stability = get_stability_table(currH)

            # Convert all keys in the stability dictionary to upper case
            # if math.isclose(currH,0.6): 
            #     # print(stability)
            #     stability = {k.lower(): v for k, v in stability.items()}
            #     save_pickle(f"h{currH}_gametic_stability_res.pickle", stability)
            #     return

            for i, config in enumerate(gd_configs):
                s_gd, c_gd, h_gd = config
                pl = [s_gd, c_gd, h_gd]


                if s_gd*100%2==0 and math.isclose(pl[FV], float(fv_val)) and math.isclose(pl[PARAM_V], var): # check if h and c are correct
                    # print("matching h and c")
                    params = {'s': s_gd, 'c': c_gd, 'h': h_gd, 'q0':0.001,'target_steps': 40000, 'conversion':"gametic"}
                    eq = 1.0 if state == "fixation" else get_eq(params)['q3']
                    if (tuple(pl), eq) in stability[state]:
                        print(s_gd, c_gd, h_gd)
                        print("find fixation with required parameters")
                        if (s_gd, c_gd, h_gd) in grid_res:
                            if math.isclose(s_gd, 0.75) and math.isclose(c_gd, 0.6) and math.isclose(h_gd, 0.5):
                                continue
                            # print("also in smap_grid result")
                            x_vals.append(pl[x_param]) # store s (x_axis)

                            # mapped selection and dominance values from grid
                            mapped_params = grid_res[(s_gd, c_gd, h_gd)] 
                            # print(mapped_params)
                            if diploid: 
                                s_ngd, h_ngd = float(mapped_params[0]), float(mapped_params[1])
                                s_ngd_vals.append(s_ngd)
                                h_ngd_vals.append(h_ngd)
                                se = solve_sngd(s_gd, c_gd, h_gd)[0]
                                he = solve_sngd(s_gd, c_gd, h_gd)[1]
                                se_vals.append(se)
                                he_vals.append(he)
                                records.append((c_gd, s_gd, s_ngd))
                            else:
                                s_ngd = mapped_params
                                s_ngd_vals.append(s_ngd)
                                se = s_gd * h_gd - c_gd + c_gd * s_gd * h_gd
                                se_vals.append(se)
                                records.append((c_gd, s_gd, s_ngd))
                                se_records.append((c_gd, s_gd, se))

            # Sort by c for cleaner plotting
            sorted_indices = np.argsort(x_vals)
            x_vals = np.array(x_vals)[sorted_indices]
            s_ngd_vals = np.array(s_ngd_vals)[sorted_indices]
            se_vals = np.array(se_vals)[sorted_indices] if se_flag else None
            if diploid: 
                h_ngd_vals = np.array(h_ngd_vals)[sorted_indices]
            # 2)  load records into a NumPy array and sort by c_GD once
            # arr = np.asarray(records)                  
            # c_vals  = np.unique(arr[:,0])

            # Plot
            param_strings = [r"S_{GD}", r"C_{GD}", r"H_{GD}"]
            color = color_cycle[idx % len(color_cycle)]

            # Fit a linear regression line: y = mx + b
            if 'S' in NGD_p:
                plt.plot(x_vals, s_ngd_vals, color=color, alpha=0.8, marker='o', label=f"${param_strings[PARAM_V]}$ = {var:.1f} ({hap} grid)", ls='--')
                plt.plot(x_vals, se_vals, color=color, alpha=0.8, marker='o', label=f"${param_strings[PARAM_V]}$ = {var:.1f} ({hap} analytical)", ls='-')
                # if len(x_vals) >= 2:  # only fit if enough points
                #     slope, intercept = np.polyfit(x_vals, s_ngd_vals, 1)
                #     coeffs.append([slope, intercept])
                #     var_vals.append(var)
                #     x_fit = np.linspace(min(x_vals), max(x_vals), 100)
                #     y_fit = slope * x_fit + intercept
                #     plt.plot(x_fit, y_fit, linestyle='--', color=color, label=f"${param_strings[PARAM_V]}$ = {var:.3f}, slope = {slope:.2f}")
                    # slope2, intercept2 = np.polyfit(x_vals, se_vals, 1)
                    # x_fit = np.linspace(min(x_vals), max(x_vals), 100)
                    # y_fit2 = slope2 * x_fit + intercept2
                    # plt.plot(x_fit, y_fit2, linestyle='-', color=color, label=f"${param_strings[PARAM_V]}$ = {var:.3f}, slope = {slope:.2f}")
            else: 
                y_vals = h_ngd_vals if 'H' in NGD_p else s_ngd_vals
                plt.plot(x_vals, y_vals, color=color, marker='o', label=f"${param_strings[PARAM_V]}$ = {var:.1f}",  ls='--')
                plt.plot(x_vals, he_vals, color=color, alpha=0.8, marker='o', label=f"${param_strings[PARAM_V]}$ = {var:.1f} (He)", ls='-')
        # plt.plot(x_vals, h_ngd_vals, label=r'$h_{NGD}$', marker='s')

        plt.xlabel(f'${param_strings[x_param]}$ in Gene Drive model', fontsize=14)
        plt.ylabel(f'Mapped {NGD_p}', fontsize=14)
        plt.title(f"Mapped {NGD_p} vs. ${param_strings[x_param]}$ for different ${param_strings[PARAM_V]}$ at ${param_strings[FV]}$ = {fv_val}", fontsize=16)
        if PARAM_V == 0 and NGD_p == r"$S_{NGD}$":
            loc = "lower left"
        elif PARAM_V == 1 and NGD_p == r"$S_{NGD}$":
            loc = "lower right"
        elif PARAM_V == 1 and NGD_p == r"$H_{NGD}$":
            loc = "upper right"
        else:
            loc = "lower right"
        plt.legend(loc=loc, ncols=2, fontsize=9)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"plots_fixation/{state}_{hap}_{param_strings[x_param]}_{NGD_p}_var{param_strings[PARAM_V]}{gdFile}_neww.pdf", format="pdf", bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()

        if slope_flag:
            plt.figure(figsize=(8, 7))
            ### Plotting slopes and intercepts
            slopes = [ce[0] for ce in coeffs]
            intercepts = [ce[1] for ce in coeffs]
            print("c_vals:", var_vals, "slopes", slopes, "intercepts", intercepts)
            plt.plot(var_vals, slopes, label=f"${param_strings[2]}$ = {currH:.2f}", marker = 'o')
            plt.plot(var_vals, intercepts, label=f"${param_strings[2]}$ = {currH:.2f}", marker = 'x')
            plt.xlabel(f'${param_strings[1]}$ in Gene Drive model', fontsize=14)
            plt.ylabel(f'Slopes of Mapped {NGD_p} over ${param_strings[0]}$', fontsize=14)
            plt.title(f"{NGD_p} Slopes vs. ${param_strings[1]}$ over differnt ${param_strings[2]}$", fontsize=16)
            plt.legend()
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(f"plots_fixation/slope_varH_xC_y{NGD_p}_h{currH}.jpg", dpi=600)
            plt.close()
        
        if regression_flag:
            # calculate coefficients a and b for a*s_gd + b
            df = pd.DataFrame(records, columns=['c','S_GD','S_NGD'])
            # 1) precompute polynomial columns
            df['c2'] = df['c']**2
            # df['c3'] = df['c']**3

            # 2) fit with S_GD interactions on each power of c
            formula = '''
            S_NGD ~ 
                S_GD 
            + c + c2
            + S_GD:c + S_GD:c2
            '''
            res = smf.ols(formula, data=df).fit()
            print(res.summary())
            print(res.params)
            save_pickle("regression", f"h{currH}_mapping_coeffs_sq.pickle", res.params)

def plot_h_ngd_vs_h(currH, s_gd=0.3, 
                   c_values=np.arange(0.3, 1.0, 0.2), 
                   h_values=np.arange(0.1, 1.1, 0.1),
                   mapping_dir="mapping_result",
                   outfile="plots_fixation/h_ngd_vs_h_s0.1.pdf"):
    """
    For each s in s_values, load the mapping_result/h{h}_grid_fix001_G.pickle
    files (one per h in h_values), extract H_NGD = mapped_params[1] for key
    (s, c_gd, h), and plot H_NGD vs h.
    """
    plt.figure(figsize=(8,6))
    fixed = 's'
    outfile=f"plots_fixation/h_ngd_vs_h_{fixed}{s_gd}.pdf"
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    state = 'fix' ### CHANGE THIS EVERYTIME SWITCHING BTW UNSTABLE AND FIXATION

    for idx, c_gd in enumerate(c_values):
        h_ngd_vals = []
        he_vals = []
        for h_gd in h_values:
            h_gd = round(h_gd, 3)
            p_subdir = mapping_dir
            p_filename = f"h{h_gd}_grid_{state}001_G.pickle"
            if not (Path(PICKLE_DIR) / p_subdir / p_filename).exists():
                print(f"no current hval: {h_gd}")
                return

            grid = load_pickle(p_subdir, p_filename)
            # grid['map'] maps (s_gd, c_gd, h_gd) → (s_ngd, h_ngd)
            # sMap = grid.get('map', {})
            sMap = grid.get('map', {}) if 'map' in grid else grid 
            # print(h_gd, sMap)

            key = (round(float(s_gd),3), round(float(c_gd),3), round(float(h_gd),3))
            if key in sMap:
                if type(sMap[key]) == dict:
                    _, h_ngd = sMap[key]['config']
                else:
                    _, h_ngd = sMap[key]
                # se = solve_sngd(s_gd, c_gd, h_gd)[0]
                he = solve_sngd(s_gd, c_gd, h_gd)[1]
                # se_vals.append(se)
                he_vals.append(he)
            else:
                h_ngd = np.nan
                he_vals.append(np.nan)
            h_ngd_vals.append(h_ngd)

        color = color_cycle[(idx+1) % len(color_cycle)]
        plt.plot(h_values, 
                 h_ngd_vals, 
                 marker='o',
                 color=color,
                 label=fr"$C_{{GD}}={c_gd:.1f}$ (Diploid Grid)", 
                 ls='--')
        plt.plot(h_values, he_vals, color=color, marker='x', label=fr"$C_{{GD}}={c_gd:.1f}$ (Haploid Analytical)", ls='-')

    plt.xlabel(r"$H_{GD}$", fontsize=14)
    plt.ylabel(r"$H_{NGD}$", fontsize=14)
    plt.title(rf"Mapped $H_{{NGD}}$ vs $H_{{GD}}$, $S_{{GD}}={s_gd}$", fontsize=16)
    plt.legend(title=r"$S_{GD}$", ncol=2, fontsize=8)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.show()

def test_mapping_trajectory(s_gd, c_gd, h_gd,
                             param_file,
                             file_type='json',
                             target_steps=40000,
                             q0=0.001,
                             show=True):
    """
    1) Load saved linear-regression coefficients from `param_file` (pickle).
    2) Compute s_ngd and h_ngd from the loaded params.
    3) Run both trajectories and plot them together.
    """

    # --- 1) Load the fitted model parameters ------------------------
    if file_type == 'pickle' and param_file:
        if "/" in param_file:
            _subdir, _fname = param_file.rsplit("/", 1)
        else:
            _subdir, _fname = "", param_file
        res = load_pickle(_subdir, _fname)
    else:
        res = None
    coeffs = res.to_dict()
    print('coeffs:', coeffs)

    # --- 2) Compute mapped parameters from the regression ------------
    # Basic terms:
    intercept = coeffs.get('Intercept', 0.0)
    a_s      = coeffs.get('S_GD',    0.0)   # slope coef on S_GD
    a_c      = coeffs.get('c',       0.0)   # slope coef on c
    a_sc     = coeffs.get('S_GD:c',  0.0)   # interaction term
    a_c2    = coeffs.get('c2',      0.0)   # slope coef on c^2
    a_sc2   = coeffs.get('S_GD:c2', 0.0)   # interaction term

    # compute S_NGD
    s_ngd = (
        intercept
      + a_s   * s_gd
      + a_c   * c_gd
      + a_sc  * s_gd * c_gd
      + a_c2  * c_gd**2
      + a_sc2 * s_gd * c_gd**2
    )
    print(s_ngd)
    # Compute H_NGD
    if s_ngd != 0:
        h_ngd = (1 - (1 - h_gd*s_gd)*(1 + c_gd)) / s_ngd
        if h_ngd < 0:
            h_ngd = 0
        elif h_ngd > 1:
            h_ngd = 1
    else:
        h_ngd = h_gd
    
    print(h_ngd)

    # --- 3) Generate trajectories -----------------------------------
    # GD trajectory
    gd_params = {
        's': s_gd,
        'c': c_gd,
        'h': h_gd,
        'target_steps': target_steps,
        'q0': q0
    }
    traj_gd = run_model(gd_params)['q']

    # NGD trajectory
    traj_ngd = wm({'s': s_ngd, 'h': h_ngd, 'target_steps': target_steps, 'q0': q0})['q']

    # --- 4) Plot comparison -----------------------------------------
    plt.figure(figsize=(8,5))
    gd_generations = list(range(len(traj_gd)))
    ngd_generations = list(range(len(traj_ngd)))
    plt.plot(gd_generations, traj_gd, label=f'GD: s={s_gd:.3f}, c={c_gd:.3f}, h={h_gd:.3f}')
    plt.plot(ngd_generations, traj_ngd, 
             label=f'NGD: s_ngd={s_ngd:.3f}, h_ngd={h_ngd:.3f}', 
             linestyle='--')

    plt.xlabel('Generation')
    plt.ylabel('Allele frequency  q(t)')
    plt.title('GD vs NGD Trajectories')
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()

    # also return data if you need to compute MSE or other metrics
    return {
        'traj_gd': traj_gd,
        'traj_ngd': traj_ngd,
        's_ngd': s_ngd,
        'h_ngd': h_ngd
    }


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_fixation_surface(currH):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH))
    gridResult_diploid = load_pickle("", f"h{currH}_grid_fix001_G_unstable.pickle")
    stability = get_stability_table(currH)

    gd_configs, _ = gd_results
    sMap_grid_diploid = gridResult_diploid['map']

    s_gd_vals = []
    c_gd_vals = []
    s_ngd_vals = []
    h_ngd_vals = []

    for config in gd_configs:
        s_gd, c_gd, h_gd = config

        if not math.isclose(h_gd, float(currH)):
            continue

        if ((s_gd, c_gd, h_gd), 1.0) not in stability["fixation"]:
            continue

        if (s_gd, c_gd, h_gd) not in sMap_grid_diploid:
            continue

        mapped_params = sMap_grid_diploid[(s_gd, c_gd, h_gd)]
        s_ngd = mapped_params[0]
        h_ngd = mapped_params[1]

        s_gd_vals.append(s_gd)
        c_gd_vals.append(c_gd)
        s_ngd_vals.append(s_ngd)
        h_ngd_vals.append(h_ngd)

    # Convert to arrays for plotting
    s_gd_vals = np.array(s_gd_vals)
    c_gd_vals = np.array(c_gd_vals)
    s_ngd_vals = np.array(s_ngd_vals)
    h_ngd_vals = np.array(h_ngd_vals)

    # Plot the points
    sc = ax.scatter(s_gd_vals, c_gd_vals, h_ngd_vals, c=h_ngd_vals, cmap='Spectral', marker='o')
    fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.5, label=r'$H_{NGD}$')

    ax.view_init(elev=5, azim=200)

    ax.set_xlabel(r'$S_{GD}$', fontsize=12)
    ax.set_ylabel(r'$C_{GD}$', fontsize=12)
    ax.set_zlabel(r'$H_{NGD}$', fontsize=12)
    ax.set_title(fr"Mapped $H_{{NGD}}$ surface for $H_{{GD}} = {currH}$", fontsize=14)

    plt.tight_layout()
    plt.savefig(f"plots_fixation/3d_surface_hngd_h{currH}.jpg", dpi=600)
    plt.close()

from .solve import solve_sngd, solve_sngd_unstable

def plot_sngd(currH):
    """
    Plot the mapped s_NGD as a colormap on the (h_NGD, q) plane
    for a fixed GD configuration (s_gd, c_gd, h_gd).
    """
    # 1) fixed gene-drive params
    gd_s, gd_c, gd_h = 0.5, 0.8, 0.8
    gridResult = load_pickle("", f"h{currH}_grid_fix001_G.pickle")

    grid_map, grid_mapcurves = gridResult['map'], gridResult['ngC']
    mapped_params = grid_map[(gd_s, gd_c, gd_h)]
    mapped_s, mapped_h = mapped_params[0], mapped_params[1]

    # 2) prepare grid in q and h_NGD
    q_vals = np.arange(0.001, 1.0, 0.001)
    h_vals = np.arange(-1.0, 1.0, 0.01)

    h_s_map = dict()

    # 3) collect mapping results
    H, Q, S = [], [], []
    for h_ngd in h_vals:
        curr_s = []
        for q in q_vals:
            try:
                s_ngd = solve_sngd(gd_s, gd_c, gd_h, h_ngd, q)
            except Exception:
                continue
            if s_ngd is None or np.isnan(s_ngd):
                continue
            H.append(h_ngd)
            Q.append(q)
            S.append(round(float(s_ngd), 5))
            curr_s.append(round(float(s_ngd), 3)) 
        h_s_map[round(h_ngd, 3)] = curr_s


    # convert to arrays
    H = np.array(H)
    Q = np.array(Q)
    S = np.array(S)
    print("S", S)

    # 4) write the s_ngd sequence for the mapped h_ngd to a file
    os.makedirs("sngd_heatmap", exist_ok=True)
    txt_path = f"sngd_heatmap/sngd_s{gd_s}_c{gd_c}_h{gd_h}.txt"
    with open(txt_path, 'w') as fout:
        fout.write(
            f"gd_configuration=({gd_s}, {gd_c}, {gd_h}), "
            f"mapped h_ngd={h_ngd:.6f}, "
            f"sngd list={h_s_map[mapped_h]}\n"
        )
    print(f"Line data written to {txt_path}")

    # 5) scatter plot with colormap
    plt.figure(figsize=(8,6))
    norm = mcolors.Normalize(vmin=-5, vmax=5)
    print(norm.vmin, norm.vmax)
    sc = plt.scatter(
        Q, H,
        c=S,
        cmap='magma',
        norm=norm,
        s=10,
        marker='o',
        edgecolors='none',
        alpha=0.8
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$s_{NGD}$', fontsize=12)

    plt.axhline(y=mapped_h, color="black", linestyle="--", linewidth=2,
                label=fr"mapped $h_{{NGD}}={mapped_h:.3f}$")
    plt.legend(loc="upper right")

    # 5) labels and title
    plt.xlabel(r'$q_{init}$', fontsize=12)
    plt.ylabel(r'$h_{NGD}$', fontsize=12)
    plt.title(
        f"Mapped $s_{{NGD}}$ on ($h_{{NGD}}$, $q$)-plane for GD (s={gd_s}, c={gd_c}, h={gd_h})",
        fontsize=14
    )
    plt.grid(True, linestyle='--', alpha=0.5)

    # 6) save and show
    plt.tight_layout()
    outname = f"sngd_heatmap/sngd_vs_hq_s{gd_s}_c{gd_c}_h{gd_h}.png"
    plt.savefig(outname, dpi=600)
    plt.show()

    # plot_msedl((0.7, 0.4, 0.3))
    # plot_eq_lambda((0.2, 0.1, 0.3))

def build_df(configs, grid_map, q_vals):
    """
    configs : list of (s_GD, c_GD, h_GD) tuples
    grid_map : dict mapping (s_GD, c_GD, h_GD) to (s_NGD, h_NGD)
    q_vals  : 1-D numpy array of q grid
    returns : long-form DataFrame
              columns = ['s_gd','c_gd','h_gd','q','h_ngd','s_ngd']
    """
    rows = []
    for s_gd, c_gd, h_gd in configs:
        # mapped h_NGD for *this* config (from your pickle)
        h_ngd = grid_map[(s_gd, c_gd, h_gd)][1]
        for q in q_vals:
            s_ngd = solve_sngd(s_gd, c_gd, h_gd, h_ngd, q)
            rows.append([s_gd, c_gd, h_gd, q, h_ngd, s_ngd])
    return pd.DataFrame(rows,
                        columns=['s_gd','c_gd','h_gd','q','h_ngd','s_ngd'])


def plot_sngd_all(currH):
    """
    plot the change in sngd with q_init for different configurations
    """
    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH))
    gridResult = load_pickle("", f"h{currH}_grid_fix001_G.pickle")

    gd_configs, _ = gd_results[0], gd_results[1]
    currH = float(currH)
    configs_to_plot = [(0.1, 0.1, currH), (0.1, 0.9, currH), (0.2, 0.2, currH), (0.2, 0.6, currH), (0.4, 0.6, currH), (0.5, 0.9, currH)]

    grid_map, _ = gridResult['map'], gridResult['ngC']

    # Build the dataframe of s_ngd vs q_init for each config
    q_vals = np.arange(0.001, 1.0, 0.001)
    df = build_df(configs_to_plot, grid_map, q_vals)

    plt.figure(figsize=(8,6))
    palette = sns.color_palette("tab10", n_colors=len(configs_to_plot))

    for (cfg, colour) in zip(configs_to_plot, palette):
        s_gd, c_gd, h_gd = cfg
        subset = df[(df['s_gd']==s_gd) & (df['c_gd']==c_gd) & (df['h_gd']==h_gd)]
        plt.plot(subset['q'], subset['s_ngd'],
                label=f"s={s_gd}, c={c_gd}, h={h_gd}",
                color=colour, lw=1.6)

    plt.xlabel(r"$q_{init}$", fontsize=12)
    plt.ylabel(r"$s_{NGD}$ (mapped)", fontsize=12)
    plt.title("Mapped $s_{NGD}$ vs q curves for each drive configuration at the mapped $h_{NGD}$",
            fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, title="GD config", ncol=2)
    plt.tight_layout()
    plt.savefig(f"h{currH}_sngd_curves_all_configs.png", dpi=600)
    plt.show()

### 10.22 Latest version of plotting unstable/stable mapping error figures #################################
def find_state_analytic(model, sval, hval, cval=0.0):
    """Delegate to regime.get_regime_and_eq; returns regime string (fixation, loss, stable, unstable)."""
    from .regime import get_regime_and_eq
    model_key = "GD" if model.lower() == "gd" else "NGD"
    c = cval if model_key == "GD" else None
    r, _ = get_regime_and_eq(model_key, sval, hval, c=c)
    return r if r else ""

### NEWEST PLOTTING FUNCTION FOR HEATMAP OF UNSTABLE MAPPING ERROR
### Work for unstable, stable or loss, analytic or grid search result
def get_unstable_diff(currH, state="unstable", ana="grid"):
    '''
    Compute the difference between mapped/solved s_ngd and GD curves for unstable/stable regime
    Take the current h value, calculate total mse error over a range of q_init
    Result stored in the mapdiff pickle file and the qerror txt file. 
    '''
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH))
    # gradResult = load_pickle(f"h{currH}_hap_gradient_G_fix.pickle") # all gradient results
    # if ana == "grid":
    if state == 'loss':
        mapres_fn = "all_loss_map_results_diploid_qlarge"
    else:
        mapres_fn = f"h{currH}_grid_{state.lower()}001_G"
    gridResult = load_pickle(f"{state.lower()}_mapping_result", f"{mapres_fn}.pickle")
    sMap_grid = gridResult
        # print("check grid values", gridResult[(0.53, 0.25, 0.8)])

    gd_configs, _ = gd_results[0], gd_results[1]

    # sMap_grid, wms_grid = gridResult['map'], gridResult['ngC']

    stabilityRes = get_stability_table(currH)
    # Print the contents of stabilityRes
    print("Contents of stabilityRes:", stabilityRes.keys())

    # Write the contents to a readable txt file
    # txt_path = f"h{currH}_gametic_stability_res_readable.txt"
    # with open(txt_path, "w") as fout:
    #     for key, value in stabilityRes.items():
    #         fout.write(f"{key}:\n")
    #         fout.write(f"{value}\n")
    # print(f"stabilityRes written to {txt_path}")
    # collect data
    all_s, all_c, all_diff = [], [], []

    solved_res = dict()
    all_q_error = dict()
    diff_map = dict()

    state_dict = dict()
    gd_configs = stabilityRes[state.lower()]

    for (config, eq) in gd_configs:
        s_gd, c_gd, h_gd = config
        if not math.isclose(h_gd, float(currH)):
            continue
        
        # state = find_state_analytic('gd', s_gd, h_gd, c_gd)
        # state_dict[state] = (s_gd, c_gd, h_gd)
        # continue

        # if find_state_analytic('gd', s_gd, h_gd, c_gd) != state.lower(): 
        #     continue
        # eq = max(0, min(1.0, eq))
        
        # if find_state_analytic(s_gd, c_gd, h_gd) != "fixation":
        #     continue
        # # print(f"================eq:{eq}")
        # if ((s_gd, c_gd, h_gd), eq) not in stabilityRes[state]:
        #     print(f"{s_gd}, {c_gd}, {h_gd}")
        #     continue
        # must exist in grid mapping
        if ana == "grid" and (round(s_gd, 3), round(c_gd, 3), round(h_gd, 3)) not in sMap_grid:
            print(f"@ ERROR: Config  {(s_gd, c_gd, h_gd)} not in sMap_grid")
            continue
        
        # if not (math.isclose(s_gd, 0.6) and math.isclose(c_gd, 0.91)):
        #     continue

        total_map_diff = 0
        print(config)
        # total_ana_diff = 0
        # (a) mapped NGD from simulation (unstable)
        if ana == 'grid': 
            if type(sMap_grid[(s_gd, c_gd, h_gd)]) != tuple:
                continue
            mapped_s, mapped_h = sMap_grid[(s_gd, c_gd, h_gd)][0], sMap_grid[(s_gd, c_gd, h_gd)][1] 
            mapped_s, mapped_h = round(float(mapped_s), 4), round(float(mapped_h), 4)
    
        # print(mapped_s, mapped_h)
        # (b) solved NGD from analytic formula
        # UPDATE 10.22: add in stable analytics
        else: 
            if state == 'stable' or state == 'unstable':
                mapped_s, mapped_h = solve_sngd_unstable(s_gd, c_gd, h_gd)
            else: 
                mapped_s, mapped_h = solve_sngd(s_gd, c_gd, h_gd)
            mapped_config = (mapped_s, mapped_h)
            solved_res[(s_gd, c_gd, h_gd)] = mapped_config


        q_error = dict()
        if state == 'unstable':
            q_range = np.arange(0.01, 1.0, 0.1) 
        elif state == 'fixation':
            q_range = [0.01]
        else:
            q_range = [0.8]

        # trajectory difference
        for q_init in q_range:
            gd_params = {'s': s_gd, 'c': c_gd, 'h': h_gd, 'q0':q_init,'target_steps': 40000}
            gd_curve=run_model(gd_params)['q']
            
            mapped_curve = wm({'s': mapped_s, 'h': mapped_h, 'target_steps': 40000, 'q0': q_init})['q']

            map_diff = euclidean(mapped_curve, gd_curve)
            # ana_diff = euclidean(analytical_curve, gd_curve)
            total_map_diff+=map_diff
            # total_ana_diff += ana_diff
            # print("========Single Mapping diff", map_diff)
            # print("--------Single Analytic diff", ana_diff)
            # sq_diff = ana_diff if ana=='analytical' else map_diff
            q_error[round(float(q_init), 3)] = round(float(map_diff), 6)
        
        all_q_error[(s_gd, c_gd, h_gd)] = q_error
        debug = False

        ### DEBUGGING SPECIFIC CONFIGURATION ##################
        if debug and math.isclose(s_gd, 0.6) and math.isclose(c_gd, 0.91):
            print("solved params", mapped_s, mapped_h)
            plt.figure(figsize=(8,6))
            plt.plot(mapped_curve, label='Mapped NGD', color='blue')
            # plt.plot(analytical_curve, label='Solved NGD', color='orange')
            plt.plot(gd_curve, label='GD', color='red')
            plt.xlabel('Generation')
            plt.ylabel('Allele frequency q(t)')
            plt.title(f'Comparison of Mapped vs Solved NGD Trajectories\n'
                      f'for GD (s={s_gd}, c={c_gd}, h={h_gd})')
            plt.legend()
            plt.grid(False, linestyle='--', alpha=0.5)
            plt.tight_layout()
            # plt.savefig(f"plots_diff/mapped_vs_solved_h{currH}_s{s_gd}_c{c_gd}_q02.jpg", dpi=600)
            plt.show()
        ### DEBUGGING SPECIFIC CONFIGURATION ##################

        total_diff = total_map_diff / len(q_range)
        # print("Mapping diff", total_map_diff)
        # print("Analytic diff", total_ana_diff)
        all_s.append(s_gd)
        all_c.append(c_gd)
        all_diff.append(total_diff)
        diff_map[(s_gd, c_gd, h_gd)] = total_diff
    
    # save_pickle(f"analytical_partition_gd.pickle", state_dict)
    # with open("analytical_partition_gd.txt", "w") as f:
    #     for state, configs in state_dict.items():
    #         f.write(f"=== STATE: {state} ===\n")
    #         if isinstance(configs, (list, tuple)):
    #             for cfg in configs:
    #                 f.write(f"{cfg}\n")
    #         else:
    #             f.write(f"{configs}\n")
    #         f.write("\n")  # blank line between states

    # print(f"✅ Wrote formatted text to {txt_path}")
    # return 
    ### Saving analytic mapdiff to pickle
    save_pickle("anasim_mapdiff", f"h{currH}_{state}_{ana}_mapdiff.pickle", diff_map)
    ### Write all_q_error dict to a txt file (key: value per line)
    txt_path = f"mapping_error_txt/h{currH}_{state.lower()}_{ana}_qerrors.txt"
    with open(txt_path, "w") as fout:
        for key, value in all_q_error.items():
            fout.write(f"{key}: {value}\n")
    print(f"{ana} q_errors written to {txt_path}")

    ### Write solved_result to txt file 
    if ana == 'analytical': 
        txt_path = f"mapping_error_txt/h{currH}_{state.lower()}_analytical_solved_res.txt"
        with open(txt_path, "w") as fout:
            for key, value in all_q_error.items():
                fout.write(f"{key}: {value}\n")
        print(f"{ana} solved results written to {txt_path}")

### PLOT the mapping diff heatmap on (s, c) axis based on get_unstable_diff
def plot_diff(currH):
    '''
    Plot the unstable error heatmap based on get_unstable_diff
    '''
    ana = "analytical" # or "analytical"
    state = "unstable" # unstable
    # get_unstable_diff(currH, state, ana) # comment off if only plotting
    # return 
    all_s = []
    all_c = []
    all_diff = []
    mapdiff = load_pickle("anasim_mapdiff", f"h{currH}_{state}_{ana}_mapdiff.pickle")
    for config, diff in mapdiff.items():
        # if math.isclose(config[0], 1.0): continue
        if state == "unstable":
            q_range = np.arange(0.01, 1.0, 0.1)
            diff *= len(q_range)
        all_s.append(config[0])
        all_c.append(config[1])
        all_diff.append(diff)
    all_s = np.array(all_s)
    all_c = np.array(all_c)
    all_diff = np.array(all_diff)

    # make plot directory
    os.makedirs('plots_diff', exist_ok=True)

    # scatter heatmap
    plt.figure(figsize=(8,6))
    # print(all_diff)
    minval, maxval = np.min(all_diff), np.max(all_diff)
    # maxval = 0.01
    # maxval for h=0.6 stable = 0.001
    # maxval from h0.8 plots = 0.23998756559392723
    maxval = 0.24
    print(np.min(all_diff), np.max(all_diff))

    # Color of the analytical graph should match with grid search 
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=minval, vmax=maxval)

    def truncate_colormap(cmap, minval=0.5, maxval=1.0, n=256):
        new_colors = cmap(np.linspace(minval, maxval, n))
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            f'truncated({cmap.name},{minval:.2f},{maxval:.2f})', new_colors)
        return new_cmap
    
    if state == 'fixation':
        base_cmap = plt.get_cmap('Reds')
        cmap = truncate_colormap(base_cmap, 0.1, 0.7)

        unique_s = np.unique(all_s)
        unique_c = np.unique(all_c)
        diff_grid = np.full((len(unique_s), len(unique_c)), np.nan)
        print(max(unique_s), max(unique_c))

        # Map (s, c) -> grid indices
        s_to_idx = {val: i for i, val in enumerate(unique_s)}
        c_to_idx = {val: j for j, val in enumerate(unique_c)}

        for s_val, c_val, d_val in zip(all_s, all_c, all_diff):
            i = s_to_idx[s_val]
            j = c_to_idx[c_val]
            diff_grid[i, j] = d_val

        pt = plt.imshow(
            diff_grid,
            origin='lower',
            extent=[min(unique_c), max(unique_c), min(unique_s), max(unique_s)],
            aspect='auto',
            cmap=cmap,
            norm=norm
        )
    else:
        cmap = "Blues"
        pt = plt.scatter(all_c, all_s, c=all_diff, cmap=cmap, norm=norm, s=50, linewidths=0.3, alpha=0.8)

    cbar = plt.colorbar(pt)
    cbar.set_label('Euclidean diff of trajectories', fontsize=12)

    plt.xlabel('c_GD', fontsize=12)
    plt.ylabel('s_GD', fontsize=12)
    plt.title(f'Trajectory error: GD vs {ana} NGD (h_GD={currH})', fontsize=14)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.grid(False)
    plt.tight_layout()

    outpath = f'plots_diff/{state}_{ana}_diff_heatmap_h{currH}.pdf'
    plt.savefig(outpath, format="pdf", dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {outpath}") 
    
         