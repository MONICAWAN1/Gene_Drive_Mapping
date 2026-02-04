import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import sys, os
import pandas as pd

import sympy as sp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import wm

def compute_lambda_gd():
# Define symbols
    s, c, h, q = sp.symbols('s c h q')

    # Define numerator and denominator
    sc = (1 - h*s) * c
    sn = 0.5 * (1 - h * s) * (1 - c)
    numerator = q**2 * (1 - s) + 2 * q * (1 - q) * c * (sc+sn)
    wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (sc+2*sn) + (1 - q)**2

    # Define q(t+1)
    q_next = numerator / wbar

    # Compute the derivative ∂q(t+1)/∂q(t)
    dq_next_dq = sp.simplify(sp.diff(q_next, q))

    dq_dq_func = sp.lambdify((s, c, h, q), dq_next_dq, "numpy")

    # Print the simplified result
    sp.pprint(dq_next_dq)
    return dq_dq_func

def get_gd_stability(s, c, h, q, dfunc):
    try:
        slope = dfunc(s, c, h, q)
    except ZeroDivisionError:
        slope = 'NA'
        print(f"Division by zero: config = {(s, h, q)}")
    return slope

def get_ngd_stability_old(s, h, q):
    num = 2*q**2*(1-q)*(1-s)*(1-h*s) - q**2*(1-2*q)*(1-s)*(1-h*s) + 2*q*(1-q)**2*(2-s-h*s) + (1-2*q)*(1-q)**2*(1-h*s)

    denom = (q**2*(1-s) + 2*(1-q)*q*(1-h*s) + (1-q)**2)**2

    if denom != 0:
        slope = num/denom
    else:
        slope = 'NA'
        print(f"no slope result, config = {(s, h, q)}")
    
    return slope

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_pickle,save_pickle

def derivative(params):
    s, c, h = params['config']
    q = params['currq']
    if params['conversion'] == 'zygotic':
        sc = c*(1-s)
    else:
        sc = c*(1-h*s)
    sn = 0.5*(1-c)*(1-h*s)

    num = 2*q**2*(1-q)*(1-s)*(2*sn+1) - 2*q**2*(1-2*q)*(1-s)*sn + 2*q*(q-1)**2*(1-s+2*sn+2*sc) + 2*(1-2*q)*(1-q)**2*(sn+sc)

    denom = (q**2*(1-s) + 2*(1-q)*q*(2*sn+sc) + (1-q)**2)**2

    if denom != 0:
        slope = num/denom
    else:
        slope = 'NA'
        print(f"no slope result, config = {params['config']}")
    
    return slope

def get_eq(params):
    s, c, h = params['config']
    q = params['q0']
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
return allres: dict{"state": [(config1, eq), ...]}
'''
def runall(params):
    allres = {'Stable': [], 'Unstable': [], 'dq=1': [], 'Fixation': [], 'Loss':[]}
    if params['conversion'] == 'zygotic':
        g_res = load_pickle("gd_simulation_results", f"h{params['config'][2]}_allgdres001.pickle")
    else:
        g_res = load_pickle("gd_simulation_results", f"h{params['config'][2]}_allgdres001G.pickle")
    configs, res = g_res[0], g_res[1]
    print('allconfigs', configs[:2])
    h = params['config'][2]

    for s in np.arange(0.01, 1.01, 0.01):
        for c in np.arange(0.01, 1.01, 0.01):
    
            params['config'] = (round(float(s), 3),round(float(c), 3),round(float(h), 3))
            config = params['config']

            if config not in configs:
                continue

            alleq=get_eq(params) ### get all eq points for the current config
            eq1, eq2, eq3 = alleq['q1'], alleq['q2'], alleq['q3']

            # compute stability for eq3

            if eq3 != 'NA' and 0 < eq3 and eq3 < 1 and not math.isclose(eq3, 0) and not math.isclose(eq3, 1):
                params['currq']=eq3
                eq_stability = derivative(params)
                # print(f"config = {params['config']}, slope={eq_stability}")

                # if slope < 1: stable
                if 0 < eq3 < 1 and eq_stability != 'NA':
                    if abs(eq_stability) < 1:
                        allres['Stable'].append((params['config'], eq3))
                    elif abs(eq_stability) > 1:
                        allres['Unstable'].append((params['config'], eq3))
                    elif math.isclose(eq_stability, 1):
                        allres['dq=1'].append((params['config'], eq3))
                # find fixation and loss
            elif eq3 != 'NA' and (eq3 >= 1 or eq3 <= 0 or math.isclose(eq3, 0) or math.isclose(eq3, 1)):
                if math.isclose(eq3, 1) or res[params['config']]['state'] == 'fix':
                    allres['Fixation'].append((params['config'], 1.0))
                elif res[params['config']]['state'] == 'loss':
                    allres['Loss'].append((params['config'], 0.0))
            # if math.isclose(s, 0.08) and math.isclose(c, 0.08):
            #     print('something eq3', eq3, math.isclose(eq3, 1))
            #     print(((s, c, h), eq3) in allres['Fixation'])
            elif eq3 == 'NA':
                if res[params['config']]['state'] == 'loss':
                    allres['Loss'].append((params['config'], 0.0))
                elif res[params['config']]['state'] == 'fix':
                    allres['Fixation'].append((params['config'], 1.0))
    # -----------------------------------------------------------------
    # write the dictionary to a text file
    # -----------------------------------------------------------------
    os.makedirs("phase_partition", exist_ok=True)
    h_val = h
    fname = f"phase_partition/partition_h{h_val}.txt"
    with open(fname, "w") as fout:
        for regime in ['Stable', 'Unstable', 'dq=1', 'Fixation', 'Loss']:
            fout.write(f"## {regime}\n")
            for (cfg, eq) in allres[regime]:
                s_val, c_val, h_val = cfg
                fout.write(f"s={s_val:.3f}, c={c_val:.3f}, h={h_val:.3f}, eq={eq:.4f}\n")
            fout.write("\n")

    print(f"Partition written to {fname}")
        
    return allres

def plot_all_by_regime(allres, params):
    '''
    plot the partition of gene drive configurations based on final result 
    by colors

    '''
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    # pick a qualitative colormap and get one color per state
    cmap      = plt.cm.get_cmap('Set3')
    states    = list(allres.keys())
    n_states  = len(states)
    color_map = { state: cmap(i % cmap.N) for i, state in enumerate(states) }

    for state, configs in allres.items():
        # unpack s and c
        s_values = [c[0][0] for c in configs]
        c_values = [c[0][1] for c in configs]

        if not configs:
            continue

        ax.scatter(
            c_values,
            s_values,
            color=color_map[state],
            label=state,
            s=50,
            alpha=0.8,
        )

    # tidy up
    ax.set_xlabel('Conversion Factor (c)', fontsize=12)
    ax.set_ylabel('Selection Coefficient (s)', fontsize=12)
    ax.set_title(
        f"Partition of Gene Drive Configurations Based on Stability (H={params['config'][2]})",
        fontsize=14
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True)

    # draw legend
    ax.legend(title='Regime', fontsize=10, title_fontsize=11, loc='upper right')

    # save & show
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text remains text
    plt.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    plt.savefig(f"plot_partition/gd_h{params['config'][2]}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
            
def plot_all(allres, params):
    fig, ax = plt.subplots()
    for state in allres:

        unstable_configurations = allres[state]
        s_values = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
        c_values = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
        eq_values = [max(0, min(1, config[1])) for config in unstable_configurations]
        # print(state, eq_values)

        
        if eq_values != []:
            # Plotting
            norm = mcolors.Normalize(0.0, 1.0)
            # if state == 'Stable' or state == 'Unstable':
            cmap = plt.cm.get_cmap('viridis')
            # plt.figure(figsize=(8, 6))
            # plt.scatter(c_values, s_values, alpha=0.7, edgecolors='k', label=f"Unstable Configurations" )
            ax.scatter(c_values, s_values, c=eq_values, cmap=cmap, s=50, norm=norm)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Gene Drive Allele Frequency at Eq')

    # Adding labels, title, and legend
    ax.set_xlabel('Conversion Factor (c)', fontsize=12)
    ax.set_ylabel('Selection Coefficient (s)', fontsize=12)
    # plt.title(f'Scatter Plot of {state} Configurations', fontsize=14)
    ax.set_title(f"Partition of Gene Drive Configurations Based on Stability (H={params['config'][2]})", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    # plt.legend(fontsize=10)

    # Show plot
    plt.grid(True)
    plt.savefig(f"plot_partition/gd_h{params['config'][2]}.jpg", dpi=600)
    plt.show()


def plot_regions(allres, state, conversion):
    unstable_configurations = allres[state]
    s_values = [config[0][0] for config in unstable_configurations]  # First element in the tuple (s)
    c_values = [config[0][1] for config in unstable_configurations]  # Second element in the tuple (c)
    print("unstable", unstable_configurations)
    h_val = unstable_configurations[0][0][2]
    eq_values = [max(0, min(1, config[1])) for config in unstable_configurations]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mcolors.Normalize(min(eq_values), max(eq_values))
    cmap = plt.cm.get_cmap('Blues')
    # plt.figure(figsize=(8, 6))
    # plt.scatter(c_values, s_values, alpha=0.7, edgecolors='k', label=f"Unstable Configurations" )
    if state != 'Fixation' and state != 'Loss':
        scatter = ax.scatter(c_values, s_values, c=eq_values, cmap=cmap, s=50, norm=norm)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Gene Drive Allele Frequency at Eq')
    else: 
        scatter = ax.scatter(c_values, s_values, color = '#8488d1', s=50, norm=norm)

    # Adding labels, title, and legend
    ax.set_xlabel('Conversion Factor (c)', fontsize=12)
    ax.set_ylabel('Selection Coefficient (s)', fontsize=12)
    # plt.title(f'Scatter Plot of {state} Configurations', fontsize=14)
    ax.set_title(f"Equilibrium Values in the {state} Regime at gd_h = {h_val}", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    # plt.legend(fontsize=10)

    # Show plot
    plt.grid(True)
    plt.savefig(f"plot_partition/h{h_val}_{state}_{conversion}.jpg", dpi=600)
    plt.show()

#######################################################################################
# NGD stability 
#######################################################################################


# ────────────────────────────
# Scan grid, compute stability & second derivative at each eq
# ────────────────────────────

def compute_ngd_stability_grid(s_range, h_range):
    """
    Returns a DataFrame with columns:
      s, h, q_eq, f1 (dq'/dq), f2 (d^2 q'/dq^2), regime
    """
    rows = []
    for s_val in s_range:
        for h_val in h_range:
            # find interior equilibria
            eqs = get_eq_ngd(s_val, h_val)
            
            # if no interior eq, treat q=0 or q=1 as regimes
            if not eqs:
                # classify loss/fixation regimes
                q_init = 0.001
                res = wm(s_val, h_val, 40000, q_init)
                res_state = res['state'] 
                if res_state == 'loss':
                    regime = 'loss'
                elif res_state == 'fix':
                    regime = 'fixation'
                rows.append({
                    's': s_val, 'h': h_val,
                    'q_eq':   0.0 if regime == 'loss' else 1.0,
                    'f1':     f1_num(0.0, s_val, h_val),
                    'f2':     f2_num(0.0, s_val, h_val),
                    'regime': regime
                })
                continue

            # otherwise each interior eq
            # print(f"Computing stability for s={s_val}, h={h_val}, eqs={eqs}")
            for q_eq in eqs:
                f1_val = f1_num(q_eq, s_val, h_val)
                f2_val = f2_num(0, s_val, h_val)
                # classify stability
                if abs(f1_val) < 1:
                    regime = 'stable'
                elif abs(f1_val) > 1:
                    regime = 'unstable'
                else:
                    regime = 'neutral'
                rows.append({
                    's':     s_val,
                    'h':     h_val,
                    'q_eq':  q_eq,
                    'f1':    f1_val,
                    'f2':    f2_val,
                    'regime':regime
                })
    return pd.DataFrame(rows)


def compute_lambda():
    q, s, h = sp.symbols('q s h')

    numer = q**2*(1 - s) + q*(1 - q)*(1 - h*s)
    wbar = q**2*(1 - s) + 2*q*(1 - q)*(1 - h*s) + (1 - q)**2
    q_next = numer / wbar

    # 1st and 2nd derivatives wrt q
    f1 = sp.simplify(sp.diff(q_next, q))
    f2 = sp.simplify(sp.diff(f1, q))

    # the polynomial whose roots are the equilibria:
    #   solve numer/q_next - q = 0  ⇒  numer - q*wbar = 0
    poly_eq = sp.simplify(sp.expand(numer - q*wbar))
    poly_q  = sp.Poly(poly_eq, q)
    coeffs  = poly_q.all_coeffs()      # [a2(s,h), a1(s,h), a0(s,h)]

    # lambdify everything
    f1_num       = sp.lambdify((q, s, h), f1,  'numpy')
    f2_num       = sp.lambdify((q, s, h), f2,  'numpy')
    coef_funcs   = [sp.lambdify((s,h), c, 'numpy') for c in coeffs]

    # Print the simplified result
    sp.pprint(f1_num)
    return coef_funcs, f1_num, f2_num

coef_funcs, f1_num, f2_num = compute_lambda()

def get_ngd_stability(s, h, q, f1_num):
    try:
        slope = f1_num(s, h, q)
    except ZeroDivisionError:
        slope = 'NA'
        print(f"Division by zero: config = {(s, h, q)}")
    return slope

'''
Given the NGD parameter (s, h), return the stable eq
'''
def get_eq_ngd(s, h):

    if not math.isclose(h, 0.5):
        if (math.isclose(2*h*s - s, 0.0)):
            print(f'Division by zero: config = {(s, h)}')
            return None
        eq = (h*s)/(2*h*s - s)
        if eq < 0 or eq > 1 or math.isclose(eq, 0.0) or math.isclose(eq, 1.0):
            return None
        else:
            return [eq]
    return None

print(get_eq_ngd(-0.2, 0.1))

def plot_ngd_partition(df, q_init=None, save_path=None):
    """
    Plot the NGD stability partition from a DataFrame `df` with columns
      ['s','h','q_eq','f1','f2','regime'].
    Colors each point by its 'regime' category.
    """
    palette = {
        'loss'      : 'lightgray',
        'fixation'  : 'black',
        'stable'    : 'tab:blue',
        'unstable'  : 'tab:red',
        'neutral'   : 'tab:orange',
    }
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    # scatter with continuous colour = q_eq
    sc = ax.scatter(
        df['s'], df['h'],
        c=df['q_eq'],              # colour by equilibrium frequency
        cmap='viridis',              # warm → cool dark
        norm=mcolors.Normalize(vmin=0, vmax=1),
        s=30,                      # point size
        # edgecolors='k', lw=0.3,    # black border for clarity
        alpha=0.8
    )

    # add colourbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Equilibrium frequency $q_{eq}$', fontsize=12)

    
    # axes labels, title, grid, legend
    ax.set_xlabel('Selection coefficient $s$', fontsize=12)
    ax.set_ylabel('Dominance $h$', fontsize=12)
    
    title = "NGD Stability Partition by Equilibrium Frequency"
    if q_init is not None:
        title += f" (q0={q_init})"
    ax.set_title(title, fontsize=14)
    
    # ax.legend(title='Regime', loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', 'box')
    
    # determine save path
    if save_path is None:
        if q_init is not None:
            save_path = f"plot_partition/NGD_partition_sim_q{q_init}.jpg"
        else:
            save_path = "plot_partition/NGD_partition.jpg"
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)


def main():
    s=0.6
    c=0.6
    h=0.7
    q_init=0.001
    regime = 'Fixation'
    params = {'config':(s, c, h), 'q0':q_init, 'conversion': "gametic"}
    stability_subdir = ""
    filename = f"new_h{h}_{params['conversion']}_stability_res.pickle"

    ##########################################################
    # OLD VERSION OF PLOTTING GD PARTITION
    # run through all configurations
    allres = runall(params)
    save_pickle(stability_subdir, filename, allres) ### change file name
    
    allres = load_pickle(stability_subdir, filename)

    # print(allres)
    f_out = open(f"stability_res/h{h}_g_{regime}.txt", 'w') #### change file name!!!!
    f_out.write(f"gene drive model configuration\t\tequilibrium\n")
    for state in allres.keys():
        if state == regime:  ### change state check
            for config, eq_val in allres[state]:
                f_out.write(f"(s, c, h) = {config}\t\teq = {eq_val}\n")
    # plot_regions(allres, regime, params['conversion'])
    #
    plot_all_by_regime(allres, params)
    

    ### PLOT NGD PARTITION
    q_init = 0.001
    # file = f"ngd_sim_stability_q{q_init}.pickle"
    # s_vals = np.round(np.linspace(-2, 2, 81), 3)
    # h_vals = np.round(np.linspace(-2, 2, 81), 3)
    # df_stability = compute_ngd_stability_grid(s_vals, h_vals)

    # df_stability.to_csv("stability_res/NGD_stability_partition.csv", index=False)
    df_stab = pd.read_csv("stability_res/NGD_stability_partition.csv")
    # plot_ngd_partition(df_stab)



if __name__ == '__main__':
    main()

