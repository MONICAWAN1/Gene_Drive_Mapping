# %%
import sympy
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# %%
q = sympy.Symbol('q')
s = sympy.Symbol('s')
c = sympy.Symbol('c')
h = sympy.Symbol('h')

expr_gd = (q**2 * (1-s) + q * (1-q) * (1+c) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2)

# %%
q1, q2, q3 = sympy.solvers.solve(expr_gd - q, q)

# solve for s at q3 = 0 and q3 = 1
s_q3_0 = sympy.solvers.solve(q3, s)[0]
s_q3_1 = sympy.solvers.solve(q3 - 1, s)[0]

# find dq and evaluate at q3, if > 1 then unstable 
expr_gd_derivative = sympy.diff(expr_gd, q) 
expr_gd_derivative_q3 = sympy.simplify(expr_gd_derivative.subs(q, q3))

s_q3_0_lambda = sympy.lambdify((c, h), s_q3_0, 'numpy')
s_q3_1_lambda = sympy.lambdify((c, h), s_q3_1, 'numpy')
q3_lambda = sympy.lambdify((s, c, h), q3, 'numpy')
derivative_q3_lambda = sympy.lambdify((s, c, h), expr_gd_derivative_q3, 'numpy')
expr_gd_0_5_lambda = sympy.lambdify((s, c, h), expr_gd.subs(q, 0.5), 'numpy')

# expr_gd_derivative2 = sympy.diff(expr_gd_derivative, q)
# expr_gd_derivative2_0_5 = sympy.simplify(expr_gd_derivative2.subs(q, 0.5))
# derivative2_0_5_lambda = sympy.lambdify((s, c, h), expr_gd_derivative2_0_5, 'numpy')
# derivative2_0_5_mesh = derivative2_0_5_lambda(s_mesh, c_mesh, h_chosen)

# %%
partition_map = {
    0: 'stable',
    1: 'unstable',
    2: 'dq=1',
    3: 'fixation',
    4: 'loss'
}

## Defining colormaps for plotting
cmap = plt.cm.get_cmap('Set3')
cmap = ListedColormap(cmap([0,1,2,3,4]))
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # Create boundaries between -1, 0, 1
norm = BoundaryNorm(bounds, cmap.N)

# %%
# h_chosen_list = list(np.arange(0.0, 1.1, 0.1))
h_chosen_list = [0.4, 0.45]
print(h_chosen_list)

## Generate h_gametic_stability_res.pickle file for all configs with same h_val 
# h_chosen = 0.0
for h_chosen in h_chosen_list:
    c_chosen = np.arange(0.01, 1.01, 0.01)
    s_chosen = np.arange(0.01, 1.01, 0.01)

    c_mesh, s_mesh = np.meshgrid(c_chosen, s_chosen)
    partition = np.zeros_like(c_mesh)
    partition = partition.astype(int)

    q3_mesh = q3_lambda(s_mesh, c_mesh, h_chosen)
    derivative_q3_mesh = derivative_q3_lambda(s_mesh, c_mesh, h_chosen)
    expr_gd_0_5_mesh = expr_gd_0_5_lambda(s_mesh, c_mesh, h_chosen)

    s_q3_0_line = s_q3_0_lambda(c_chosen, h_chosen)
    s_q3_1_line = s_q3_1_lambda(c_chosen, h_chosen)

    # stable 0
    partition[((0 < q3_mesh) * (q3_mesh < 1)) * (derivative_q3_mesh < 1)] = 0

    # unstable 1
    partition[((0 < q3_mesh) * (q3_mesh < 1)) * (derivative_q3_mesh > 1)] = 1

    # dq=1 2
    # partition[((0 < q3_mesh) * (q3_mesh < 1)) * (derivative_q3_mesh == 1)] = 2

    # fixation 3
    partition[((q3_mesh >= 1) + (q3_mesh <= 0)) * (expr_gd_0_5_mesh > 0.5)] = 3

    # loss 4
    partition[((q3_mesh >= 1) + (q3_mesh <= 0)) * (expr_gd_0_5_mesh < 0.5)] = 4

    partition_unique = np.unique(partition)
    # partition_unique = partition_unique.astype(int)

    plotting=False
    if plotting: 
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(c_mesh, s_mesh, partition, cmap=cmap, norm=norm, shading='auto')

        # Overlay the curve
        if h != 0 and h != 0.5:
            plt.plot(c_chosen, s_q3_0_line, color='black', linewidth=1)
        if h != 1:
            plt.plot(c_chosen, s_q3_1_line, color='black', linewidth=1)
        # plt.colorbar(ticks=boundaries)

        legend_elements = [
            Patch(facecolor = col, edgecolor='black', label=partition_map[i_c]) for i_c, col in enumerate(cmap.colors) if i_c in partition_unique
        ]

        # plt.legend(handles=legend_elements, title='Regime', loc='upper right')

        # # Decorations
        # plt.xlabel(r'Conversion Factor, $c$')
        # plt.ylabel(r'Selection Coeffiecient, $s$')
        # plt.title(f"h = {h_chosen}")
        # # plt.legend()
        # plt.axis('equal')
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.tight_layout()
        # plt.savefig(f'plot_partition/GD_partition_h_{h_chosen}.pdf', dpi=600, bbox_inches='tight')
        # plt.show()
        # plt.close()

    # --- NEW: build regimes dict ---
    regimes = {
        'stable':   [],   # internal eq, derivative<1
        'unstable': [],   # internal eq, derivative>1
        'fixation': [],   # drives to 1
        'loss':     []    # drives to 0
    }

    # Walk over every grid cell
    n_rows, n_cols = partition.shape
    for i in range(n_rows):
        for j in range(n_cols):
            code = partition[i, j]
            s_val = s_mesh[i, j]
            c_val = c_mesh[i, j]
            s_val, c_val = round(float(s_val), 3), round(float(c_val), 3)

            if code == 0:           # stable
                eq = q3_mesh[i, j]
                regimes['stable'].append(((s_val, c_val, h_chosen), eq))

            elif code == 1:         # unstable
                eq = q3_mesh[i, j]
                regimes['unstable'].append(((s_val, c_val, h_chosen), eq))

            elif code == 3:         # fixation
                eq = 1.0
                regimes['fixation'].append(((s_val, c_val, h_chosen), eq))

            elif code == 4:         # loss
                eq = 0.0
                regimes['loss'].append(((s_val, c_val, h_chosen), eq))
        
    
    # Compile all configs in the same regime together
    target_regime = "stable"
    fout = f"{h_chosen}_{target_regime}_param.txt"
    with open(fout, "a") as fout:
        for config, eq in regimes[target_regime]:
            s,c,h = config
            s, c, h = round(float(s), 3), round(float(c), 3), round(float(h), 3)
            fout.write(f"{s:.3f} {c:.3f} {h:.3f}\n")

    #%%
    # --- OLD VERSION: save to pickle and write to txt ---
    # out_path = f"../pickle/h{h_chosen}_gametic_stability_res.pickle"
    # with open(out_path, "wb") as pf:
    #     pickle.dump(regimes, pf)

    # print(f"Saved regimes for h={h_chosen} → {out_path}")

    #%%
    # Write all configurations in the same regime to its txt in stability_res folder
    # h = 1.0
    # regime = "unstable"
    # f_out = open(f"stability_res/h{h}_g_{regime}.txt", 'w') #### change file name!!!!
    # f_out.write(f"gene drive model configuration\t\tequilibrium\n")
    # for state in regimes.keys():
    #     if state == regime:  ### change state check
    #         for config, eq_val in regimes[state]:
    #             s, c, h = config
    #             s, c, h = round(float(s), 3), round(float(c), 3), round(float(h), 3)
    #             f_out.write(f"(s, c, h) = {(s, c, h)}\t\teq = {eq_val}\n")
    
    # print(f"Partition written to {f_out}")
    # #%%
    # # -----------------------------------------------------------------
    # # write the dictionary to a text file
    # # -----------------------------------------------------------------
    # os.makedirs("phase_partition", exist_ok=True)
    # h_val = h
    # fname = f"phase_partition/partition_h{h_val}.txt"
    # with open(fname, "w") as fout:
    #     for regime in ['stable', 'unstable', 'fixation', 'loss']:
    #         fout.write(f"## {regime}\n")
    #         for (cfg, eq) in regimes[regime]:
    #             s_val, c_val, h_val = cfg
    #             fout.write(f"s={s_val:.3f}, c={c_val:.3f}, h={h_val:.3f}, eq={eq:.4f}\n")
    #         fout.write("\n")

    # print(f"Partition written to {fname}")

# %%


# UPDATE 11.05: analytics for NGD stable/unstable boundary 
#%%
import numpy as np
def wm(s, h, target_steps, q_init):
    q_freqs = np.zeros(target_steps)
    p_freqs = np.zeros(target_steps)
    q_freqs[0] = q_init # mutant
    p_freqs[0] = 1 - q_init
    w = np.zeros(target_steps)
    final = target_steps

    for t in range(target_steps-1):
        # freqs[t+1] = freqs[t] + s * freqs[t] * (1-freqs[t])
        curr_q = q_freqs[t]
        curr_p = p_freqs[t]
        w_bar = curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (1-h*s) + (1 - curr_q)**2
        w[t] = w_bar
        q_freqs[t+1] = (curr_q**2 * (1 - s) + curr_q * (1 - curr_q) * (1 - h * s)) / w_bar
        p_freqs[t+1] = (curr_p**2 + curr_p * (1 - curr_p) * (1 - h * s)) / w_bar

        # print("step=%2d, currq=%.8f, q=%.20f, p=%.8f"%(t, q_freqs[t], q_freqs[t+1], p_freqs[t+1]))

        if (q_freqs[t+1] < 0 or q_freqs[t+1] > 1 or math.isclose(q_freqs[t+1], 1) or math.isclose(q_freqs[t+1], 0) 
            or math.isclose(curr_q, q_freqs[t+1], rel_tol=1e-6)):
            if t == 0:
                print("STOP AT step=%2d, w=%.4f, q=%.20f, p=%.8f, s=%.4f, h=%.4f"%(t, w[t], q_freqs[t+1], p_freqs[t+1], s, h))
            final = t+1
            break

    # return {'q': q_freqs[:final], 'w_bar': w[:final-1]}
    return {'q': q_freqs[:final]}
#%%
# LEARNED: in the equilibrium state, fix/loss has nothing to do with s, q3 only depends on h
import sympy as sp 
import math
#%%
q = sp.Symbol('q')
s = sp.Symbol('s')
h = sp.Symbol('h')

expr_ngd = (q**2 * (1-s) + q * (1-q) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2) 

ngd_q1, ngd_q2, ngd_q3 = sp.solvers.solve(expr_ngd - q, q)
q3_lam = sp.lambdify(h, ngd_q3, 'numpy')

# find dq at q3 = 1, solve it for s
expr_ngd_dq = sp.simplify(sp.diff(expr_ngd, q))
# dq'/dq | (q = q3)
# Find values of s and h (ngd) that make this value = 1 => boundary of stable and unstable 
dq_at_q3 = sp.simplify(expr_ngd_dq.subs(q, ngd_q3)) # functino of s, h
boundary_eq = sp.Eq(dq_at_q3, 1)

# solve symbolically for s as a function of h
# expression for boundary s values between stable/unstable
s_boundary = sp.solve(boundary_eq, s)
h_boundary = sp.solve(boundary_eq, h)
s_boundary = [sp.simplify(expr) for expr in s_boundary]

s_boundary_lambda = sp.lambdify((h), s_boundary[0], 'numpy')
s_boundary_val = s_boundary_lambda(h)

#%%
h_val = -1.6
# sh <= 1
s_val = -2
q3 = q3_lam(h_val)
print('q3:', q3)
print('sval:', s_val)
print(wm(s_val, h_val, 10000, 0.1)['q'][-1])

# %%
