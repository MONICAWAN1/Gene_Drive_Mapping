#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import math
from ngd_model import ngd_simulation

#%%
# ======= USER PARAMETERS ==========================================
alpha      = 2.5                 # relative fitness of your favoured allele (0 < α < 1)
m_range    = (0.0, 1.0)           # search domain for migration rate
s_range    = (0.0, 1.0)           # search domain for selection coefficient
a_range = (0.0, 2.0)
ms_range = (0.0, 10.0)
resolution = 300                # number of grid points along each axis
# ==================================================================

# build grid
m_vals = np.linspace(max(1e-6, m_range[0]), m_range[1], resolution)
s_vals = np.linspace(max(1e-6, s_range[0]), s_range[1], resolution)
a_vals = np.linspace(max(1e-6, a_range[0]), a_range[1], resolution)
ms_vals = np.linspace(max(1e-6, ms_range[0]), ms_range[1], resolution)
M, S = np.meshgrid(m_vals, s_vals, indexing="xy")
A, MS = np.meshgrid(a_vals, ms_vals, indexing="xy")

#%%
# # Gene-swamping criterion (from the paper):
# crit_ratio = alpha / abs(1 - alpha)
# gene_swamp = (M / S) > crit_ratio     # Boolean mask (True/False)
gs_ratio = MS > A / abs(1-A)

#%%
label = 'msratio'
if label == "simulation":
# prep output array
    partition = np.zeros((resolution, resolution), dtype=int)

    #--------------------------------------------------------------
    # fill partition by simulation
    eps = 1e-3   # threshold for “extinct”
    for i in range(resolution):
        for j in range(resolution):
            m = M[i,j]
            s = S[i,j]

            params = {
                'q1': 0.8,        
                'q2': 0.2,         
                'target_steps': 1000,
                's': s,            
                'c': 0.0,         
                'h': 1.0,          
                'm': m,            
                'alpha': alpha     
            }

            result = ngd_simulation(params)
            q1_end = result['q1'][-1]
            q2_end = result['q2'][-1]

            # mark 1 if both go to (near) zero → gene swamping
            if q1_end < eps and q2_end < eps:
                partition[i,j] = 1

#--------------------------------------------------------------
# plot
#%%
if label == "msratio":
    partition = np.zeros_like(gs_ratio, dtype=int)       # default 0
    partition[(gs_ratio) & (A < 1)]  = 1                # α < 1  &  GS
    partition[(gs_ratio) & (A > 1)]  = 2                # α > 1  &  GS
    # (Points with α == 1 never satisfy GS, so stay 0)

    # ---------------------------------------------------------------
    # 3-colour map  (white, light-blue, dark-blue)
    cmap  = ListedColormap(['white', '#9ecae1', '#08519c'])
    norm  = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)   # bins centre on 0,1,2

    plt.figure(figsize=(6,5))
    plt.imshow(partition,
            extent=[*a_range, *ms_range],   # x = α, y = m/s
            origin='lower', aspect='auto',
            cmap=cmap, norm=norm)

    # ---------------------------------------------------------------
    # theoretical red line separating regions (unchanged from yours)
    # x_vals = a_vals
    # mask1  = x_vals < 1
    # mask2  = x_vals > 1

    # y_vals1 = np.zeros_like(x_vals)
    # y_vals2 = np.zeros_like(x_vals)
    # y_vals1[mask1] = x_vals[mask1] / (1 - x_vals[mask1])
    # y_vals2[mask2] = x_vals[mask2] / (x_vals[mask2] - 1)

    # # Restrict y-values to be less than 1.0
    # mask1_final = mask1 & (y_vals1 < ms_range[1])
    # mask2_final = mask2 & (y_vals2 < ms_range[1])

    # plt.plot(x_vals[mask1_final], y_vals1[mask1_final], 'r-',  lw=2, label=r'$\alpha<1$ line')
    # plt.plot(x_vals[mask2_final], y_vals2[mask2_final], 'r-', lw=2, label=r'$\alpha>1$ line')

    # ---------------------------------------------------------------
    # Manual legend patches
    legend_patches = [
        Patch(facecolor='white',      edgecolor='k', label='No gene swamping'),
        Patch(facecolor='#9ecae1',    edgecolor='k', label=r'GS if $\alpha<1$'),
        Patch(facecolor='#08519c',    edgecolor='k', label=r'GS if $\alpha>1$')
    ]
elif label == "simulation":
    plt.imshow(partition,
            extent=[m_range[0], m_range[1], s_range[0], s_range[1]],
            origin='lower', aspect='auto',
            cmap='BuPu')      # 1=white (swamp), 0=black
    x_label = "Migration rate (m)"
    y_label = "Selection coefficient (s)"
    title = fr"Partition of $(m,s)$ for $\alpha={alpha}$"

    # overlay theoretical line: m = (α/(1−α))·s
    crit_ratio = alpha / abs(1 - alpha)
    s_line = m_vals / crit_ratio
    mask = s_line <= s_range[1]
    plt.plot(m_vals[mask], s_line[mask], 'r-', lw=2, label=f'$s = (1-{alpha})/{alpha} \\cdot m$')

y_label = "Migration rate/Selection (m/s)"
x_label = fr"alpha $(\alpha)$"
title = fr"Partition of $(m/s, \alpha)$"
plt.legend(handles=legend_patches, loc='upper right')
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.tight_layout()
plt.savefig(f"ngd_{label}_alpha{a_range[0]}_{a_range[1]}_partition.png", dpi=600, bbox_inches="tight")
plt.show()

