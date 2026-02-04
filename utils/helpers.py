import numpy as np

def isclose(a, b, rel_tol=1e-7, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

#### Euclidean distance formula #############################
def euclidean(ngd, gd):
    diff1 = diff2 = 0
    q0 = ngd
    q0_m = gd
    if len(q0) < len(q0_m):
        q0 = np.append(q0, [q0[-1]] * (len(q0_m) - len(q0)))
    # else:
    #     q0_m = np.append(q0_m, [q0_m[-1]] * (len(q0) - len(q0_m)))

    for i in range(len(q0_m)):
        diff1 += (q0[i]-q0_m[i])**2
    return diff1/len(q0_m)

def at_eq(freqs):
    differences = [abs(freqs[i+1] - freqs[i]) for i in range(len(freqs)-1)]
    for diff in differences[-10:]:
        if diff >= 0.001:
            return False
    return True

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)