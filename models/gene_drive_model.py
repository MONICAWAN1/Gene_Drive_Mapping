import numpy as np
import math

"""
run_model: takes params (s, c, h, target_steps, q0), runs a single gene drive simulation,
returns dict with 'q', 'p', 'w_bar' curves only.
"""
def run_model(params):
    s = params['s']
    c = params['c']
    h = params['h']
    ts = params['target_steps']
    s_c = (1 - h * s) * c
    s_n = 0.5 * (1 - h * s) * (1 - c)
    wtfreqs = np.zeros(ts)
    freqs = np.zeros(ts)
    w = np.zeros(ts)
    freqs[0] = params['q0']
    wtfreqs[0] = 1 - params['q0']
    final = ts

    for t in range(ts - 1):
        curr_q = freqs[t]
        curr_p = wtfreqs[t]
        w_bar = curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + 2 * s_n) + (1 - curr_q)**2
        w[t] = w_bar
        freqs[t + 1] = (curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + s_n)) / w_bar
        wtfreqs[t + 1] = 1 - freqs[t + 1]

        if math.isclose(freqs[t + 1], 1) or math.isclose(freqs[t + 1], 0) or math.isclose(curr_q, freqs[t + 1], rel_tol=1e-6):
            final = t + 1
            break

    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final - 1]}


