import numpy as np
import math

def wm(params):
    """
    NGD diploid recurrence; params: s, h, target_steps, q0.
    Returns dict with 'q', 'p', 'w_bar' only.
    """
    s = params['s']
    h = params['h']
    target_steps = params['target_steps']
    q_init = params['q0']
    q_freqs = np.zeros(target_steps)
    p_freqs = np.zeros(target_steps)
    q_freqs[0] = q_init
    p_freqs[0] = 1 - q_init
    w = np.zeros(target_steps)
    final = target_steps

    for t in range(target_steps - 1):
        curr_q = q_freqs[t]
        curr_p = p_freqs[t]
        w_bar = curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (1 - h * s) + (1 - curr_q)**2
        w[t] = w_bar
        q_freqs[t + 1] = (curr_q**2 * (1 - s) + curr_q * (1 - curr_q) * (1 - h * s)) / w_bar
        p_freqs[t + 1] = (curr_p**2 + curr_p * (1 - curr_p) * (1 - h * s)) / w_bar

        if (q_freqs[t + 1] < 0 or q_freqs[t + 1] > 1 or math.isclose(q_freqs[t + 1], 1) or math.isclose(q_freqs[t + 1], 0)
                or math.isclose(curr_q, q_freqs[t + 1], rel_tol=1e-6)):
            if t == 0:
                print("STOP AT step=%2d, w=%.4f, q=%.20f, p=%.8f, s=%.4f, h=%.4f" % (t, w[t], q_freqs[t + 1], p_freqs[t + 1], s, h))
            final = t + 1
            break

    return {'q': q_freqs[:final], 'p': p_freqs[:final], 'w_bar': w[:final - 1]}



def haploid_se(params):
    s, c, h = params['s'], params['c'], params['h']
    s = h * s - c + c * h * s
    ts = params['target_steps']
    freqs = np.zeros(ts)
    wtfreqs = np.zeros(ts)
    w = np.zeros(ts)
    freqs[0] = params['q0']
    wtfreqs[0] = 1 - params['q0']
    final = ts
    for t in range(ts - 1):
        curr_q = freqs[t]
        curr_p = wtfreqs[t]
        w_bar = curr_q * (1 - s) + curr_p
        w[t] = w_bar
        freqs[t + 1] = curr_q * (1 - s) / w_bar
        wtfreqs[t + 1] = 1 - freqs[t + 1]
        if freqs[t + 1] > 1 or math.isclose(freqs[t + 1], 1) or math.isclose(freqs[t + 1], 0) or math.isclose(curr_q, freqs[t + 1], rel_tol=1e-5):
            final = t + 2
            break
    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final - 1]}

def haploid(params):
    s = params['s']
    ts = params['target_steps']
    freqs = np.zeros(ts)
    wtfreqs = np.zeros(ts)
    w = np.zeros(ts)
    freqs[0] = params['q0']
    wtfreqs[0] = 1 - params['q0']
    final = ts
    for t in range(ts - 1):
        curr_q = freqs[t]
        curr_p = wtfreqs[t]
        w_bar = curr_q * (1 - s) + curr_p
        w[t] = w_bar
        freqs[t + 1] = curr_q * (1 - s) / w_bar
        wtfreqs[t + 1] = 1 - freqs[t + 1]
        if math.isclose(freqs[t + 1], 1) or math.isclose(freqs[t + 1], 0) or math.isclose(curr_q, freqs[t + 1], rel_tol=1e-5):
            final = t + 1
            break
    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final - 1]}
