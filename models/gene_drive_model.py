import numpy as np
import math
import pickle

from utils import at_eq
#### run_model: takes in params (with a fixed sch), runs a single gene drive simulation, returns a dictionary of q arrays and wbar array
def run_model(params):
    s = params['s']
    c = params['c']
    h = params['h']
    ts = params['target_steps']
    state = None
    # s_c = (1 - h * s) * c ## Gametic on modelrxiv
    # s_c = (1 - s) * c ## zygotic in paper
    s_c = (1 - h*s) * c ## gametic
    s_n = 0.5 * (1 - h * s) * (1 - c) # when c = 1, h doesn't matter 
    wtfreqs = np.zeros(params['target_steps'])
    freqs = np.zeros(params['target_steps'])
    w = np.zeros(params['target_steps'])
    freqs[0] = params['q0']
    wtfreqs[0] = 1 - params['q0']
    final = ts

    for t in range(ts-1):
        curr_q = freqs[t] # mutant
        curr_p = wtfreqs[t] # wildtype
        w_bar = curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + 2 * s_n) + (1 - curr_q)**2
        w[t] = w_bar
        freqs[t+1] = (curr_q**2 * (1 - s) + 2 * curr_q * (1 - curr_q) * (s_c + s_n)) / w_bar
        # wtfreqs[t+1] = (curr_p**2 + 2 * curr_p * (1 - curr_p) * s_n) / w_bar
        wtfreqs[t+1] = 1-freqs[t+1]

        if math.isclose(freqs[t+1], 1) or math.isclose(freqs[t+1], 0) or math.isclose(curr_q, freqs[t+1], rel_tol=1e-6):
            final = t+1
            break
        # if not math.isclose(freqs[t+1] + wtfreqs[t+1], 1.0):
        #     print(freqs[t+1], wtfreqs[t+1], t)
    # print(freqs, wtfreqs)
    state = checkState(final, freqs, params)
    if state == None: print('none', freqs[t], freqs[t+1])

    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final-1], 'state': state}

def checkState(final, freqs, params):
    if freqs[final-1] >= 0.99: state = 'fix'
    elif freqs[final-1] <= 0.01: 
        if freqs[final-1] > freqs[final-2]:
            state = 'fix'
        else: state = 'loss'
    elif at_eq(freqs): state = 'stable'
    else: 
        state = 'unstable'
    return state


