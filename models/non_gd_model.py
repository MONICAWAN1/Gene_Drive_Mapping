import numpy as np
import math
import pickle
from utils import at_eq

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

    state = checkState(final, q_freqs)
    if state == None: print('none', q_freqs[t], q_freqs[t+1])
    # return {'q': q_freqs[:final], 'w_bar': w[:final-1]}
    return {'q': q_freqs[:final], 'state': state}



#ngd model
def haploid_se(params):
    s,c,h = params['s'], params['c'], params['h']
    # s,c,h = 0.4, 0.6, 0
    # s = 1-c+c*s*(2-h)-h*s # zygotic
    s = h*s-c+c*h*s  #gametic se
    ts = params['target_steps']
    freqs = np.zeros(params['target_steps'])
    wtfreqs = np.zeros(params['target_steps'])
    w = np.zeros(params['target_steps'])
    freqs[0] = params['q0']
    wtfreqs[0] = 1- params['q0']
    final = ts
    for t in range(ts-1):
        curr_q = freqs[t] # mutant
        curr_p = wtfreqs[t] # wildtype
        w_bar = curr_q*(1-s) + curr_p
        w[t] = w_bar

        freqs[t+1] = curr_q*(1-s)/w_bar
        wtfreqs[t+1] = 1-freqs[t+1]

        if freqs[t+1] > 1 or math.isclose(freqs[t+1], 1) or math.isclose(freqs[t+1], 0) or math.isclose(curr_q, freqs[t+1], rel_tol=1e-5):
            final = t+2
            break

    state = checkState(final, freqs)
    if state == None: print('none', freqs[t], freqs[t+1])

    return {'q': freqs[:final], 'w_bar': w[:final-1], 'state': state}

# ngd haploid without se
def haploid(params):
    s = params['s']
    # s = 1-c+c*s*(2-h)-h*s
    ts = params['target_steps']
    freqs = np.zeros(params['target_steps'])
    wtfreqs = np.zeros(params['target_steps'])
    w = np.zeros(params['target_steps'])
    freqs[0] = params['q0']
    wtfreqs[0] = 1- params['q0']
    final = ts
    for t in range(ts-1):
        curr_q = freqs[t] # mutant
        curr_p = wtfreqs[t] # wildtype
        w_bar = curr_q*(1-s) + curr_p
        w[t] = w_bar

        freqs[t+1] = curr_q*(1-s)/w_bar
        wtfreqs[t+1] = 1-freqs[t+1]

        if math.isclose(freqs[t+1], 1) or math.isclose(freqs[t+1], 0) or math.isclose(curr_q, freqs[t+1], rel_tol=1e-5):
            final = t+1
            break

    state = checkState(final, freqs)
    if state == None: print('none', freqs[t], freqs[t+1])

    return {'q': freqs[:final], 'p': wtfreqs[:final], 'w_bar': w[:final-1], 'state': state}

def checkState(final, freqs):
    if freqs[final-1] >= 0.99: state = 'fix'
    elif freqs[final-1] <= 0.01: 
        # if freqs[final-1] > freqs[final-2]:
        #     state = 'fix'
        state = 'loss'
    elif at_eq(freqs): state = 'stable'
    else: 
        state = 'unstable'
    return state
