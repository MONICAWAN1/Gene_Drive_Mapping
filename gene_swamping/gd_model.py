### Gene Drive Model for Gene Swamping ###
import numpy as np
import matplotlib.pyplot as plt
import math

def one_step(params, q1, q2):
    '''
    This function performs one step of the gene drive simulation.
    
    params: dictionary containing parameters s, c, h, m, q1, q2
    q1: frequency of the gene drive allele
    q2: frequency of the wild-type allele
    '''
    s1, c1, h, m = params['s'], params['c'], params['h'], params['m']
    s2 = s1 * params['alpha']
    c2 = c1 * params['beta']
    # Migration
    q1_m = m * q2 + (1 - m) * q1
    q2_m = m * q1 + (1 - m) * q2
    
    # Selection
    s_c1 = (1 - h * s1) * c1  # Gametic selection coefficient
    s_n1 = 0.5 * (1 - h * s1) * (1 - c1)  # Zygotic selection coefficient
    s_c2 = (1 - h * s2) * c2  # Gametic selection coefficient
    s_n2 = 0.5 * (1 - h * s2) * (1 - c2)  # Zygotic selection coefficient
    
    w_bar1 = q1_m**2 * (1 - s1) + 2 * q1_m * (1 - q1_m) * (s_c1 + 2 * s_n1) + (1 - q1_m)**2
    w_bar2 = q2_m**2 * (1 - s2) + 2 * q2_m * (1 - q2_m) * (s_c2 + 2 * s_n2) + (1 - q2_m)**2
    
    # Update frequencies
    q1_next = (q1_m**2 * (1 - s1) + 2 * q1_m * (1 - q1_m) * (s_c1 + s_n1)) / w_bar1
    q2_next = (q2_m**2 * (1 - s2) + 2 * q2_m * (1 - q2_m) * (s_c2 + s_n2)) / w_bar2
    
    return q1_next, q2_next, w_bar1, w_bar2

def gd_simulation(params):
    '''
    This model calculates the allele frequency dynamics of a 2-deme gene drive 
    system given a set of parameters.

    s: selection coefficient
    c: cost of the gene drive
    h: fitness of the homozygous drive genotype
    m: migration rate
    q1: fitness of the heterozygous drive genotype
    q2: fitness of the wild-type genotype
    '''
    q1, q2, ts = params['q1'], params['q2'], params['target_steps']

    q1_freqs = np.zeros(ts)
    q2_freqs = np.zeros(ts)
    w1 = np.zeros(ts)
    w2 = np.zeros(ts)
    q1_freqs[0] = q1
    q2_freqs[0] = q2
    final = ts

    for t in range(ts-1):
        # migration
        q1_freqs[t+1], q2_freqs[t+1], w1[t], w2[t] = one_step(params, q1_freqs[t], q2_freqs[t])

        if math.isclose(q1_freqs[t+1], 1) or math.isclose(q1_freqs[t+1], 0) or math.isclose(q1, q1_freqs[t+1], rel_tol=1e-6):
            final = t+1
            break
    return {'q1': q1_freqs[:final], 'q2': q2_freqs[:final], 'w_bar': (w1[:final-1], w2[:final-1])}

