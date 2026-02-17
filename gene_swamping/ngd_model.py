import numpy as np
import math

def ngd_simulation(params):
    q1, q2, ts = params['q1'], params['q2'], params['target_steps']

    q1_freqs = np.zeros(ts)
    q2_freqs = np.zeros(ts)
    w1 = np.zeros(ts)
    w2 = np.zeros(ts)
    q1_freqs[0] = q1
    q2_freqs[0] = q2
    final = ts
    diploid = True if params['type'] == "diploid" else False
    # print("s and alpha in hapngd sim", params['s'], params['alpha'])
    # print("check fitness > 0", 1+params['alpha']*params['s'])

    for t in range(ts-1):
        # migration
        q1_freqs[t+1], q2_freqs[t+1], w1[t], w2[t] = ngd_step(params, q1_freqs[t], q2_freqs[t], diploid)
        # if t<=5:
        #     print(f"step={t}", q1_freqs[t+1], q2_freqs[t+1], {w1[t]}, {w2[t]})

        if math.isclose(q1_freqs[t+1], 1) or math.isclose(q1_freqs[t+1], 0) or math.isclose(q1, q1_freqs[t+1], rel_tol=1e-6):
            final = t+1
            break
    return {'q1': q1_freqs[:final], 'q2': q2_freqs[:final], 'w_bar': (w1[:final-1], w2[:final-1])}

def ngd_step(params, q1, q2, diploid):
    s, c, hvals, m, alpha = params['s'], params['c'], params['h'], params['m'], params['alpha']
    h1, h2 = hvals
    # Migration
    q1_m = m * q2 + (1 - m) * q1
    q2_m = m * q1 + (1 - m) * q2
    
    # Selection
    if diploid:
        w_bar1 = q1_m**2 * (1 - s) + 2 * q1_m * (1 - q1_m) * (1-h1*s) + (1 - q1_m)**2
        w_bar2 = q2_m**2 * (1 - alpha*s) + 2 * q2_m * (1 - q2_m) * (1 - h2*alpha*s) + (1 - q2_m)**2

        q1_next = (q1_m**2 * (1 - s) + q1_m * (1 - q1_m) * (1 - h1 * s)) / w_bar1
        q2_next = (q2_m**2 * (1 - alpha*s) + q2_m * (1 - q2_m) * (1 - h2 * alpha*s)) / w_bar2
    else:
        # Haploid selection
        w_bar1 = q1_m * (1 - s) + (1 - q1_m)
        w_bar2 = q2_m * (1 - alpha*s) + (1 - q2_m)

        q1_next = q1_m * (1 - s) / w_bar1
        # q2_next = q2_m
        q2_next = q2_m * (1 - alpha*s) / w_bar2

    return q1_next, q2_next, w_bar1, w_bar2

import matplotlib.pyplot as plt

def run_and_plot_simulation():
    # Initialize parameters
    params = {
        'q1': 0.8,          # Initial frequency in population 1
        'q2': 0.2,          # Initial frequency in population 2
        'target_steps': 1000,
        's': 0.1,           # Selection coefficient
        'c': 0.0,           # Not used in current model
        'h': 0.9,           # Dominance coefficient
        'm': 0.2,          # Migration rate
        'alpha': 2.5        # Relative fitness of the favoured allele
    }

    # Run simulation
    result = ngd_simulation(params)
    q1_traj = result['q1']
    q2_traj = result['q2']

    # Plot trajectories
    plt.figure(figsize=(8, 5))
    plt.plot(q1_traj, label='Population 1 (q1)')
    plt.plot(q2_traj, label='Population 2 (q2)')
    plt.xlabel('Time step')
    plt.ylabel('Allele frequency')
    plt.title('Gene Drive Simulation Trajectory')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Run the simulation and plot the results
    run_and_plot_simulation()

if __name__ == "__main__":
    main()