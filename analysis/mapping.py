import numpy as np
import math
import pickle
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sympy as sp
import concurrent.futures
from itertools import repeat

from .stability import derivative, compute_lambda_gd
# from .plotting import plot_lambda_curve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_pickle, save_pickle, euclidean
from models import haploid, run_model, wm

def get_eq(params):
    s, c, h = params['s'], params['c'], params['h']
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
Given the NGD parameter (s, h), return the stable eq
'''
def get_eq_ngd(s, h):
    if not math.isclose(h, 0.5):
        return (h*s)/(2*h*s - s)
    return 'NA'

# def get_ngd_stability(s, h, q):
#     num = 2*q**2*(1-q)*(1-s)*(1-h*s) - q**2*(1-2*q)*(1-s)*(1-h*s) + 2*q*(1-q)**2*(2-s-h*s) + (1-2*q)*(1-q)**2*(1-h*s)

#     denom = (q**2*(1-s) + 2*(1-q)*q*(1-h*s) + (1-q)**2)**2

#     if denom != 0:
#         slope = num/denom
#     else:
#         slope = 'NA'
#         print(f"no slope result, config = {(s, h, q)}")
    
#     return slope
q, s, h = sp.symbols('q s h')

# Define the symbolic expression for q(t+1)
numerator = q**2 * (1 - s) + q * (1 - q) * (1 - h * s)
wbar = q**2 * (1 - s) + 2 * q * (1 - q) * (1 - h * s) + (1 - q)**2
q_next = numerator / wbar

# Derivative of q(t+1) with respect to q(t)
dq_next_dq = sp.simplify(sp.diff(q_next, q))

# Convert the symbolic derivative into a Python function
dq_dq_func = sp.lambdify((s, h, q), dq_next_dq, "numpy")

# print(dq_dq_func)



def get_ngd_stability(s, h, q):
    try:
        slope = dq_dq_func(s, h, q)
    except ZeroDivisionError:
        slope = 'NA'
        print(f"Division by zero: config = {(s, h, q)}")
    return slope


'''
Check if current GD eq stability is the same as mapping NGD
'''
def check_state_eq(s, h, q, gdState):
    ngd_stability = get_ngd_stability(s, h, q)
    # print("++++++++++++++++++++++")
    # print(ngd_stability)
    if ngd_stability != 'NA':
        ngd_state = "Stable" if ngd_stability < 1 else "Unstable"
        return ngd_state == gdState
    return False

def loadStability(currH):
    return load_pickle("OLD", f"h{currH}_gametic_stability_res.pickle")

def loadGres(currH, gdFile):
    '''
    remember to change filename in getDiff before run map
    '''
    return load_pickle("gd_simulation_results", f"h{currH}_allgdres{gdFile}G.pickle")

def f(ngd_s, ngd_h, s, c, h, params, gdFile):
    gd_results = loadGres(h, gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']
    f = 0
    ngd_curve = wm(ngd_s, ngd_h, params['target_steps'], params['q0'])['q']

    # DEBUGGING PLOTS
    # if ngd_h != h:
    #     print(f"NGD_H: {ngd_h}, H: {h}")
    #     ngd_curve2 = wm(ngd_s, h, params['target_steps'], params['q0'])['q']
    #     plt.figure(figsize=(10, 8))
    #     time = np.arange(len(ngd_curve))
    #     time2 = np.arange(len(ngd_curve2))
    #     plt.plot(time, ngd_curve)
    #     plt.plot(time2, ngd_curve2)
    #     plt.xlabel("Time")
    #     plt.ylabel("Gene Drive Allele Frequency")
    #     plt.show()

    return euclidean(ngd_curve, gd_curve)

def optimization(ngd_s, s, c, gd_h, params):
    N = 0
    delta = 0.001
    curr_h = gd_h + delta
    new_h = curr_h

    # Newton-Raphson optimization based on f(h)
    while N < 1000:
        print(curr_h)
        f_h2 = f(ngd_s, curr_h+delta, s, c, gd_h, params)
        f_h1 = f(ngd_s, curr_h-delta, s, c, gd_h, params)
        f_h = f(ngd_s, curr_h, s, c, gd_h, params)
        print('f values', f_h, f_h1, f_h2)
        df = (f_h2 - f_h1) / (2*delta)
        print(df)
        if df == 0:
            break
        new_h = curr_h - f_h/df
        if f(ngd_s, new_h, s, c, gd_h, params) > f(ngd_s, curr_h, s, c, gd_h, params):
            break
        curr_h = new_h
        N += 1
    
    return curr_h

def optimization_s(ngd_s, s, c, gd_h, params):
    N = 0
    delta = 0.001
    curr_s = ngd_s + delta
    print('init', curr_s)
    new_s = curr_s

    # Newton-Raphson optimization based on f(h)
    while N < 10:
        curr_s = new_s
        print(curr_s)
        f_h2 = f(curr_s+delta, gd_h,  s, c, gd_h, params)
        f_h1 = f(curr_s-delta, gd_h, s, c, gd_h, params)
        f_h = f(curr_s, gd_h, s, c, gd_h, params)
        print('s values', f_h, f_h1, f_h2)
        df = (f_h2 - f_h1) / (2*delta)
        if df == 0:
            break
        if f(new_s, gd_h, s, c, gd_h, params) > f(curr_s, gd_h, s, c, gd_h, params):
            break
        new_s = curr_s - f_h/df
        N += 1
    
    return curr_s

def opt_r(F, grad_F, vars, s, c, h, params, gdFile, alpha=1e-4, beta=0.9, r_init=1.0):
    r = r_init
    while True:
        new_vars = vars - r * grad_F
        new_F = f(new_vars[0], new_vars[1], s, c, h, params, gdFile)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

### gradient haploid helper functions ##########################################################
def f_hap(ngd_s, s, c, h, params, gdFile):
    gd_results = loadGres(h, gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    gd_curve = gd_res[(s, c, h)]['q']

    # haploid mapping
    params['s'] = ngd_s
    ngd_curve = haploid(params)['q']
    f = 0
    return euclidean(ngd_curve, gd_curve)

def opt_r_hap(F, grad_F, vars, s, c, h, gdFile, params, alpha=1e-4, beta=0.0099, r_init=0.1):
    r = r_init
    while True:
        new_vars = vars - r * grad_F
        new_F = f_hap(new_vars[0], s, c, h, params, gdFile)
        if  new_F <= F - alpha * r * np.dot(grad_F, grad_F):
            return r
        r *= beta

### gradient haploid & diploid functions ##########################################################

def gradient(ngd_s, ngd_h, s, c, h, params, gdFile):
    vars = np.array([ngd_s, ngd_h])
    deltas = [0.001, 1]
    iterations = 1000
    grad_F = np.zeros(len(vars))
    F = 0
    prev_F = F+10
    trace = [(ngd_s, ngd_h)]
    fout = open(f"s{s}_c{c}_h{h}_gt.txt", 'w')

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f(vars[0], vars[1], s, c, h, params, gdFile)
        print(f'ITERATION {i}')
        if abs(F - prev_F) < 1e-8:
            print("Stopping: function change is too small.")
            break
        for var_i in range(len(vars)):
            vars_plus_delta = vars.copy()
            vars_plus_delta[var_i] += deltas[var_i]
            print(f"IN LOOP with {var_i}")
            f_s1 = f(vars_plus_delta[0], vars_plus_delta[1], s, c, h, params, gdFile)
            df_s = (f_s1 - F)/deltas[var_i]
            grad_F[var_i] = df_s

        
        # find step size minimalize f((s, h) - r*grad(s, h))
        print(grad_F)
        r = opt_r(F, grad_F, vars, s, c, h, params, gdFile)
        # r = 1
        print(r)
        new_vars = vars - r*grad_F
        print('1st', F, new_vars)

        # if new_vars[1] < 0 or new_vars[1] > 1:
        #     print(vars)
        #     break
        prev_F = F

        new_vars[0] = np.clip(new_vars[0], -100, 1) 
        new_vars[1] = np.clip(new_vars[1], 0, 1)  # Constraint on 'h'
        # if math.isclose(f(new_vars[0], new_vars[1], s, c, h, params), F, rel_tol=1e-5):
        #     break
        vars = new_vars
        trace.append((vars[0], vars[1]))
        fout.write(f"Iteration {i}\nGradient: {grad_F}\tStep: {r}\tNew Vars: s={new_vars[0]}, h={new_vars[1]}\tDiff: {F}\n")
        print('2nd', F, vars)
        # break
    
    save_pickle(f"s{s}_c{c}_h{h}_gradient_trace_r.pickle", trace)
    return vars

def hap_gradient(ngd_s, s, c, h, params, gdFile):
    vars = np.array([ngd_s])
    delta = 0.001
    iterations = 500
    grad_F = np.zeros(len(vars))
    F = 0
    prev_F = F+10

    for i in range(iterations):
        print('iteration', i, s, c, h)
        F = f_hap(vars[0], s, c, h, params, gdFile)
        if abs(F - prev_F) < 1e-8:
            print("Stopping: function change is too small.")
            break
        for var_i in range(len(vars)):
            vars_plus_delta = vars.copy()
            vars_plus_delta[var_i] += delta
            f_s1 = f_hap(vars_plus_delta[0], s, c, h, params, gdFile)
            df_s = (f_s1 - F)/delta
            grad_F[var_i] = df_s

        
        # find step size minimalize f((s, h) - r*grad(s, h))
        r = opt_r_hap(F, grad_F, vars, s, c, h, params, gdFile)
        print(r)
        new_vars = vars - r*grad_F
        print('1st', F, new_vars)

        # if new_vars[1] < 0 or new_vars[1] > 1:
        #     print(vars)
        #     break
        prev_F = F

        new_vars[0] = np.clip(new_vars[0], -100, 1) 
        # if math.isclose(f(new_vars[0], new_vars[1], s, c, h, params), F, rel_tol=1e-5):
        #     break
        vars = new_vars
        print('2nd', F, vars)
    return vars

def hap_grid_mapping(params, gdFile):
    '''
    Map haploid gene drive configurations to the stable NGD configurations.
    Write the results to a text file in the mapping_result_txt folder. 
    Add flag -s to store results in as .pickle under mapping_result
    '''
    seffMap = dict()
    hapMappedCurves = []
    f_out = open(f"mapping_result_txt/h{params['h']}_hap_grid{gdFile}_G_fix_map.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\ts in haploid ngd population\n")

    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = loadStability(params['h'])
    # print('params', gd_configs)

    hap_results = load_pickle(f"allhapresults/allhapres0001G.pickle")

    for (s, c, h) in gd_configs:
        # only testing for s = 0.4, c=0.6
        # if math.isclose(s, 0.01) and math.isclose(c, 0.01):
        # if gd_res[(s, c, h)]['state'] == 'fix':
        if ((s, c, h), 1.0) in stabilityRes['fixation']:
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = None
            best_h = h
        
            for ngd_key, ngd_curve in hap_results.items():
                diff = euclidean(ngd_curve['q'], gd_curve)
                if diff < best_diff:
                    best_diff = diff
                    best_s = ngd_key

            print('before gradient:', best_s)

            ### haploid
            hap_params = {'s': best_s, 'q0': params['q0'], 'target_steps': params['target_steps']}
            best_curve = haploid(hap_params)['q']
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = round(best_s, 3)
            hapMappedCurves.append(best_curve)
    return {'map': seffMap, 'ngC': hapMappedCurves}

def hap_gradient_mapping(params, gdFile):
    seffMap = dict()
    hapMappedCurves = []
    f_out = open(f"mapping_result_txt/h{params['h']}_hap_gradient{gdFile}_G_fix_map.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\ts in haploid ngd population\n")

    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    # print('params', gd_configs)
    stabilityRes = loadStability(params['h'])
    grid_results = load_pickle("fixation_mapping_result_haploid", f"h{params['h']}_hap_grid{gdFile}_G_fix.pickle")

    for (s, c, h) in gd_configs:
        # only testing for s = 0.4, c=0.6
        # if math.isclose(s, 0.4) and math.isclose(c, 0.6):
        # if ((s, c, h), 1.0) in stabilityRes['fixation']: 
        if gd_res[(s,c,h)]['state'] == 'fix': ### only for h = 0.5
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s = grid_results['map'][(s, c, h)]

            # gradient descent or newton_raphson
            best_vars = hap_gradient(best_s, s, c, h, params, gdFile)
            best_s = best_vars[0]

            ### haploid
            hap_params = {'s': best_s, 'q0': params['q0'], 'target_steps': params['target_steps']}
            best_curve = haploid(hap_params)['q']
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = round(best_s, 3)
            hapMappedCurves.append(best_curve)
    return {'map': seffMap, 'ngC': hapMappedCurves}

def get_delta_lambda(gd_config, ngd_config, eq):
    params = {'config':gd_config, 'currq':eq, 'conversion': "gametic"}
    gd_lambda = derivative(params)
    ngd_lambda = get_ngd_stability(ngd_config[0], ngd_config[1], eq)
    return (gd_lambda-ngd_lambda, ngd_lambda)

def same_eq(s_ngd, h_ngd, eq):
    """
    Check if the NGD eq is the same as the GD eq.
    """
    ngd_eq = get_eq_ngd(s_ngd, h_ngd)
    return math.isclose(ngd_eq, eq)
    
def find_candidate(
        s_ngd:float, 
        h_ngd:float, 
        q_ngd:float,
        state:str,  
        gd_curve:list, 
        eq:float) -> tuple:
    """
    Compute MSE and traj for a given (s_ngd, h_ngd) if it matches the target state and eq.
    Returns mse or np.inf if not matching.
    """
    # check if NGD has same state and eq (within tolerance)
    if not check_state_eq(s_ngd, h_ngd, eq, state):
        # print("NOT SAME STATE", s_ngd, h_ngd, eq)
        return None, np.inf
    if not same_eq(s_ngd, h_ngd, eq):
        # print("NOT SAME EQ", s_ngd, h_ngd, eq)
        return None, np.inf

    # compute trajectory and metrics

    traj = wm(s_ngd, h_ngd, len(gd_curve), q_ngd)['q']
    if len(traj) == 1:
        # print("TRAJ LENGTH 1", s_ngd, h_ngd, q_ngd)
        # print("stop at 0 step:", len(gd_curve))
        # print(gd_curve)
        return None, np.inf
    mse = euclidean(traj, gd_curve)  # MSE
    # delta_lambda, ngd_lambda = get_delta_lambda(gd_config, (s_ngd, h_ngd), eq)
    return (traj, mse)

def evaluate_sngd(s_ngd, h_ngd, params, q_init_list, eq, state, gd_config):
    s_ngd = round(s_ngd, 3)
    s, c, h = gd_config
    all_q_MSE = 0
    ### boundary for s_ngd and h_ngd
    if s_ngd * h_ngd > 1:
        # print("s_ngd * h_ngd > 1", s_ngd, h_ngd)
        return s_ngd, np.inf, None
    new_params = params.copy()
    new_params['s'] = round(s, 3)
    new_params['c'] = round(c, 3)
    new_params['h'] = round(h, 3)
    for q_ngd in q_init_list:
        curr_ngd_curve = None
        if q_ngd == eq:
            print("q_ngd == eq", q_ngd)
            continue
        # print(new_params)
        new_params['q0'] = q_ngd
        gd_curve = run_model(new_params)['q']
        ### DEBUGGING
        # if q_ngd == 0.2:
        #     print("Q=0,2", gd_curve)
        # out1.write(f"q_init = {q_ngd:.4f}=====\n")
        ### check candidate and get MSE
        res = find_candidate(s_ngd, h_ngd, q_ngd, state, gd_curve, eq)
        if res[1] != np.inf:
            curr_ngd_curve, curr_MSE = res[0], res[1]
            all_q_MSE += curr_MSE
        # if math.isclose(s, 0.3) and math.isclose(c, 0.2):
        #     print("SINGLE S RESULT", s_ngd, curr_ngd_config, curr_MSE)
        # print(curr_ngd_config, curr_MSE)
    return s_ngd, all_q_MSE, curr_ngd_curve
        
#### grid search only ################
def grid_mapping(params, gdFile):
    seffMap = dict()
    ngd_mapped = []
    ts = 0.4
    tc = 0.4
    state = "Unstable"
    ### !!! CHANGE THE FILE NAME IF RUNNING FOR ALL !!!!!!
    # f_out: the text file tto store (s,c,h) -> (s_ngd, h_ngd) mapping
    f_out = open(f"unstable_mapping_v2/h{params['h']}_grid_G{gdFile}_unstable.txt", 'w')
    # f_out = open(f"unstable_mapping_v2/h{params['h']}_s{ts}_c{tc}_grid_G_unstable.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h) in NGD population (S_Effective)\n")

    ### LOADING NECESSARY RESULTS FOR MAPPING 
    stabilityRes = loadStability(params['h'])
    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    # print("STABILITY UNSTABLE REGIME:", stabilityRes[state])

    for (s, c, h) in gd_configs:

        # compute eq for stable regime:
        params_eq = {'s': s, 'c': c, 'h': h, 'conversion': 'gametic'}
        eq = get_eq(params_eq)['q3']

        if ((s, c, h), eq) in stabilityRes[state]:
        # if ((s, c, h), eq) in stabilityRes[state] and math.isclose(s, ts) and math.isclose(c, tc):
        # if gd_res[(s,c,h)]['state'] == 'fix':
            print(f"===================={state} Mapping===============")
            print(s, c, h)
            gd_curve = gd_res[(s, c, h)]['q']
            all_best_diff = dict() 
            
            # outtxt: the text file to store the MSE for each s_ngd
            output_folder = f"sngd_candidates"
            outtxtName = os.path.join(output_folder, f"gd{s}_{c}_{h}_unstable_best.txt")
            if state == "Stable" or state == "Unstable":
                # for each gd config, open a new file to store the MSE for all s_ngd
                out1 = open(outtxtName, "w")
                out1.write(f"GD config: s={s}, c={c}, h={h}, eq={eq}\n")
                out1.write("s_ngd, h_ngd, MSE (summed over q)\n")

                if math.isclose(eq, 0.5):
                    continue
               
                #### compute h_ngd first
                h_ngd = eq/(2*eq-1)
                # print('----h_ngd----', h_ngd)
                # print("STABLE MAPPING WITH H_ngd =", h_ngd)
                if state == 'Unstable':
                    q_init_list = np.arange(0, 1.0, 0.005)
                    # q_init_list = [0.2]
                else:
                    q_init_list = [0.001]
               
                best = {'mse': np.inf, 's_ngd': None, 'traj': None}
                best_ngd_config = None
                best_diff = np.inf
                ### range of s_ngd to test: -10.0 to 1.0
                s_values = np.arange(-10.0, 1.0, 0.01)
                gd_config = (s, c, h)

                with concurrent.futures.ProcessPoolExecutor() as exe:
                # exe.map can take multiple iterables in parallel
                    for s_ngd, all_q_MSE, traj in exe.map(
                        evaluate_sngd, 
                        s_values, 
                        repeat(h_ngd),
                        repeat(params),
                        repeat(q_init_list),
                        repeat(eq),
                        repeat(state),
                        repeat(gd_config)
                    ):
                        if traj is None:
                            continue
                        out1.write(f"s_ngd = {s_ngd:.4f}---------------------\n")
                        if 0 < all_q_MSE < best['mse']:
                            best['mse'], best['s_ngd'], best['traj'] = all_q_MSE, s_ngd, traj
                        
                        out1.write(f"s_ngd={s_ngd:.3f}, h_ngd={h_ngd:.3f}, curr_mse={all_q_MSE:.6f}\n")

                # all_q_MSE, curr_ngd_curve = evaluate_sngd(s_ngd, h_ngd, eq, state, gd_curve, new_params, q_init_list, out1)

                # if all_q_MSE < best['mse'] and all_q_MSE != 0:
                #     best.update({'mse': all_q_MSE, 's_ngd': s_ngd, 'traj': curr_ngd_curve})

                # print(q_ngd, best_diff)

            best_ngd_config, best_ngd_curve, best_diff = best['s_ngd'], best['traj'], best['mse']
            if best_ngd_config == None: 
                print("!!!!!!!!!!!!!NO STABLE SNGD")
                continue

            # summary line
            print("####Writing best NGD config to f_out####")
            f_out.write(
                f"s={s:.3f}, c={c:.3f}, h={h:.3f}\t"
                f"s_ngd={best['s_ngd']:.3f}, h_ngd={h_ngd:.3f}, MSE={best['mse']:.6f}\n"
            )

            seffMap[(s, c, h)] = {'config':best_ngd_config, 'traj': best_ngd_curve}

    #         print(seffMap[(s, c, h)])
            
    #         for qkey in best_ngd_config.keys():
    #             # f_out.write(f"q_init = {qkey:.4f}")
    #             best_config = best_ngd_config[qkey]
    #             best_mse = all_best_diff[qkey]

    #             f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\tq_init={qkey:.4f}, s_NGD={'%.3f' % float(best_config[0])}, h_NGD={'%.3f' % float(best_config[1])}\terror={'%.6f' % float(best_mse)}\n")

    output_folder = "unstable_mapping_results"
    pickle_filename = f"{params['h']}_{state}_grid_{gdFile}.pickle"
    # Save the results dictionary into a pickle file.
    save_pickle(output_folder, pickle_filename, seffMap)
            # print(s_mse_map)
    
    # print(seffMap)
    
    return seffMap
        

#### grid search for fixation regimes only ################
def grid_mapping_fix(params, gdFile):
    seffMap = dict()
    ngd_mapped = []
    f_out = open(f"h{params['h']}_grid_G{gdFile}_loss.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h) in NGD population (S_Effective)\n")

    stabilityRes = loadStability(params['h'])
    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    ngd_results = load_pickle("", "allngdres001G.pickle")

    for (s, c, h) in gd_configs:
        # if ((s, c, h), 1.0) in stabilityRes['fixation']:
        if gd_res[(s,c,h)]['state'] == 'loss':
            print(s, c, h)
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_ngd_config = None
            best_ngd_curve = None
            for ngd_key, ngd_curve in ngd_results.items():
                diff = euclidean(ngd_curve, gd_curve)
                if diff < best_diff:
                    best_diff = diff
                    best_ngd_config = ngd_key
                    best_ngd_curve = ngd_curve

            seffMap[(s, c, h)] = best_ngd_config
            ngd_mapped.append(best_ngd_curve)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_ngd_config[0]}, h={'%.3f' % best_ngd_config[1]}\terror={'%.3f' % best_diff}\n")
    
    # print(seffMap)
    
    return {'map': seffMap, 'ngC': ngd_mapped}
        


#### grid + gradient ########################
def gradient_mapping(params, gdFile):
    seffMap = dict()
    wms = []
    f_out = open(f"h{params['h']}_grad_G{gdFile}_fix.txt", 'w')
    f_out.write(f"Gene Drive Configuration\t\t(s, h)in NGD population\n")

    gd_results = loadGres(params['h'], gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = loadStability(params['h'])
    grid_results = load_pickle("fixation_mapping_result", f"h{params['h']}_grid{gdFile}_G_fix.pickle")

    for (s, c, h) in gd_configs:
        if math.isclose(s, 0.2) and math.isclose(c, 0.9):
        # if ((s, c, h), 1.0) in stabilityRes['fixation']:
        # if gd_res[(s,c,h)]['state'] == 'fix':
            gd_curve = gd_res[(s, c, h)]['q']
            best_diff = 10000
            best_s, best_h = grid_results['map'][(s,c,h)][0], grid_results['map'][(s,c,h)][1]
            # print('before gradient', best_s)
            best_s = -0.1
            best_h = h

            best_vars = gradient(best_s, best_h, s, c, h, params, gdFile)
            best_s, best_h = best_vars[0], best_vars[1]
            # best_s = optimization_s(best_s, s, c, h, params)
            # best_h = optimization(best_s, s, c, h, params)

            best_curve = wm(best_s, best_h, params['target_steps'], params['q0'])['q']
            best_diff = euclidean(best_curve, gd_curve)
        
            # print((s, c, h), 'best:', best_s)
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\ts={'%.3f' % best_s}, h={'%.3f' % best_h}\terror={'%.3f' % best_diff}\n")
            seffMap[(round(float(s), 3), round(float(c), 3), round(float(h), 3))] = (round(float(best_s), 3), round(float(best_h), 3))
            wms.append(best_curve)
    return {'map': seffMap, 'ngC': wms}

def check_mapping(params, gdFile):
    '''
    Check if the mapping from GD to NGD is correct by checking if the equilibrium
    frequency of the NGD population is the same as the equilibrium frequency of the
    GD population.
    '''
    sanity_failures = []
    h_val = params['h']
    stabilityRes = loadStability(h_val)
    gd_results = loadGres(h_val, gdFile)
    gd_configs, gd_res = gd_results[0], gd_results[1]

    dfunc = compute_lambda_gd()

    for (s, c, h) in gd_configs:
        if not math.isclose(c, 0.0):  # Only consider c = 0 configs
            continue

        # Extract equilibrium
        params_eq = {'s': s, 'c': c, 'h': h, 'conversion': 'gametic'}
        eq = get_eq(params_eq)['q3']

        if eq == 'NA' or not (0 < eq < 1):
            continue

        if ((s, c, h), eq) not in stabilityRes['Unstable']:
            continue  # Only focus on unstable region

        # The expected h_ngd based on q_eq should match h
        mapped_h = eq / (2 * eq - 1)
        # Compare to the original GD h
        if not math.isclose(mapped_h, h, rel_tol=1e-2):
            sanity_failures.append(((s, c, h), mapped_h))

        print(f"GD (s={s:.3f}, h={h:.3f}, c=0.000) → q_eq={eq:.4f} → h_ngd={mapped_h:.4f} (original h={h:.4f})")

    if not sanity_failures:
        print("Sanity check passed: all h_ngd ≈ h for c = 0")
    else:
        print("Sanity check failed for the following configs:")
        for config, mapped_h in sanity_failures:
            s, c, h = config
            print(f"GD (s={s:.3f}, h={h:.3f}) → mapped h_ngd = {mapped_h:.4f}")

    return sanity_failures

# params = params = {'n': 500, 'h': 0.3, 'target_steps': 40000, 'q0': 0.001}
# check_mapping(params)

