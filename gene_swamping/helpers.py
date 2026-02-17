import os
import pickle
import numpy as np
PICKLE_DIR = "../pickle"

def load_pickle(filename):
    filePath = os.path.join(PICKLE_DIR, filename)
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Error: File '{filePath}' not found.")
    try:
        with open(filePath, 'rb') as f:
            return pickle.load(f)

    except Exception as e:
        raise Exception(f"Error loading pickle file '{filePath}': {e}")
    
def save_pickle(filename, res):
    filePath = os.path.join(PICKLE_DIR, filename)
    with open(filePath, 'wb') as f:
            pickle.dump(res, f)

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