import numpy as np
import math
import pickle
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sympy as sp


'''
# Loop through all hap/diploid mapping result, 
# get a mapping dictionary with ngd s values as keys and the ngd config as value
'''
directory_path = '../pickle/mapping_result'

def is_hap_result(filename):
    return filename.endswith('.pickle') and 'hap' in filename and '001' in filename
        
def combine_reverse():
    reversed_dict = dict()
    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path) and is_hap_result(filename):
            print(f"Found haploid result: {full_path}")
            with open(full_path, 'rb') as pf:
                    mapping = pickle.load(pf)['map']
                    for config, se_val in mapping.items():
                        # Optional: round se_val to avoid float precision issues
                        se_val_rounded = round(se_val, 6)
                        if se_val_rounded in reversed_dict:
                            print(f"Warning: Duplicate se_val {se_val_rounded} found in {filename}")
                            reversed_dict[se_val_rounded].append(config)
                        else: 
                            reversed_dict[se_val_rounded] = [config]
    with open('haploid_gd_all.pickle', "wb") as fout:
         pickle.dump(reversed_dict, fout)
    return reversed_dict

if __name__ == "__main__":
    print(sorted(list(combine_reverse().keys())))