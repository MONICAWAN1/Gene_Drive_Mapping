import argparse, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import grid_mapping, grid_mapping_fix, gradient_mapping, hap_grid_mapping, hap_gradient_mapping, getdiff, check_mapping
from utils import save_pickle

def main():
    parser = argparse.ArgumentParser(description="Run specific mapping functions from mapping.py")
    parser.add_argument("map_function", type=str, help="Name of the mapping function to run")
    parser.add_argument("h", type=str, help="Value of H in GD")
    parser.add_argument("gdFile", type=str, help="Name of GDres file")
    parser.add_argument("-s", action="store_true", help="Save results and run getdiff")

    args = parser.parse_args()
    
    # Dictionary to map function names to actual functions
    map_functions = {
        "grid": grid_mapping,
        "gradient": gradient_mapping,
        "hap_grid": hap_grid_mapping,
        "hap_gradient": hap_gradient_mapping,
        "grid_fix": grid_mapping_fix,
        "check_mapping": check_mapping
    }

    print(f"Running {args.map_function}...")
    if '0001' in args.gdFile:
        label='0001'
    elif '001' in args.gdFile:
        label='001'
    else:
        label=''

    params = {'n': 500, 'h': args.h, 'target_steps': 40000, 'q0': 0.001}

    if args.map_function in map_functions:
        if args.map_function == "check_mapping":
            map_functions[args.map_function](params, label)
        else:
            mapping_result = map_functions[args.map_function](params, label)
    else:
        print('NOT IN MAP_FUNCTIONS', args.map_function)
        mapping_result = map_functions[args.map_function]()
    print(f'Running {args.map_function} mapping...')

    # DEBUGGING GRADIENT DESCENT
    ts = 0.2
    tc = 0.9 
    pickle_saved_to = "fixation_mapping_result_haploid" if 'hap' in args.map_function else "fixation_mapping_result"
    pickle_filename = f"h{args.h}_{args.map_function}_fixation{label}_G.pickle"
    print("Saving results and computing differences...")
    save_pickle(pickle_saved_to, pickle_filename, mapping_result)
    
    if args.s:
        getdiff(args.h, args.map_function, label)

if __name__ == "__main__":
    main()