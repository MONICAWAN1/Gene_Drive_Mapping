import argparse, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import plot_mapping, plot_gd, plotMapDiff, ploterror, partition, plot_errorh, getHapseMapDiff, plot_qmaps, gd_to_ngd_diff, plot_fixation_surface, plot_fixation_res, test_mapping_trajectory, plot_sngd, plot_sngd_all, plot_diff, plot_h_ngd_vs_h, plot_sh_heatmap, plot_line_se_he

'''
ploterror: (run getDiff.py first) Error Heatmap for the Grid/gradient Mapping from GD to NGD Haploid Model at some h
plotMapDiff: (run getHapseMapDiff first) Error heatmap for haploid_se vs GD at some h
partition: partition plot for stable/unstable/fixation/loss regimes
plot_mapping: plot the curves for a specific mapping for certain configurations
'''

def main():
    parser = argparse.ArgumentParser(description="Run specific plotting functions from plotting.py")
    parser.add_argument("plot_function", type=str, help="Name of the plotting function to run")
    parser.add_argument("h_val", type=str, help="Value of H in GD")

    args = parser.parse_args()
    
    # Dictionary to map function names to actual functions
    plot_functions = {
        "plotmapping": plot_mapping,
        "plotgd": plot_gd,
        "partition": partition,
        "ploterror": ploterror,
        "plotMapDiff": plotMapDiff,
        "plot_errorh": plot_errorh,
        "getHapseMapDiff": getHapseMapDiff,
        "gd_to_ngd_diff": gd_to_ngd_diff,
        "plot_qmaps": plot_qmaps,
        "plot_fixation_res": plot_fixation_res,
        "plot_fixation_surface": plot_fixation_surface,
        "test_mapping_trajectory": test_mapping_trajectory,
        "plot_sngd": plot_sngd,
        "plot_sngd_all": plot_sngd_all,
        "plot_diff": plot_diff,
        "plot_hh": plot_h_ngd_vs_h,
        "plot_sh_heatmap": plot_sh_heatmap,
        "plot_line_se_he": plot_line_se_he
    }

    if args.plot_function in plot_functions:
        print(f"Running {args.plot_function}...")
        if args.plot_function == "test_mapping_trajectory":
            s_gd, c_gd, h_gd = 0.03, 0.73, float(args.h_val)
            plot_functions[args.plot_function](s_gd, c_gd, h_gd, param_file=f'regression/h{args.h_val}_mapping_coeffs_sq.pickle',file_type='pickle')
        else: 
            plot_functions[args.plot_function](args.h_val)
    else:
        print(f"Error: {args.plot_function} is not a valid plotting function.")

if __name__ == "__main__":
    main()