import sys, os, math
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import wm
from utils import load_pickle, save_pickle, euclidean, GD_RES_SUBDIR, gd_res_filename
from .regime import get_stability_table

def get_eq(params):
    s, c, h = params['config']
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

def getdiff(currH, mapfunction, gdFile):
    '''
    given gd results and one mapping result
    saves a dictionary (s,c,h): error of map
    store results as .txt in mapping_diff_txt folder and as .pickle in mapping_diff folder
    '''
    # test s, c values
    ts = 0.2
    tc = 0.9
    state = "fixation"
    haploid = True if 'hap' in mapfunction else False

    fileName = f"mapping_diff_txt/h{currH}_mappingdiff_{mapfunction}_{state}{gdFile}_G.txt"
    f_out = open(fileName, 'w')
    print(f"writing to {fileName}")
    f_out.write(f"Gene Drive Configuration\t\t{mapfunction} result\t\tMSE error compared to Gene Drive\n")

    # load gd curves and stability results
    gd_results = load_pickle(GD_RES_SUBDIR, gd_res_filename(currH, gdFile))
    gd_configs, gd_res = gd_results[0], gd_results[1]
    stabilityRes = get_stability_table(currH)

    savedPickle_subdir = "mapping_diff"
    savedPickle_name = f"h{currH}_mappingdiff_{mapfunction}_{state}{gdFile}_G.pickle"

    # with open('pickle/allngdres.pickle', 'rb') as f1:
    #     wm_results = pickle.load(f1)
    pickle_saved_to = "fixation_mapping_result_haploid" if haploid else "fixation_mapping_result"
    gridResFile_name = f"h{currH}_{mapfunction}_{state}{gdFile}_G.pickle"
    gridResult = load_pickle(pickle_saved_to, gridResFile_name)

    if haploid: 
        sMap_grid, _ = gridResult['map'], gridResult['ngC']
    else: 
        sMap_grid = gridResult
    # sMap_grad, wms_grad = gradResult['map'], gradResult['ngC'] ## comment out for switching mapping method
    print('len of grid map', len(sMap_grid))

    diffmap = dict()
    ngd_config = []

    for (s, c, h) in gd_configs:
        # if math.isclose(s, 0.2) and math.isclose(c, 0.9):
        # if ((s, c, h) in sMap_grid and ((s, c, h), 1.0) in stabilityRes['Fixation']) or (math.isclose(h, 0.5) and gd_res[(s, c, h)]['state']=='fix'):
        params_eq = {'config': (s, c, h), 'conversion': 'gametic'}
        # eq = get_eq(params_eq)['q3']

        if ((s, c, h) in sMap_grid and ((s, c, h), 1.0) in stabilityRes['fixation']):
            gd_curve = gd_res[(s, c, h)]['q']
            if 'hap' in mapfunction:
                ngd_s = sMap_grid[(s, c, h)]
                params = {'s': ngd_s, 'target_steps': 40000, 'q0': 0.001} # set parameters to the mapped s value for ngd model
                ngd_curve = haploid(params)['q']
            else:
                ngd_s, ngd_h = sMap_grid[(s, c, h)][0], sMap_grid[(s, c, h)][1]
                ngd_curve = wm({'s': ngd_s, 'h': ngd_h, 'target_steps': 40000, 'q0': 0.001})['q']
            
            # paramSe = {'s':s, 'c':c, 'n': 500, 'h':h, 'target_steps': 40000, 'q0': 0.001}
            # ngd_curve = haploid_se(paramSe)['q']

            diff = euclidean(ngd_curve, gd_curve)
            diffmap[(s, c, h)] = diff
            f_out.write(f"s={'%.3f' % s}, c={'%.3f' % c}, h={'%.3f' % h}\t\tmapped to {sMap_grid[(s, c, h)]}\t\tMSE error={'%.6f' % diff}\n")
            # ngd_config.append(ngd_key)
    save_pickle(savedPickle_subdir, savedPickle_name, diffmap)

def main():
    getdiff(0.5, "grid", "001")

if __name__ == '__main__':
    main()