import numpy as np
from sklearn import ensemble
import protocols,utils
import inliner

def multi_exp(in_path,exp=None):
    if(exp is None):
        exp=basic_exp
    dec_fun=utils.dir_fun(as_dict=True)(exp)
    dict_output=dec_fun(in_path)
    all_lines=[]
    for name_i,lines_i in dict_output.items():
        for line_j in lines_i:
            all_lines.append(f'{name_i},{line_j}')
    print('\n'.join(all_lines))

def basic_exp(in_path):
    lines=['ECSCF']+stats(protocols.escf_exp(in_path))
    lines=','.join(lines)
    lines=[lines]
    algs={'RF':ensemble.RandomForestClassifier(),
          'Bag':ensemble.BaggingClassifier(),
          'Grad':ensemble.GradientBoostingClassifier()}
    for type_i,clf_i in algs.items():
        print(type_i)
        acc_i=protocols.check_alg(in_path,clf_i)
        stats_i=','.join(stats(acc_i))
        lines.append(f'{type_i},{stats_i}')	
    print('\n'.join(lines))
    return lines

def stats(acc):
    return [f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std]]

def inliner_exp(in_path):
    inliner_voting=inliner.InlinerVoting()
    acc_i=inliner_voting(in_path)
    stats_i=','.join(stats(acc_i))
    line_i=f'INLINER NECSCF,{stats_i}'
    print(line_i)
    return [line_i]

#multi_exp('uci')
multi_exp('uci',inliner_exp)
#basic_exp('wine_cv2')
