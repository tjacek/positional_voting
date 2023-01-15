import numpy as np
from sklearn import ensemble
import protocols

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

def stats(acc):
    return [f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std]]

basic_exp('wine_cv2')