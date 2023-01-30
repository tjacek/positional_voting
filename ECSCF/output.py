import numpy as np
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
import protocols,utils,ecscf,ens
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

class BasicExp(object):
    def __init__(self,escf_algs=['LR','RF'],algs=['LR','RF']):
        self.escf_algs=get_escf_algs(escf_algs)
        self.algs=get_algs(algs)

    def __call__(self,in_path):
        lines=[]
        for type_i,ens_i in self.escf_algs.items():
            print(type_i)
            line_i=[f'ESCF,{type_i}']+stats(ens_i(in_path))
            lines.append(','.join(line_i))
        for type_i,clf_i in self.algs.items():
            print(type_i)
            acc_i=protocols.check_alg(in_path,clf_i)
            stats_i=','.join(stats(acc_i))
            lines.append(f'{type_i},{stats_i}')	
        print('\n'.join(lines))
        return lines

def stats(acc):
    return [f'{fun_i(acc):.4f}' 
        for fun_i in [np.mean,np.std]]

def get_algs(names):
    alg={}
    for name_i in names:
        alg[name_i]=get_clf(name_i)
    return alg

def get_clf(name_i):
        if(name_i=="LR"):
            return LogisticRegression(solver='liblinear')
        if(name_i=="RF"):
            return ensemble.RandomForestClassifier()
        if(name_i=='Bag'):
            return ensemble.BaggingClassifier()
        if(name_i=='Grad'):
            return ensemble.GradientBoostingClassifier()

def get_escf_algs(names):
    alg={}
    for name_i in names:
        raw_i=name_i.split('_')
        if(raw_i[0]=='binary'):
            clf_i=get_clf(raw_i[1])
            ens_factory=ens.RawBinary
        else:
            clf_i=get_clf(raw_i[0])
            ens_factory=ens.EnsembleFactory
#        if(name_i=="LR"):
#            ens_factory=ens.EnsembleFactory()
#        if(name_i=='RF'):
#            ens_factory=ens.EnsembleFactory(ensemble.RandomForestClassifier())  
        alg[name_i]=protocols.ESCFExp(ens_factory(clf_i))
    return alg

def inliner_exp(in_path):
    inliner_voting=inliner.InlinerVoting()
    acc_i=inliner_voting(in_path)
    stats_i=','.join(stats(acc_i))
    line_i=f'INLINER NECSCF,{stats_i}'
    print(line_i)
    return [line_i]


basic_exp=BasicExp(['LR','RF','Bag','Grad',
                   'binary_LR','binary_RF','binary_Bag','binary_Grad'],
                    ['LR','RF','Bag','Grad'])
#lines=basic_exp('imb/wine-quality-red')
multi_exp('test',basic_exp)
#print(lines)