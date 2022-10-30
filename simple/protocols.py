import numpy as np
import json
import cv,data,clfs,opv,learn,utils

class ExpOutput(object):
    def __init__(self,base_results,opv_results):
        self.base_results=base_results
        self.opv_results=opv_results

    def __len__(self):
        return len(self.base_results)
    
    def diff(self):
        return [ (base_i.get_acc() - opv_i.get_acc()) 
           for base_i,opv_i in zip(self.base_results,self.opv_results)]

    def mean(self):
        base_acc=[base_i.get_acc() 
            for base_i in self.base_results]
        opv_acc=[opv_i.get_acc() 
            for opv_i in self.opv_results]
        return np.mean(base_acc),np.mean(opv_acc)

    def save(self,out_path):
        utils.make_dir(out_path)
        gen=zip(self.base_results,self.opv_results)
        for i,(base_i,opv_i) in enumerate(gen):
            base_i.save(f'{out_path}/base_{i}')
            opv_i.save(f'{out_path}/opv_{i}')

def read_output(in_path):
    paths=utils.get_paths(in_path)
    base,opv=[],[]
    for path_i in paths:
        result_i=learn.read_result(path_i)
        if('base' in path_i):
            base.append(result_i)
        if('opv' in path_i):
            opv.append(result_i)
    return ExpOutput(base,opv)

def find_opv(in_i,clf_alg,metric=None,selector=None):
#    selector=cv.SplitSelector(0,3)
    train_i,valid_i=in_i.split(selector)
    hyper_i=find_hyperparams(train_i,clf_alg)
    ens_i=clf_alg.fit(train_i,hyper_i)
    votes=predict_votes(ens_i,valid_i)
    pref=opv.make_pref(votes)
    weights=opv.OPV(metric)(pref)
    print(weights)
    return weights,ens_i

def evaluate_opv(weights,ens_i,out_i):
    votes=predict_votes(ens_i,out_i)
    pref=opv.make_pref(votes)
    opv_result= pref.positional_voting(weights)
    base_results = votes.voting()
    return base_results,opv_result 

def predict_votes(ens_i,data_i):
    data_tuple= data_i.as_dataset()
    votes=learn.Votes()
    for j,estm_j in enumerate(ens_i.estimators_):
        y_pred_j=estm_j.predict_proba(data_tuple[0])
        result_i=learn.make_result(y_pred_j,data_tuple[-1])
        votes.results.append(result_i)
    return votes

def find_hyperparams(train,clf_alg):
    bayes_cf=cv.BayesOptim(clf_alg,n_split=2)
    train_tuple=train.as_dataset()[:2]
    best_estm,best_params= bayes_cf(*train_tuple)
    return best_params

def gen_splits(data_i,n_splits=10):
    if(type(data_i)==str):
        data_i=data.read_data(data_i)
    data_i.norm()
    for k in range(n_splits):
        selector_k=cv.SplitSelector(k,n_splits)
        out_k,in_k=data_i.split(selector_k)
        yield in_k,out_k

def metric_exp(in_path,clf_alg):
    all_metrics=[opv.auc_metric,opv.acc_metric,opv.f1_metric]
    pair=[exp(in_path,clf_alg,metric_i) 
            for metric_i in all_metrics]
    for pair_i in pair:
        show_result(*pair_i)

def multi_exp(in_path,clf_alg,metric=None,n_iters=2,opv_exp=None):
    if(opv_exp is None):
        opv_exp=OPVExp()
    all_base,all_opv=[],[]
    for i in range(n_iters):
        base_i,opv_i=opv_exp(in_path,clf_alg,metric)
        all_base.append(base_i)
        all_opv.append(opv_i)
    return ExpOutput(all_base,all_opv)

class OPVExp(object):
    def __init__(self,n_splits=10,limit=1,selector=None):
        if(selector is None):
            selector=cv.SplitSelector(0,3)
        self.n_splits=n_splits
        self.selector=selector
        self.limit=limit

    def __call__(self,in_path,clf_alg,metric=None):
        partial_base,partial_opv=[],[]
        split_gen=gen_splits(in_path)
        for k,(in_k,out_k) in enumerate(split_gen,self.n_splits):
            weights,ens_i=find_opv(in_k,clf_alg,metric,self.selector)  
            base_results,opv_result =evaluate_opv(weights,ens_i,out_k)
            partial_base.append(base_results)
            partial_opv.append(opv_result)
            if(k>self.limit):
                break
        result_base=learn.unify_results(partial_base)
        result_opv=learn.unify_results(partial_opv)
        return result_base,result_opv

def show_result(result_base,result_opv=None):
    if(result_opv is None):
        result_base,result_opv=result_base
    base_acc,opv_acc=result_base.get_acc(),result_opv.get_acc()
    print(f"base:{base_acc} opv:{opv_acc} diff:{opv_acc-base_acc}")

if __name__ == "__main__":
    clf_alg=clfs.rf_clf()
#    output=multi_exp("cleveland",clf_alg,metric=None,n_iters=2)
#    output.save('mult_test')
    output=read_output('mult_test')
    print( output.mean())