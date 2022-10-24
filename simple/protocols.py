import cv,data,clfs,opv,learn

def find_opv(in_i,clf_alg,metric=None):
    selector=cv.SplitSelector(0,3)
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

def exp(in_path,clf_alg,metric=None):
    partial_base,partial_opv=[],[]
    for k,(in_k,out_k) in enumerate(gen_splits(in_path)):
        weights,ens_i=find_opv(in_k,clf_alg,metric)  
        base_results,opv_result =evaluate_opv(weights,ens_i,out_k)
        partial_base.append(base_results)
        partial_opv.append(opv_result)
        if(k>2):
            break
    result_base=learn.unify_results(partial_base)
    result_opv=learn.unify_results(partial_opv)
    return result_base,result_opv

def show_result(result_base,result_opv=None):
    if(result_opv is None):
        result_base,result_opv=result_base
    base_acc,opv_acc=result_base.get_acc(),result_opv.get_acc()
    print("base:{base_acc} opv:{opv_acc} diff:{opv_acc-base_acc}")

if __name__ == "__main__":
    clf_alg=clfs.rf_clf()
    metric_exp( "cleveland",clf_alg)
#    result=exp("cleveland",clf_alg,metric=None)
#    show_result(result)
#    print(result_base.get_acc())
#    print(result_opv.get_acc())