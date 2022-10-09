import cv,data,clfs,OPV,learn

def find_opv(in_i,clf_alg):
    selector=cv.SplitSelector(0,3)
    train_i,valid_i=in_i.split(selector)
    hyper_i=find_hyperparams(train_i,clf_alg)
    ens_i=clf_alg.fit(train_i,hyper_i)
    votes=predict_votes(ens_i,valid_i)
    pref=OPV.make_pref(votes)
    weights=OPV.optim(pref,maxiter=100)
    print(weights)
    return weights

#    print(len(in_i))
#    print(len(valid_i))
#    print(len(votes.results[0]))

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
        in_k,out_k=data_i.split(selector_k)
        yield in_k,out_k

if __name__ == "__main__":
    clf_alg=clfs.rf_clf()
    for in_k,out_k in gen_splits("cleveland"):
#        print(len(in_k))
#        print(len(out_k))
#        names= in_k.names()
#        print(names.cats_stats())
        find_opv(in_k,clf_alg)  
        raise Exception('Ok')