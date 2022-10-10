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
    return weights,ens_i

def evaluate_opv(weights,ens_i,out_i):
    votes=predict_votes(ens_i,out_i)
    pref=OPV.make_pref(votes)
    return pref.positional_voting(weights)

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

if __name__ == "__main__":
    clf_alg=clfs.rf_clf()
    partial=[]
    for k,(in_k,out_k) in enumerate(gen_splits("cleveland")):
        weights,ens_i=find_opv(in_k,clf_alg)  
        out_votes_i=evaluate_opv(weights,ens_i,out_k)
        partial.append(out_votes_i)
        print(f'Ok{len(out_votes_i)}')
        print((len(in_k),len(out_k)))
        if(k>2):
            break
#        raise Exception('Ok')
    result=learn.unify_results(partial)
    print(len(result))
    print(result.get_acc())