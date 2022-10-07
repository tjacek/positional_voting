import cv,data,clfs

def train_ensemble(train,clf_alg):
    bayes_cf=cv.BayesOptim(clf_alg)
    train_tuple=train.as_dataset()[:2]
    best_estm,best_params= bayes_cf(*train_tuple)
    print(best_params)

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
        names= in_k.names()
        print(names.cats_stats())
        train_ensemble(in_k,clf_alg)
        raise Exception('Ok')