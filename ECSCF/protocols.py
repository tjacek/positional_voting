from sklearn import ensemble
import os.path
import data,exp,splits,ecscf,cv,learn,utils

class Protocol(object):
    def __init__(self,fun=None,search_space=None):
        if(fun is None):
            fun=one_out_iter
        if(search_space is None):
            search_space={'n_hidden':[25,50,100,200],
                          'n_epochs':[100,250,500]}
        self.search_space=search_space 
        self.fun=fun

    @utils.lazy_dir_fun
    def __call__(self,in_path,out_path,
                    n_split=2,n_iters=10):     
        hyperparams=cv.find_hyperparams(in_path,
            self.search_space,
            n_split=n_split)
        iters_fun=utils.iter_fun(n_iters)(self.fun)
        iters_fun(in_path,out_path,hyperparams,n_split)
        print(hyperparams)

def unify_cv(fun):
    def helper(*args, **kwargs):
        results=[]
        in_path=args[1]
#        raise Exception(in_path)
        for path_i in data.top_files(in_path):
#            print(path_i)
            args=list(args)
            args[1]=path_i
            result_i=fun(*args,**kwargs)
            results.append(result_i)
        full_results=learn.unify_results(results)
        return full_results.get_acc()  
    return helper

def one_out_iter(in_path,out_path,
    hyperparams,n_split=10):
    data.make_dir(out_path)
    fold_path=f'{out_path}/fold'
    feat_path=f'{out_path}/feats'
    cv_folds=cv.prepare_folds(in_path,fold_path,n_split)
    data.make_dir(feat_path)
    for i,data_i in enumerate(cv_folds):
        out_i=f'{feat_path}/{i}'
        data.make_dir(out_i)
        clf_i=ecscf.ECSCF(**hyperparams)
        data_i.save(f'{out_i}/common')
        datasets=clf_i.fit_dataset(data_i,features=True)
        datasets.save(f'{out_i}/binary')  

@utils.dir_fun(False)
def escf_exp(in_path):
    results=[]
    feats_path=f'{in_path}/feats'
    for path_i in data.top_files(feats_path):
        ens_i=ecscf.read_binary_ensemble(path_i)
        result_i=ens_i.evaluate()
        print(result_i.names().cats_stats())
        results.append(result_i)
        print(result_i.get_acc())
    full_results=learn.unify_results(results)
    return full_results.get_acc()

@utils.dir_fun(False)
def check_alg(in_path,clf=None):
    if(clf is None):
        clf=ensemble.RandomForestClassifier()
    results=[]
    feats_path=f'{in_path}/feats'
    for path_i in data.top_files(feats_path):
        common_path_i=f'{path_i}/common'
        data_i= data.read_data(common_path_i)
        result_i=learn.fit_lr(data_i,clf_i=clf)
        results.append(result_i)
    full_results=learn.unify_results(results)
    print(in_path)
    return full_results.get_acc()


if __name__ == "__main__":
    protocol=Protocol()
    protocol('data','uci')
#    out=escf_exp('wine_no_cv')
#    out=check_alg('wine_no_cv')
#    print(out)