from sklearn import ensemble
import os.path
import data,exp,splits,ecscf,cv,learn,utils

def split_protocol(splits_group,alg=None):
    if(type(splits_group)==str):
    	splits_group=splits.read_splits(splits_group)
    if(alg is None):
        alg=ensemble.RandomForestClassifier()	
    acc=[exp.simple_exp(split_i,alg) 
             for split_i in splits_group]
    exp.stats(str(alg),acc)

@utils.iter_fun(n_iters=2)
def one_out_protocol(in_path,out_path):
    data.make_dir(out_path)
    fold_path=f'{out_path}/fold'
    feat_path=f'{out_path}/feats'
    if(os.path.exists(fold_path)):
        cv_folds=cv.read_folds(fold_path)
    else:    
        cv_folds=cv.make_folds(in_path,k_folds=10)
        cv_folds.save(fold_path)
    data.make_dir(feat_path)
    hyperparams=cv.find_hyperparams(cv_folds.data,n_split=2)
    for i,data_i in enumerate(cv_folds):
        out_i=f'{feat_path}/{i}'
        data.make_dir(out_i)
        clf_i=ecscf.ECSCF(**hyperparams)
        data_i.save(f'{out_i}/common')
        datasets=clf_i.fit_dataset(data_i,features=True)
        datasets.save(f'{out_i}/binary')  

def escf_exp(in_path):
    results=[]
    for path_i in data.top_files(in_path):
        ens_i=ecscf.read_binary_ensemble(path_i)
        result_i=ens_i.evaluate()
        results.append(result_i)
        print(result_i.get_acc())
    full_results=learn.unify_results(results)
    full_results.report()

one_out_protocol('wine.json','wine_cv2')
#escf_exp('wine_cv/feats')