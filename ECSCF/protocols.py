from sklearn import ensemble
import os.path
import data,exp,splits,ecscf,cv,learn

def split_protocol(splits_group,alg=None):
    if(type(splits_group)==str):
    	splits_group=splits.read_splits(splits_group)
    if(alg is None):
        alg=ensemble.RandomForestClassifier()	
    acc=[exp.simple_exp(split_i,alg) 
             for split_i in splits_group]
    exp.stats(str(alg),acc)

def one_out_protocol(in_path,out_path):
    data.make_dir(out_path)
    common,binary=f'{out_path}/common',f'{out_path}/binary'
    data.make_dir(binary)
    if(os.path.exists(common)):
        cv_folds=cv.read_folds(common)
    else:    
        cv_folds=cv.make_folds(in_path,k_folds=10)
        cv_folds.save(common)
    for i,data_i in enumerate(cv_folds):
        clf_i=ecscf.ECSCF()
        datasets=clf_i.fit_dataset(data_i,features=True)
        datasets.save(f'{out_path}/binary/{i}')  

#def one_out_save(cv_folds,out_path):
#    if(type(cv_folds)==str): 
#        cv_folds=cv.read_folds(cv_folds)
#    binary=f'{out_path}/binary'
#    data.make_dir(out_path)
#    data.make_dir(binary)
#    for i in range(len(cv_folds)):
#        data_i=cv_folds.as_dataset(i)
#        clf_i=ecscf.ECSCF()
#        datasets=clf_i.fit_dataset(data_i,features=True)
#        datasets.save(f'{binary}/{i}')

def escf_exp(in_path):
    common_path=f'{in_path}/common/data'
    binary_path=f'{in_path}/binary'
    common=data.read_data(common_path)
#    binary=data.read_data_group(binary_path)
    results=[]
    for path_i in data.top_files(binary_path):
        binary_i=ecscf.read_ensemble(path_i)
        full_i=binary_i.concat(common)
#        feats_i=common.concat(binary_i)
#        result_i= ecscf.fit_lr(feats_i)
#        result_i=ecscf.fit_lr(data_i)
#        results.append(result_i)
#    full_results=learn.unify_results(results)
#    full_results.report()

#one_out_protocol('wine.json','wine_cv')
escf_exp('wine_cv')