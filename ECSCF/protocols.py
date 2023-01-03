from sklearn import ensemble
import data,exp,splits,ecscf,cv,learn

def split_protocol(splits_group,alg=None):
    if(type(splits_group)==str):
    	splits_group=splits.read_splits(splits_group)
    if(alg is None):
        alg=ensemble.RandomForestClassifier()	
    acc=[exp.simple_exp(split_i,alg) 
             for split_i in splits_group]
    exp.stats(str(alg),acc)

#def one_out_protocol(cv_folds):
#    if(type(cv_folds)==str): 
#        cv_folds=cv.read_folds(cv_folds)
#    partial=[]
#    for i in range(len(cv_folds)):
#        data_i=cv_folds.as_dataset(i)
#        clf_i=ecscf.ECSCF()
#        partial.append( clf_i.fit_dataset(data_i))
#    result=learn.unify_results(partial)
#    print(result.get_acc())

def one_out_save(cv_folds,out_path):
    if(type(cv_folds)==str): 
        cv_folds=cv.read_folds(cv_folds)
    data.make_dir(out_path)
    for i in range(len(cv_folds)):
        data_i=cv_folds.as_dataset(i)
        clf_i=ecscf.ECSCF()
        datasets=clf_i.fit_dataset(data_i,features=True)
        datasets.save(f'{out_path}/{i}')
        print(len(datasets))

def escf_exp(in_path):
    data_group=data.read_data_group(in_path)
    results=[]
    for data_i in data_group:
        result_i=ecscf.fit_lr(data_i)
        results.append(result_i)
    full_results=learn.unify_results(results)
    print(full_results.get_acc())

#s=splits.make_splits('wine.json',n_iters=10)
#s.save('wine')
#splits=read_splits('wine')
#split_protocol('wine',alg=None)
#one_out_save('wine_cv','wine_binary')
escf_exp('wine_binary/0')