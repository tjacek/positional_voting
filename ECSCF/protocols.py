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

def one_out_protocol(cv_folds):
    if(type(cv_folds)==str): 
        cv_folds=cv.read_folds(cv_folds)
    partial=[]
    for i in range(len(cv_folds)):
        data_i=cv_folds.as_dataset(i)
        clf_i=ecscf.ECSCF()
        partial.append( clf_i.fit_dataset(data_i))
    result=learn.unify_results(partial)
    print(result.get_acc())


#s=splits.make_splits('wine.json',n_iters=10)
#s.save('wine')
#splits=read_splits('wine')
#split_protocol('wine',alg=None)
one_out_protocol('wine_cv')