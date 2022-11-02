import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
from collections import defaultdict
import data,learn,utils

class BayesOptim(object):
    def __init__(self,clf_alg,n_split=5):
        self.clf_alg=clf_alg 
        self.n_split=n_split

    def __call__(self,X_train,y_train):
        cv_gen=RepeatedStratifiedKFold(n_splits=self.n_split, 
                n_repeats=3, random_state=1)
        search = BayesSearchCV(estimator=self.clf_alg.alg(), 
            search_spaces=self.clf_alg.hyperparams,n_jobs=-1,cv=cv_gen)
        search.fit(X_train,y_train) 
        best_params=search.cv_results_['params'][0]
        return search.best_estimator_,best_params

class SplitSelector(object):
    def __init__(self,k,n_split):
        self.k=k 
        self.n_split=n_split 
        self.i=0 

    def __call__(self,name_i):
        select= (self.i % self.n_split)==self.k	
        self.i+=1
        return select

class BalancedSelector(object):
    def __init__(self,k,n_split):
        self.k=k 
        self.n_split=n_split 
        self.class_counter=defaultdict(lambda :0)

    def __call__(self,name_i):
        cat_i=name_i.get_cat()
        select=(self.class_counter[cat_i] % self.n_split)
        select= (select==self.k)
        self.class_counter[cat_i]+=1
        return select

@utils.dir_exp
def gen_votes(in_path,out_path,n_split=5):
    data_i=data.read_data(in_path)
    data_i.norm()
    train_i,test_i=data_i.split()
    
    train_tuple=train_i.as_dataset()
#    clf,params=bag_clf()
    clf,params=rf_clf()
    bayes_cf=BayesOptim(clf,params)
    best_estm,best_params= bayes_cf(*train_tuple[:2])
#    n_clfs=best_params['n_estimators'] #best_estm.estimators_)

    votes=learn.Votes([])#n_clfs)
    for k in range(n_split):
        selector_k=SplitSelector(k, n_split)
        in_k,out_k=data_i.split(selector_k)
        in_tuple=in_k.as_dataset()         
        clf_k=clf(**best_params)
        clf_k.fit(*in_tuple[:2])
        out_tuple=out_k.as_dataset() 
        for j,estm_j in enumerate(clf_k.estimators_):
            if(len(votes)<(j+1)):
                votes.results.append(learn.Result())
            y_pred_j=estm_j.predict_proba(out_tuple[0])
            for name_t,pred_t in zip(out_tuple[-1],y_pred_j):
           	    votes.results[j][name_t]=pred_t
    votes.results=[result_j for result_j in votes.results
                    if(len(result_j)>0)]
    votes.save(out_path)     

if __name__ == "__main__":
    gen_votes("splits",'RF')#,"0")