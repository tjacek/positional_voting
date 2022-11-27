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

if __name__ == "__main__":
    gen_votes("splits",'RF')#,"0")