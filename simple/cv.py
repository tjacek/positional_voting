import numpy as np
from sklearn import ensemble
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import data,learn,utils

class BayesOptim(object):
    def __init__(self,clf,params,n_split=5):
        self.clf=clf 
        self.params=params
        self.n_split=n_split

    def __call__(self,X_train,y_train):
        cv_gen=RepeatedStratifiedKFold(n_splits=self.n_split, 
                n_repeats=3, random_state=1)
        search = BayesSearchCV(estimator=self.clf(), 
            search_spaces=self.params,n_jobs=-1,cv=cv_gen)
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

def bag_clf():
    params={'n_estimators': [5,10,15,20]}
    clf = ensemble.BaggingClassifier
    return clf,params 
#def rf_clf():
#    params={'max_depth': [3, 5, 10],
#            'min_samples_split': [2, 5, 10]}
#    clf = ensemble.RandomForestClassifier
#    return clf,params    

#@utils.dir_exp
def gen_votes(in_path,out_path,n_split=5):
    data_i=data.read_data(in_path)
    data_i.norm()
    train_i,test_i=data_i.split()
    
    train_tuple=train_i.as_dataset()
    clf,params= bag_clf()
    bayes_cf=BayesOptim(clf,params)
    best_params= bayes_cf(*train_tuple[:2])[1]
    result=learn.Result()
    for k in range(n_split):
        selector_k=SplitSelector(k, n_split)
        in_k,out_k=data_i.split(selector_k)
        
        in_tuple=in_k.as_dataset() 
        clf_k=clf(**best_params)
        clf_k.fit(*in_tuple[:2])

        out_tuple=out_k.as_dataset() 
        y_pred_k=clf_k.predict_proba(out_tuple[0])
        for name_t,pred_t in zip(out_tuple[-1],y_pred_k):
        	result[name_t]=pred_t
    result.save(out_path)
#    print(len(result))
#    print(best_params)

if __name__ == "__main__":
    gen_votes("splits/0",'BAG')#,"0")