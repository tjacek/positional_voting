import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import exp,feats,learn,ens,files

class ClfCV(object):
    def __init__(self,find_best,get_votes):
        self.find_best=find_best
        self.get_votes=get_votes

    @exp.dir_function(clf_decor=True)
    @exp.dir_function(clf_decor=True)
#    @exp.if_exist
    def __call__(self,in_path,out_path):
        in_path=f"{in_path}/common"
        data_i=feats.read(in_path)[0]
        print(len(data_i))
        data_i.norm()
        train_i,test_i=data_i.split()
        train_tuple=train_i.as_dataset()
        clf_i,params_i=self.find_best(*train_tuple[:2])
        full_tuple=data_i.as_dataset()
        votes_i=self.get_votes(full_tuple,clf_i)
        votes_i.save(out_path)

class GridOptim(object):
    def __init__(self,clf,params):
        self.clf=clf
        self.params=params

    def __call__(self,X_train,y_train):
        clf_i = GridSearchCV(self.clf(),self.params,
                    verbose=1,
                    scoring='neg_log_loss')
        clf_i.fit(X_train,y_train)    
        best_params=clf_i.cv_results_['params'][0]
        return clf_i.best_estimator_,best_params

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

def rf_clf(bayes=True):
    params={'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]}
    clf = ensemble.RandomForestClassifier
    search_alg= BayesOptim if(bayes) else GridOptim
    grid=search_alg(clf,params)
    return ClfCV(grid,get_votes)

def bag_clf(bayes=True):
    params={'n_estimators': [5,10,15,20]}
    clf = ensemble.BaggingClassifier
    search_alg= BayesOptim if(bayes) else GridOptim
    grid=search_alg(clf,params)
    return ClfCV(grid,get_votes)

def boost_clf(bayes=True):
    params={'max_depth': [2,4,6],'n_estimators': [5,10,15,20]}
    clf = ensemble.GradientBoostingClassifier
    search_alg= BayesOptim if(bayes) else GridOptim
    grid=search_alg(clf,params)
    return ClfCV(grid,get_boost_votes)

def get_boost_votes(result_tuple,clf_i):
    X,y_true,names=result_tuple
    results=[]
    for est_j in clf_i.estimators_:
        y_raw=np.array([tree.predict(X)
                          for tree in est_j])        
        if(y_raw.shape[0]>1):
            y_pred=np.argmax(y_raw,axis=0)
        else:
            y_pred= (y_raw<0).astype(int)
            y_pred= y_pred.ravel()
        result_j=learn.make_result(y_pred,names)
        results.append(result_j)
    return ens.Votes(results)

def get_votes(result_tuple,clf_i):
    X,y_true,names=result_tuple
    results=[]
    for est_j in clf_i.estimators_:
        y_pred=est_j.predict_proba(X)
        result_j=learn.make_result(y_pred,names)
        results.append(result_j)
    return ens.Votes(results)

def clf_exp(in_path,out_path):
    files.make_dir(out_path)
    algs={"RF":rf_clf(),
          "BAG":bag_clf(),
          "BOOST":boost_clf()}
    for name_i,alg_i in algs.items():
        out_i=f"{out_path}/{name_i}"
        alg_i(in_path,out_i)

clf_exp("A/one_vs_all","A/cv")