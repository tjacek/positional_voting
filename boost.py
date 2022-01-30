import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import exp,feats,learn,ens

@exp.dir_function()
@exp.dir_function()
#@exp.multi_iter(n_iters=10,in_iter=False)
def boost_clf(in_path,out_path):
    print(in_path)
    print(out_path)
    votes_i=train_boost(f"{in_path}/common")
    votes_i.save(out_path)

def train_boost(in_path):
    data_i=feats.read(in_path)[0]
    data_i.norm()
    train_i,test_i=data_i.split()
    train_tuple=train_i.as_dataset()
    clf_i,params_i=gs_boost(*train_tuple[:2])
    full_tuple=data_i.as_dataset()
    votes_i=get_boost_votes(full_tuple,clf_i)
    return votes_i

def gs_boost(X_train,y_train):
    clf_i = ensemble.GradientBoostingClassifier()#max_depth=3)
    clf_i = GridSearchCV(clf_i,{'max_depth': [2,4,6],
                    'n_estimators': [5,10,15,20]},
                    verbose=1,
                    scoring='neg_log_loss')
    clf_i.fit(X_train,y_train)    
    best_params=clf_i.cv_results_['params'][0]
    return clf_i.best_estimator_,best_params

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
        result_j=learn.make_result(y_pred,names) #learn.Result(y_true,y_pred,names)
        results.append(result_j)
    return ens.Votes(results)

results=boost_clf("C/one_vs_all","C/boost")