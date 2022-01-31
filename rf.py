import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import exp,feats,learn,ens

@exp.dir_function()
@exp.dir_function()
@exp.if_exist
def rf_clf(in_path,out_path):
    print(in_path)
    print(out_path)
    in_path=f"{in_path}/common"
    data_i=feats.read(in_path)[0]
    print(len(data_i))
    data_i.norm()
    train_i,test_i=data_i.split()
    train_tuple=train_i.as_dataset()
    clf_i,params_i=gs_rf(*train_tuple[:2])
    full_tuple=data_i.as_dataset()
    votes_i=get_votes(full_tuple,clf_i)
    votes_i.save(out_path)

def gs_rf(X_train,y_train):
    params={'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]}
    clf_i = ensemble.RandomForestClassifier()#max_depth=3)
    clf_i = GridSearchCV(clf_i,params,
                    verbose=1,
                    scoring='neg_log_loss')
    clf_i.fit(X_train,y_train)    
    best_params=clf_i.cv_results_['params'][0]
    return clf_i.best_estimator_,best_params

def get_votes(result_tuple,clf_i):
    X,y_true,names=result_tuple
    results=[]
    for est_j in clf_i.estimators_:
        y_pred=est_j.predict_proba(X)
        result_j=learn.make_result(y_pred,names)
        results.append(result_j)
    return ens.Votes(results)    

results=rf_clf("A/one_vs_all","A/rf")