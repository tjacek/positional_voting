from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import exp,feats,learn

@exp.dir_function()
def boost_clf(in_path,out_path):
    data_i=feats.read(in_path)[0]
    data_i.norm()
    train_i,test_i=data_i.split()
    train_tuple=train_i.as_dataset()
    clf_i,params_i=train_boost(*train_tuple[:2])
    X,y,names=data_i.as_dataset()
    y_pred=clf_i.predict_proba(X)
    return learn.make_result(y_pred,names)

def train_boost(X_train,y_train):
    clf_i = ensemble.GradientBoostingClassifier()#max_depth=3)
    clf_i = GridSearchCV(clf_i,{'max_depth': [2,4,6],
                    'n_estimators': [5,10,15,20]},
                    verbose=1,
                    scoring='neg_log_loss')
    clf_i.fit(X_train,y_train)    
    best_params=clf_i.cv_results_['params'][0]
    return clf_i.best_estimator_,best_params

results=boost_clf("B/common","B/boost")
for result_i in results:
	print(result_i.get_acc())