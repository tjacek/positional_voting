from sklearn import ensemble

class ClfAlg(object):
    def __init__(self,alg,hyperparams):
        self.alg=alg
        self.hyperparams=hyperparams

    def fit(self,data,hyper):
        clf_k=self.alg(**hyper)
        data_tuple=data.as_dataset()
        clf_k.fit(*data_tuple[:2])
        return clf_k

def bag_clf():
    clf = ensemble.BaggingClassifier    
    params={'n_estimators': [5,10,15,20]}
    return ClfAlg(clf,params)

def rf_clf():
    clf = ensemble.RandomForestClassifier
    params={'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]}
    return ClfAlg(clf,params)    