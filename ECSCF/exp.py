from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn import ensemble

import data,ecscf

def simple_exp(d,clf_alg):
    train,test=d.split()
    X,y,names=train.as_dataset()
    clf_alg.fit(X,y)
    X,y_true,names=test.as_dataset()
    y_pred=clf_alg.predict(X)
    print(f'{clf_alg}:{accuracy_score(y_true,y_pred)}')

def compare_clfs(d):
    clf_alg=ecscf.ECSCF()
    simple_exp(d,clf_alg)
    clf_alg = ensemble.RandomForestClassifier()
    simple_exp(d,clf_alg)
    clf_alg = ensemble.BaggingClassifier()
    simple_exp(d,clf_alg)

d=data.read_data('wine.json')
compare_clfs(d)

