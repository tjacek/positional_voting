import numpy as np
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn import ensemble
import data,ecscf

def simple_exp(d,clf_alg):
    train,test=d.split()
    X,y,names=train.as_dataset()
    clf_alg.fit(X,y)
    X,y_true,names=test.as_dataset()
    y_pred=clf_alg.predict(X)
    acc=accuracy_score(y_true,y_pred)
    print(f'{clf_alg}:{acc}')
    return acc

def compare_clfs(d,n_iters=10):
    algs=[ecscf.ECSCF(),
          ensemble.RandomForestClassifier(),
          ensemble.BaggingClassifier()]
    results={}
    for alg_i in algs:
        print(str(alg_i))	
        acc=[simple_exp(d,alg_i)
               for j in range(n_iters)]
        results[str(alg_i)]=acc
    print('***********************')
    for name_i,acc_i in results.items():
    	stats(name_i,acc_i)

def stats(name_i,acc_i):
    mean_i,std_i=np.mean(acc_i),np.std(acc_i)	
    print(f'{name_i}:{mean_i:.4}+/-{std_i:.4}')

if __name__ == "__main__":
    d=data.read_data('wine.json')
    compare_clfs(d)