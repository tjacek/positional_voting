import numpy as np
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
import data

class Result(data.DataDict):
    def get_pred(self):
        y_pred,y_true=[],[]
        for name_i,vote_i in self.items():
            if(type(vote_i)==np.ndarray):
                y_pred.append(np.argmax(vote_i))
            else:
                y_pred.append(vote_i)
            y_true.append(name_i.get_cat())
        print(y_pred)
        return y_pred,y_true
    
    def get_acc(self):
        y_pred,y_true=self.get_pred()
        return accuracy_score(y_pred,y_true)

    def report(self):
        y_pred,y_true=self.get_pred()
        print(classification_report(y_true, y_pred,digits=4))
	
def make_result(names,y_pred):
    result=[(name_i,pred_i) 
            for name_i,pred_i in zip(names,y_pred)]
    return Result(result)

def unify_results(results):
    pairs=[]
    for result_i in results:
        pairs+=result_i.items()
    return Result(pairs)

def voting(results):
    names= results[0].keys()
    pairs=[]
    for name_i in names:
        ballot_i=[result_i[name_i] 
            for result_i in results]
        count_i=np.sum(ballot_i ,axis=0)
        cat_i=np.argmax(count_i)
        pairs.append((name_i,cat_i))
    return Result(pairs)

def fit_lr(data_dict_i,clf_i=None):
    data_dict_i.norm()
    train,test= data_dict_i.split()
    if(clf_i is None):
        clf_i=LogisticRegression(solver='liblinear')
    X_train,y_train,names=train.as_dataset()
    clf_i.fit(X_train,y_train)
    X_test,y_true,names=test.as_dataset()
    y_pred=clf_i.predict_proba(X_test)
    return make_result(names,y_pred)