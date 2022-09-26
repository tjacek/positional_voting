import numpy as np
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.metrics import roc_auc_score
import json,os
import data,learn

class Votes(object):
    def __init__(self,results):
        self.results=results

    def save(self,out_path):
        data.make_dir(out_path)
        for i,result_i in enumerate(self.results):
            result_i.save(f'{out_path}/{i}')

def read_votes(in_path):
    paths=[f'{in_path}/{path_i}' 
        for path_i in os.listdir(in_path)]
    results=[]
    for path_i in paths:
        with open(path_i, 'r') as f:
             result_i=learn.Result(data.read_data(path_i))
             results.append(result_i)
    return Votes(results)

class Result(data.DataDict):
    def get_pred(self):
        train,test=self.split()
        names=test.names()
        y_true=[name_i.get_cat() for name_i in names]
        y_pred=[np.argmax(self[name_i]) for name_i in names]
        return y_true,y_pred,names

    def get_acc(self):
        y_true,y_pred,names=self.get_pred()
        return accuracy_score(y_true,y_pred)

def make_result(y_pred,names):
    y_pred=np.array(y_pred)
    if( y_pred.ndim<2 or y_pred.shape[1]==1):
    	y_pred=to_one_hot(y_pred)
    raw=[(name_i,pred_i) 
            for name_i,pred_i in zip(names,y_pred)]
    return Result(raw)

def to_one_hot(y,n_cats=None):
    if(n_cats is None):
        n_cats=np.amax(y)+1
    one_hot=[]
    for y_i in y:
        one_hot.append(np.zeros((n_cats,)))
        one_hot[-1][y_i]=1.0
    return np.array(one_hot)