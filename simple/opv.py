import numpy as np
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.metrics import roc_auc_score
from scipy.optimize import differential_evolution
import data,learn

class PrefDict(data.DataDict):
    def n_cand(self):
        return list(self.values())[0].shape[1]

    def get_rank(self,name_i,k):
        pref_ij=self[name_i]
        return pref_ij[:,k] 

    def positional_voting(self,weights=None):
        n_cand=self.n_cand()
        if(weights is None):
            weights=np.flip(np.arange(n_cand))
        y_pred,names=[],[]
        for name_i,ballot_i in self.items():
            score_i=np.zeros((n_cand,))
            for vote_j in ballot_i: 
                for k,cand in enumerate(vote_j):
                    score_i[cand]+= weights[n_cand-k-1]
            names.append(name_i)
            y_pred.append(np.argmax(score_i))
        return learn.make_result(y_pred,names)

def make_pref(votes:learn.Votes):
    pref=PrefDict()
    names=votes.results[0].names()
    for name_i in names:
        pref_i=[ np.argsort(result_j[name_i]) 
            for result_j in votes.results]
        print(len(pref))
        pref[name_i]=np.array(pref_i)
    return pref

class OPV(object):
    def __init__(self,metric=None,maxiter=100,
           init='latinhypercube'):
        if(metric is None):
            metric=acc_metric
#        self.loss_fun=loss_fun
        self.metric=metric
        self.maxiter=maxiter
        self.init=init

    def __call__(self,pref):
        test,train=pref.split()
        loss_fun= LossFun(pref,self.metric)
        n_cand=pref.n_cand()
        bound_w = [(0.0, n_cand)  for _ in range(n_cand)]
        result = differential_evolution(loss_fun, 
            bound_w, init=self.init,
            maxiter=self.maxiter, tol=1e-7)	
        weights= result['x']
        return weights

class LossFun(object):
    def __init__(self,train_dict,metric=None,cum=False):
        if(metric is None):
            metric=acc_metric
        self.train_dict=train_dict
        self.metric=metric
        self.n_calls=0
        self.cum=cum

    def __call__(self,score):
        if(self.cum):
            score=np.cumsum(score)
            score=np.flip(score)
#            score.reverse()
#            raise Exception(score)
        self.n_calls+=1
        result=self.train_dict.positional_voting(score)
        y_true,y_pred,names=result.get_pred()
        return self.metric(y_true,y_pred)

def auc_metric(y_true,y_pred):
    auc_ovo=roc_auc_score(y_true,y_pred,multi_class="ovo")
    return -1.0*auc_ovo

def acc_metric(y_true,y_pred):
    return -1.0*accuracy_score(y_true,y_pred)

def f1_metric(y_true,y_pred):
    return -1.0*f1_score(y_true,y_pred,average='macro')

def to_opv(loss,maxiter=100,init='latinhypercube'):
    return [ OPV(loss_i,maxiter,init) for loss_i in loss]

if __name__ == "__main__":
#    votes=learn.read_votes('cleveland_RF')
#    pref=make_pref(votes)
#    optim(pref)
    to_opv([acc_metric,acc_metric,f1_metric])