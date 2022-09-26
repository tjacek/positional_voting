import numpy as np
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

def optim(pref,maxiter=100):
    n_cand=pref.n_cand()
    bound_w = [(0.0, n_cand)  for _ in range(n_cand)]
    init= 'latinhypercube'
    def loss_fun(score):
        result=pref.positional_voting(score)
        return -1.0*result.get_acc()    
    result = differential_evolution(loss_fun, bound_w, 
            init=init,maxiter=maxiter, tol=1e-7)	
    weights= result['x']
    score= np.zeros((n_cand,))
    score[0]=1
    print(pref.positional_voting(score).get_acc())
    opt_result=pref.positional_voting(weights)
    print(opt_result.get_acc())

votes=learn.read_votes('0/RF')
pref=make_pref(votes)
optim(pref)
#result=pref.positional_voting()
#print(result.get_acc())