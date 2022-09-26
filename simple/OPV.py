import numpy as np
import data,learn

class PrefDict(data.DataDict):
    def n_cand(self):
        return list(self.values())[0].shape[1]

    def get_rank(self,name_i,k):
        pref_ij=self[name_i]
        return pref_ij[:,k] 

    def positional_voting(self,score=None):
        if(score is None):
            score=np.flip(np.arange(self.n_cand()))
        for name_i,pref_i in self.items():
        	print(pref_i)
        print(score)

def make_pref(votes:learn.Votes):
    pref=PrefDict()
    names=votes.results[0].names()
    for name_i in names:
        pref_i=[ np.argsort(result_j[name_i]) 
            for result_j in votes.results]
        print(len(pref))
        pref[name_i]=np.array(pref_i)
    return pref

votes=learn.read_votes('0/RF')
pref=make_pref(votes)
print(pref.positional_voting())