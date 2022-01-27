import numpy as np
import ens,data_dict,exp,learn

class PrefDict(data_dict.DataDict):
    def n_votes(self):
        return list(self.values())[0].shape[0]

    def n_cand(self):
        return list(self.values())[0].shape[1]

    def get_rank(self,name_i,k):
        pref_ij=self[name_i]
        return pref_ij[:,k]      

    def positional_voting(self,score):
        names= self.names() 
        y_pred=[ self.score_rule(name_i,score) 
                for name_i in names]
        return learn.make_result(y_pred,names)

    def score_rule(self,name_i,score):
        n_cand=self.n_cand()
        count=np.zeros((n_cand,))
        for j in range(n_cand):
            for vote_k in self.get_rank(name_i,j):
                count[vote_k]+=score[j]
        return np.argmax(count)

@exp.dir_function(args=None)
@exp.acc_exp
#@exp.basic_exp
def election(in_path:str):
    results=ens.read_votes(in_path).results
    pref_dict=to_pref(results)
    score=borda_weights(pref_dict.n_cand())
    return pref_dict.positional_voting(score)
#    names= pref_dict.names()
#    y_true=names.get_cats()
#    y_pred=[ pref_dict.score_rule(name_i,score) 
#                for name_i in names]
#    return learn.make_result(y_pred,names)

def to_pref(results):
    pref_dict=PrefDict()
    names=results[0].names()
    for name_i in names:
        pref_i=[]
        for result_j in results:
            pred_j=result_j[name_i]
            pref_ij=np.flip(np.argsort(pred_j))
            pref_i.append(pref_ij)
        pref_dict[name_i]=np.array(pref_i)
    return pref_dict

def borda_weights(n_cand):
    return np.array([n_cand-j 
                for j in range(n_cand)])

if __name__ == "__main__":
    in_path="B/results/"#wine/0"
    acc=election(in_path)
    print(acc)