from sklearn.metrics import roc_auc_score
import exp,ens,pref,optim

class AucLoss(object):
    def __init__(self,train_dict):
        self.train_dict=train_dict
        self.n_calls=0

    def __call__(self,score):
        result=self.train_dict.positional_voting(score)
        y_true=result.true_one_hot()
        y_pred=result.as_array()
        n_cats=result.n_cats()
        auc_ovo = roc_auc_score(y_true,y_pred,multi_class="ovo")
        self.n_calls+=1
        return -1.0*auc_ovo

class EvolScore(object):
    def __init__(self,init="borda"):
        self.alg_optim=optim.GenAlg(init_type=init)

    def __str__(self):
        return str(self.alg_optim)

    @exp.dir_function(recreate=False,clf_decor=True)
    @exp.acc_exp
    def __call__(self,in_path:str):
        print(in_path)
        votes=ens.read_votes(in_path)
        pref_dict=pref.to_pref(votes.results)
        train,test=pref_dict.split()

        loss_fun=AucLoss(train) 
        n_cand=train.n_cand()
        score=self.alg_optim(loss_fun,n_cand)
        print(score)
        result=test.positional_voting(score)
        return result

def evol_exp(in_path:str):
    algs=[exp.simple_acc,EvolScore('borda'),EvolScore('latin')]
    alg_dict={ str(alg_i):dict(alg_i(in_path)) 
                   for alg_i in algs}
    dataset=list(alg_dict.values())[0].keys()
    lines=[]
    for data_i in dataset:
        for vote_j,alg_j in alg_dict.items():
            print(data_i)
            stat_j=",".join(["%.4f" % stat 
                    for stat in alg_j[data_i]])
            lines.append(f"{data_i},{vote_j},{stat_j}")
    print(lines)

in_path="B/results/"#wine/0"
evol_exp(in_path)