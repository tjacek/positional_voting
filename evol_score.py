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


@exp.dir_function(args=None)
@exp.acc_exp
#@exp.basic_exp
def evol(in_path):
    print(in_path)
    votes=ens.read_votes(in_path)
    pref_dict=pref.to_pref(votes.results)
    train,test=pref_dict.split()

    loss_fun=AucLoss(train) 
    n_cand=train.n_cand()
    alg_optim=optim.GenAlg()
    score=alg_optim(loss_fun,n_cand)
    print(score)
    result=test.positional_voting(score)
    return result

in_path="B/results/"#wine/0"
acc_dict=evol(in_path)
print(acc_dict)