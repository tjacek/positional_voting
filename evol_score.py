from sklearn.metrics import roc_auc_score,accuracy_score
import exp,ens,pref,optim

class AucLoss(object):
    def __init__(self,train_dict):
        self.train_dict=train_dict
        self.n_calls=0

    def __call__(self,score):
        self.n_calls+=1
        result=self.train_dict.positional_voting(score)
        if(result.dim()[0]!=result.n_cats()):
            result.add_column()

        y_true=result.true_one_hot()
        y_pred=result.as_array()

#        auc_ovo = roc_auc_score(y_true,y_pred,multi_class="ovo")

        return -1.0* accuracy_score(y_true,y_pred)
#        return -1.0*auc_ovo

class EvolScore(object):
    def __init__(self,init="latin"):
        self.alg_optim=optim.GenAlg(init_type=init)

    def __str__(self):
        return "OPV"

    @exp.dir_function(clf_decor=True)
    @exp.dir_function(clf_decor=True)
    def __call__(self,in_path:str,out_path:str):
        print(in_path)
        print(out_path)

        votes=ens.read_votes(in_path)
        pref_dict=pref.to_pref(votes.results)
        train,test=pref_dict.split()

        loss_fun=AucLoss(train) 
        n_cand=train.n_cand()
        score=self.alg_optim(loss_fun,n_cand)
        print(score)
        result=test.positional_voting(score)
        result.save(out_path)
        return (out_path,score)

@exp.dir_function(clf_decor=False)
@exp.dir_function(clf_decor=False)
def borda_count(in_path:str,out_path:str):
    votes=ens.read_votes(in_path)
    pref_dict=pref.to_pref(votes.results)
    train,test=pref_dict.split()
    score=pref.borda_weights(test.n_cand())
    result=test.positional_voting(score)
    result.save(out_path)
    return (out_path,score)

#def evol_exp(in_path:str):
#    algs=[exp.simple_acc,borda_count]
#    algs=[exp.simple_acc,borda_count, EvolScore('latin')]
#    alg_dict={ get_name(alg_i):dict(alg_i(in_path)) 
#                   for alg_i in algs}
#    dataset=list(alg_dict.values())[0].keys()
#    lines=[]
#    for data_i in dataset:
#        for vote_j,alg_j in alg_dict.items():
#            print(data_i)
#            stat_j=",".join(["%.4f" % stat 
#                    for stat in alg_j[data_i]])
#            lines.append(f"{data_i},{vote_j},{stat_j}")
#    print(lines)

def evol_exp(in_path):
    paths=[f"{in_path}/{path_i}" 
        for path_i in ["BAG","RF","BOOST"]]
    evol_score= EvolScore()
    output=[]
    for path_i in paths:
        in_i,out_i=f"{path_i}/raw",f"{path_i}/opv"
        output.append(evol_score(in_i,out_i))
    print(output)

evol_exp("C")