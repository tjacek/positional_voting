from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
import exp,ens,pref,optim

class AucLoss(object):
    def __init__(self,train_dict,metric=None):
        if(metric is None):
            metric=acc_metric
        self.train_dict=train_dict
        self.metric=metric
        self.n_calls=0

    def __call__(self,score):
        self.n_calls+=1
        result=self.train_dict.positional_voting(score)
        y_true=result.true_one_hot()
        y_pred=result.as_array()
        return self.metric(y_true,y_pred)

def auc_metric(y_true,y_pred):
    auc_ovo=roc_auc_score(y_true,y_pred,multi_class="ovo")
    return -1.0*auc_ovo

def acc_metric(y_true,y_pred):
    return accuracy_score(y_true,y_pred)

def f1_metric(y_true,y_pred):
    return f1_score(y_true,y_pred,average='macro')

class EvolScore(object):
    def __init__(self,init="latin",metric=None):
        self.alg_optim=optim.GenAlg(init_type=init)
        self.metric=metric

    @exp.dir_function(clf_decor=True)
    @exp.dir_function(clf_decor=True)
    def __call__(self,in_path:str,out_path:str):
        print(in_path)
        print(out_path)

        votes=ens.read_votes(in_path)
        pref_dict=pref.to_pref(votes.results)
        train,test=pref_dict.split()

        loss_fun=AucLoss(train,metric=self.metric) 
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

def evol_exp(in_path,name="opv"):
    paths=[f"{in_path}/{path_i}" 
        for path_i in ["BAG","RF","BOOST"]]
    evol_score= EvolScore(metric=f1_metric)
    output=[]
    for path_i in paths:
        in_i,out_i=f"{path_i}/raw",f"{path_i}/{name}"
        output.append(evol_score(in_i,out_i))
    print(output)

def single_exp(in_path,out_path=None):  
    if(out_path is None):
        out_path=f"{in_path}/opv_acc"
        in_path=f"{in_path}/raw"
    evol_score= EvolScore()
    evol_score(in_path,out_path)

evol_exp("B",name="opv_f1")
#in_path= "B/one_vs_all/"
#borda_count("B/one_vs_all/raw","B/one_vs_all/borda")
#single_exp(in_path)