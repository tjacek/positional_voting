from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
import exp,ens,pref,optim

class LossFun(object):
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
    return -1.0*accuracy_score(y_true,y_pred)

def f1_metric(y_true,y_pred):
    return -1.0*f1_score(y_true,y_pred,average='macro')

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

        loss_fun=LossFun(train,metric=self.metric) 
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

def all_voting(in_path:str,dirs:list):
    metrics={"acc":acc_metric,"auc":auc_metric,"f1":f1_metric}
    evol_funcs={ f'opv_{name_i}':EvolScore(metric=metric_i) 
                    for name_i,metric_i in metrics.items()}
    evol_funcs["borda"]=borda_count
    paths=[f"{in_path}/{path_i}" 
            for path_i in dirs]
    output=[]
    for path_i in paths:
        for name_j,fun_j in evol_funcs.items():
            in_j,out_j=f"{path_i}/raw",f"{path_i}/{name_j}"
            print(out_j)
            output.append(fun_j(in_j,out_j))
    print(output)

def single_exp(in_path,out_path=None):  
    if(out_path is None):
        out_path=f"{in_path}/opv_acc"
        in_path=f"{in_path}/raw"
    evol_score= EvolScore()
    evol_score(in_path,out_path)

dirs=["one_vs_all","BAG","RF","BOOST"]
all_voting("A/cv",dirs)