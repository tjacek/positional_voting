import numpy as np
from sklearn import neighbors
import exp,ens,learn,output,files

def inliner_voting(in_path):
    paths=(f"{in_path}/common",f"{in_path}/binary")
    votes=ens.make_votes(paths,clf="LR")
    binary_feats=ens.read_dataset((0,paths[1]))
    inliners=[get_knn(feats_i)
                 for feats_i in binary_feats]
    preds,names=[],inliners[0].keys()
    for name_i in names:
        inliner_i=get_row(name_i,inliners)
        votes_i=get_row(name_i,votes.results)
        if(sum(inliner_i)>2 ):
        	s_votes_i=[result_j 
        	    for j,result_j in enumerate(votes_i)
        	        if(inliner_i[j])]
        else:
            s_votes_i=votes_i
        preds.append( np.sum(s_votes_i,axis=0))
    old_results=votes.voting()
    new_results=learn.make_result(preds,names)
    return old_results,new_results

def get_knn(data_i,k=3):
    clf= neighbors.KNeighborsClassifier(k)
    result_i=learn.train_model(data_i,clf_type=clf)
    test_result= result_i.split()[1]     
    y_true,y_pred,names=test_result.get_pred()
    correct=[ true_i==pred_i 
        for true_i,pred_i in zip(y_true,y_pred)]
    inliner_dict=dict(zip(names,correct))  
    return inliner_dict

def get_row(name_i,dicts):
    return [dict_i[name_i] for dict_i in dicts]

def exp_template(in_path,metric,cols):
    fun=exp.ResultExp(metric)(inliner_voting)
    fun=exp.dir_function(clf_decor=False)(fun)
    stats=fun(in_path)
    return output.as_dataframe(stats,cols=cols)

def count_improv(in_path):
    def helper(results):    
        sucess,neutral=[],[]
        for old_i,new_i in results:
            diff_i=  new_i.get_acc()-old_i.get_acc()
            sucess.append(int(diff_i>0))
            neutral.append(int(diff_i==0))
        sucess,neutral=np.sum(sucess),np.sum(neutral)
        fail= len(results)-sucess-neutral
        return [sucess,neutral,fail]
    cols=['Dataset','sucess','neutral','fail']
    return exp_template(in_path,helper,cols)

def inliner_stats(in_path):
    def helper(results):
        old,new=list(zip(*results))
        metric_fun=exp.MetricStats()
        stats= metric_fun(old)
        stats+=metric_fun(new)
        return stats
    cols=['Dataset','old_mean','old_std','old_max',
        'new_mean','new_std','new_max']
    return exp_template(in_path,helper,cols)

in_path= "data"
df=inliner_stats(in_path)
df=df.round(4)
print(df.to_csv())