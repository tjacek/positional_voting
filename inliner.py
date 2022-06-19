import numpy as np
from sklearn import neighbors
import format,exp,ens,learn,files

def inliner_stats(results):
    old,new=list(zip(*results))
    metric_fun=exp.MetricStats()
    stats= metric_fun(old)
    stats+=metric_fun(new)
    return stats

@exp.dir_function(clf_decor=False)
@exp.ResultExp(inliner_stats)
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

def count_improv(output):
    for dataset_i in output:
        name_i=dataset_i[0][0]
        n_neutral=sum([ int(exp_j==0) 
            for name_i,exp_j in dataset_i])
        n_pos=sum([ int(exp_j>0) 
            for name_i,exp_j in dataset_i])
        print(f"{name_i},{n_pos},{n_neutral}")

in_path= "data"
output=inliner_voting(in_path)
print(output)
#count_improv(output)
#paths=format.find_result(in_path,lambda x:False)
#print(paths)