import numpy as np
from sklearn import neighbors
import format,exp,ens,learn

@exp.dir_function(clf_decor=False)
@exp.dir_function(clf_decor=False)
def fun(in_path):
    paths=(f"{in_path}/common",f"{in_path}/binary")
    votes=ens.make_votes(paths,clf="LR")
    binary_feats=ens.read_dataset((0,paths[1]))
    inliners=[get_knn(feats_i)
                 for feats_i in binary_feats]
    print(len(binary_feats))


def get_knn(data_i,k=3):
    clf= neighbors.KNeighborsClassifier(k)
    result_i=learn.train_model(data_i,clf_type=clf)
    test_result= result_i.split()[1]     
    y_true,y_pred,names=test_result.get_pred()
    inliner_dict=dict(zip(names,zip(y_true,y_pred)))  
    return inliner_dict

in_path= "data"
fun(in_path)
#paths=format.find_result(in_path,lambda x:False)
#print(paths)