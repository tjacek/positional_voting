from sklearn import neighbors
import data,ecscf,learn

def inliner_voting(in_path):
    ens_i=ecscf.read_binary_ensemble(in_path)
    votes_i= ens_i.evaluate(True)
    knn=get_knn(ens_i,k=3)
    s_clf={}
    for name_j in votes_i.names():
        votes_j=votes_i.get_classes(name_j)
        knn_j=knn.get_classes(name_j)
        s_clf[name_j]=[t 
                for t,(v,k) in enumerate(zip(votes_j,knn_j))
                    if(v==k)]
    inliner_result=votes_i.dynamic_voting(s_clf) 
    print(inliner_result.get_acc())
    print(votes_i.vote().get_acc())

def get_knn(ens_i,k=3):
    clf= neighbors.KNeighborsClassifier(k)
    results=[learn.fit_lr(binary_j,clf) 
                for binary_j in ens_i.binary] 
    return learn.Votes(results)

inliner_voting('wine_cv2/0/feats/0')