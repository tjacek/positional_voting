from sklearn import neighbors
import data,ecscf,learn,protocols,utils

class InlinerVoting(object):
    def __init__(self,k=3):
        self.k=k

    @utils.dir_fun(False)
    @utils.unify_cv(dir_path='feats',show=True)
    def __call__(self,in_path):
        print(in_path)
        ens_i=ecscf.read_binary_ensemble(in_path)
        votes_i= ens_i.evaluate(True)
        knn=get_knn(ens_i,k=self.k)
        s_clf={}
        for name_j in votes_i.names():
            votes_j=votes_i.get_classes(name_j)
            knn_j=knn.get_classes(name_j)
            s_clf[name_j]=[t 
                    for t,(v,k) in enumerate(zip(votes_j,knn_j))
                        if(v==k)]
        inliner_result=votes_i.dynamic_voting(s_clf)
        return inliner_result

def get_knn(ens_i,k=3):
    clf= neighbors.KNeighborsClassifier(k)
    results=[learn.fit_lr(binary_j,clf) 
                for binary_j in ens_i.binary] 
    return learn.Votes(results)

#def inliner_voting(in_path):
inliner_voting=InlinerVoting()
acc=inliner_voting('wine_cv2')
print(acc)