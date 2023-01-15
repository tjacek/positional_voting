from sklearn import neighbors
import data,ecscf,learn

def inliner_voting(in_path):
    ens_i=ecscf.read_binary_ensemble(in_path)
    votes_i= ens_i.evaluate(True)
    knn=get_knn(ens_i,k=3)
    for name_i in votes_i.names():
        print(votes_i.get_classes(name_i))
        print(knn.get_classes(name_i))
#    for feats_i in ens.feats:
#        feats_i.split()

def get_knn(ens_i,k=3):
    clf= neighbors.KNeighborsClassifier(k)
    results=[learn.fit_lr(binary_j,clf) 
                for binary_j in ens_i.binary] 
    return learn.Votes(results)

inliner_voting('wine_cv2/0/feats/0')