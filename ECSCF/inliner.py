from sklearn import neighbors
import data,ecscf

def inliner_voting(in_path):
	ens_i=ecscf.read_binary_ensemble(in_path)
	votes_i= ens_i.evaluate()
	print(votes_i)
#    for feats_i in ens.feats:
#        feats_i.split()

def get_knn(data_i,k=3):
    clf= neighbors.KNeighborsClassifier(k)

inliner_voting('wine_cv2/0/feats/0')