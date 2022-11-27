import protocols,data,clfs

class SimpleBinary(object):
    def __init__(self,n_hidden=25):
        self.n_hidden=n_hidden
        	
    def fit(self,data_i):
        return None

def binary_clf():
    clf = SimpleBinary   
    params={'n_hidden': [25,50,100]}
    return clfs.ClfAlg(clf,params,"binary")

d=data.read_data("cleveland.json")
train=d.split()[0]
clf_alg=binary_clf()
protocols.find_hyperparams(train,clf_alg)