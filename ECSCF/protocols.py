from sklearn import ensemble
import data,exp,splits,ecscf

def split_protocol(splits_group,alg=None):
    if(type(splits_group)==str):
    	splits_group=splits.read_splits(splits_group)
    if(alg is None):
        alg=ensemble.RandomForestClassifier()	
    acc=[exp.simple_exp(split_i,alg) 
             for split_i in splits_group]
    exp.stats(str(alg),acc)

#s=splits.make_splits('wine.json',n_iters=10)
#s.save('wine')
#splits=read_splits('wine')
split_protocol('wine',alg=None)#ecscf.ECSCF())