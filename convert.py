import numpy as np
import exp,feats

@exp.dir_function()
def from_arff(in_path:str,out_path:str):
    from scipy.io import arff
    dataset = arff.loadarff(in_path)
    X,y=[],[]
    for tuple_i in dataset[0]:
    	tuple_i=list(tuple_i)
    	X.append(tuple_i[:-1])
    	y.append(tuple_i[-1])
    y=norm_cat(y)
    feat_dict=feats.Feats()
    for i,(X_i,y_i) in enumerate(zip(X,y)):
        name_i=f"{y_i}_{i%2}_{i}"
        feat_dict[name_i]=np.array(X_i)
    print(feat_dict.n_cats())
    print(str(feat_dict))
    feat_dict.save(out_path) 

def norm_cat(labels):
    unique=set(labels)
    cat_dict={cat_i:(i+1) 
        for i,cat_i in enumerate(list(unique))}
    return [cat_dict[y_j] for y_j in labels]

from_arff("A/arff","A/common")