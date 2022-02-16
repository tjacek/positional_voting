import numpy as np
import re
import scipy.io
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
    feat_dict=to_feats(X,y)
    print(feat_dict.n_cats())
    print(str(feat_dict))
    feat_dict.save(out_path) 

def to_feats(X,y):
    y=norm_cat(y)
    feat_dict=feats.Feats()
    for i,(X_i,y_i) in enumerate(zip(X,y)):
        name_i=f"{y_i}_{i%2}_{i}"
        feat_dict[name_i]=np.array(X_i)
    return feat_dict

def norm_cat(labels):
    unique=set(labels)
    cat_dict={cat_i:(i+1) 
        for i,cat_i in enumerate(list(unique))}
    return [cat_dict[y_j] for y_j in labels]

@exp.dir_function()
def from_keel(in_path:str,out_path:str):
    samples = [sample_i.strip()#.split() 
         for sample_i in open(in_path).readlines()]
    samples=[sample_i.replace('<null>','0')
                for sample_i in samples
                    if(not '@' in sample_i) ]
    X,y=[],[]
    for sample_i in samples:
    	sample_i=[ float(cord_j)
                    for cord_j in sample_i.split(',')]
    	X.append(sample_i[:-1])
    	y.append(sample_i[-1])
    feat_dict=to_feats(X,y)
    feat_dict.save(out_path)

def from_data(in_path,out_path):
    import csv
    X,y=[],[]
    with open(in_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            line_i=[float(cord_j) for cord_j in row]
            X.append(line_i[:-1])
            y.append(int(line_i[-1]))
    feat_dict=to_feats(X,y)
    feat_dict.save(out_path)        

from_data("A/raw/glass/glass.data","A/datasets/glass")