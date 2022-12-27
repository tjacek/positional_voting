import json
import data

class CVFolds(object):
    def __init__(self,data,folds):
        self.data=data
        self.folds=folds

    def save(self,out_path):
        data.make_dir(out_path)
        self.data.save(f'{out_path}/data')
        with open(f'{out_path}/folds', 'w') as f:
            json.dump(self.folds, f)

def make_folds(data_dict,k_folds=10):
    if(type(data_dict)==str):
        data_dict=data.read_data(data_dict)
    names=data_dict.names()
    names.shuffle()
    folds=[[] for i in range(k_folds)]
    for i,name_i in enumerate(names):
        folds[i % k_folds].append(name_i)
    return CVFolds(data_dict,folds)

def read_folds(in_path):
    with open(f'{in_path}/folds', 'r') as f:
        folds = json.load(f)
        data_dict=data.read_data(f'{in_path}/data')
        return CVFolds(data_dict,folds)

#folds=make_folds('wine.json',k_folds=10)
#folds.save('wine_cv')
read_folds('wine_cv')