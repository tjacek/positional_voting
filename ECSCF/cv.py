import data

class CVFolds(object):
    def __init__(self,data,folds):
        self.data=data
        self,folds=folds


def make_folds(data_dict,k_folds=10):
    if(type(data_dict)==str):
        data_dict=data_dict.read_data(data_dict)
    names=data_dict.names()
    names.shuffle()
    folds=[[] for i in range(k_folds)]
    for i,name_i in names:
        folds[i % k_folds].append(name_i)
    return CVFolds(data_dict,folds)
    	