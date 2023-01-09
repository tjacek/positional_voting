import json
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
import data,ecscf

class CVFolds(object):
    def __init__(self,data,folds):
        self.data=data
        self.folds=folds

    def __len__(self):
        return len(self.folds)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self):
            data_i=self.as_dataset(self.i)
            self.i+= 1
            return data_i
        else:
            raise StopIteration

    def as_dataset(self,i):
        test=self.folds[i]
        train=[]
        for j,fold_j in enumerate(self.folds):
            if(i!=j):
                train+=fold_j
        new_names={}
        for name_i in train:
            name_i=data.Name(name_i)
            new_names[name_i]=name_i.set_train(True)
        for name_i in test:
            name_i=data.Name(name_i)
            new_names[name_i]=name_i.set_train(False)
        return self.data.rename(new_names)

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

class BayesOptim(object):
    def __init__(self,clf_alg,search_spaces,n_split=5):
        self.clf_alg=clf_alg 
        self.n_split=n_split
        self.search_spaces=search_spaces

    def __call__(self,X_train,y_train):
        cv_gen=RepeatedStratifiedKFold(n_splits=self.n_split, 
                n_repeats=3, random_state=1)
        search = BayesSearchCV(estimator=self.clf_alg(), 
            search_spaces=self.search_spaces,n_jobs=-1,cv=cv_gen)
        search.fit(X_train,y_train) 
#        index=search.best_index_ #
#        params=search.cv_results_[index]
#        raise Exception(best_params)
        best_estm=search.best_estimator_
        return best_estm.get_params(deep=True)

def find_hyperparams(train,n_split=2):
    params={'n_hidden':[25,50,100],'n_epochs':[100,250,500]}
    bayes_cf=BayesOptim(ecscf.ECSCF,params,n_split=n_split)
    train_tuple=train.as_dataset()[:2]
    best_params= bayes_cf(*train_tuple)
    return best_params



if __name__ == "__main__":
    #folds=make_folds('wine.json',k_folds=10)
    #folds.save('wine_cv')
    read_folds('wine_cv')