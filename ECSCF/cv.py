import json,os.path
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
            new_names[name_i]=name_i.set_train(False)
        for name_i in test:
            name_i=data.Name(name_i)
            new_names[name_i]=name_i.set_train(True)
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
#    names.shuffle()
    folds=[[] for i in range(k_folds)]
    cats=names.by_cat()
    for cat_i in cats.values():
        cat_i.shuffle()
        for j,name_j in enumerate(cat_i):
            folds[j % k_folds].append(name_j)
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
                n_repeats=1, random_state=1)
        search = BayesSearchCV(estimator=self.clf_alg(), 
            search_spaces=self.search_spaces,n_jobs=-1,cv=cv_gen)
        search.fit(X_train,y_train) 
        best_estm=search.best_estimator_
        return best_estm.get_params(deep=True)

def find_hyperparams(train,params,ensemble_type=ecscf.OneVsAll,n_split=2):
    if(type(train)==str):
        train=data.read_data(train)

#    ensemble_factory= ecscf.OneVsAll
    bayes_cf=BayesOptim(ensemble_type,params,n_split=n_split)
    train_tuple=train.as_dataset()[:2]
    best_params= bayes_cf(*train_tuple)
    return best_params

def prepare_folds(in_path,fold_path,n_split):
    if(os.path.exists(fold_path)):
        cv_folds=read_folds(fold_path)
    else:    
        cv_folds=make_folds(in_path,k_folds=n_split)
        cv_folds.save(fold_path)
    return cv_folds

if __name__ == "__main__":
#    read_folds('wine_cv')
    eval('wine.json',n_split=10)