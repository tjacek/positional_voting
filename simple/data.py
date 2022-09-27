import numpy as np
from sklearn import preprocessing
import json,random,os.path

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def names(self):
        keys=list(self.keys())
        keys.sort()
        return keys

    def norm(self):
        names=list(self.keys())
        X=np.array([self[name_i] for name_i in names])
        new_X=preprocessing.scale(X)
        for i,name_i in enumerate(names):
            self[name_i]=new_X[i]

    def as_dataset(self):
        names=self.names()
        return self.get_X(names),self.get_labels(names),names
    
    def get_X(self,names=None):
        if(names is None):
            names=self.names()
        return np.array([self[name_i] for name_i in names])

    def get_labels(self,names=None):
        if(names is None):
            names=self.names()
        return [ name_i.get_cat() for name_i in names]

    def split(self,selector=None):
        if(selector is None):
            selector=lambda name_i: name_i[1]==0
        train,test=[],[]
        for name_i in self.names():#keys():
            pair_i=(name_i,self[name_i])
            if(selector(name_i)):
                train.append(pair_i)
            else:
                test.append(pair_i)
        return self.__class__(train),self.__class__(test)

    def save(self,out_path):
        raw_dict={}
        for name_i,data_i in self.items():
            raw_dict[str(name_i)]=list(data_i)
        with open(out_path, 'w') as f:
            json.dump(raw_dict, f)

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def __len__(self):
        return len(self.split('_'))
    
    def get_cat(self):
        return int(self.split('_')[0])-1

    def __getitem__(self,i):
        return int(self.split('_')[i])

    def set_train(self,trainset:bool):
        raw=self.split('_')
        raw[1]=str(int(trainset))
        return Name('_'.join(raw))	

def read_data(in_path):
    with open(in_path, 'r') as f:
        data_dict = json.load(f)
        data_dict={ Name(name_i):np.array(data_i)
            for name_i,data_i in data_dict.items()}
        return DataDict(data_dict)

def gen_split(data_dict,out_path,n=10):
    if(type(data_dict)==str):
        data_dict=read_data(data_dict)
    make_dir(out_path)
    names=list(data_dict.keys())
    for i in range(n):
        random.shuffle(names)
        dict_i={ name_k.set_train((k%2)==0):data_dict[name_k]
                for k,name_k in enumerate(names)}
        DataDict(dict_i).save(f'{out_path}/{i}')
        print(names[:10])

if __name__ == "__main__":
    gen_split('cleveland.json','splits')
#print(data_dict)