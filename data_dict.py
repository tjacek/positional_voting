import files

class DataDict(dict):
    def __init__(self, arg=[]):
        super(DataDict, self).__init__(arg)

    def __setitem__(self, key, value):
        if(type(key)==str):
            key=files.Name(key)
        super(DataDict, self).__setitem__(key, value)
    
    def n_cats(self):
        return self.names().n_cats()

    def names(self):
        keys=sorted(self.keys(),key=files.natural_keys) 
        return files.NameList(keys)

    def split(self,selector=None,names_only=False):
        train,test=split(self,selector)
        if(names_only):
            return train,test
        return self.__class__(train),self.__class__(test)

    def transform(self,fun,copy=False):
        new_dict= self.__class__() if(copy) else self
        for name_i,data_i in self.items():
            new_dict[name_i]=fun(data_i)
        return new_dict

    def subset(self,names,new_names=False):
        sub_dict=self.__class__()
        for i,name_i in enumerate( names):
            value_i=self[name_i]
            if(new_names):
                name_i=f'{name_i}_{i}'
            sub_dict[name_i]=value_i
        return sub_dict 

    def rename(self,name_dict):
        new_feats= self.__class__()
        for name_i,name_j in name_dict.items():
            new_feats[files.Name(name_j)]=self[name_i]
        return new_feats

    def rename_cat(self,cat_dict=None):
        if(cat_dict is None):
            names=self.names()
            cat_dict={ j:(i+1) 
                for i,j in enumerate(names.unique_cats())}
        name_dict={}
        for name_i in self.keys():
            cat_i=name_i.get_cat()
            new_name_i="_".join(name_i.split("_")[1:])
            new_name_i=f"{cat_dict[cat_i]}_{new_name_i}"
            name_dict[name_i]=new_name_i
        return self.rename(name_dict)

    def random(self):
        names=self.names()
        half=int(len(names)/2)
        rename_dict={
           name_i:f"{name_i.get_cat()+1}_{int(i<half)}_{i}"
                for i,name_i in enumerate(names)}
        return self.rename(rename_dict)

    def balanced_split(self):
        names=self.names()
        rename_dict={}
        for cat_i,names_i in names.by_cat().items():
            names_i.shuffle()
            for j,name_j in enumerate(names_i):
                new_name_j=f"{cat_i+1}_{j%2}_{len(rename_dict)}"
                rename_dict[name_j]=new_name_j
        return self.rename(rename_dict)

def split(data_dict,selector=None,pairs=True):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in data_dict.keys():
        pair_i=(name_i,data_dict[name_i]) if(pairs) else name_i
        if(selector(name_i)):
            train.append(pair_i)
        else:
            test.append(pair_i)
    return train,test

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1