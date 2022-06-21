import os,re#,itertools
from functools import wraps

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def __len__(self):
        return len(self.split('_'))

    def clean(self):
        digits=[ str(int(digit_i)) 
                for digit_i in re.findall(r'\d+',self)]
        return Name("_".join(digits))

    def get_cat(self):
        return int(self.split('_')[0])-1

    def get_person(self):
        return int(self.split('_')[1])

    def subname(self,k):
        subname_k="_".join(self.split("_")[:k])
        return Name(subname_k)

class NameList(list):
    def __new__(cls, name_list=None):
        if(name_list is None):
            name_list=[]
        return list.__new__(cls,name_list)

    def n_cats(self):
        return len(self.unique_cats())

    def unique_cats(self):
        return set(self.get_cats())

    def get_cats(self):
        return [name_i.get_cat() for name_i in self]     

    def binarize(self,j):
        return [ int(cat_i==0) for cat_i in self.get_cats()]

    def by_cat(self):
        cat_dict={cat_j:NameList() 
                for cat_j in self.unique_cats()}
        for name_i in self:
            cat_dict[name_i.get_cat()].append(name_i)
        return cat_dict

    def cats_stats(self):
        stats_dict={ cat_i:0 for cat_i in self.unique_cats()}
        for cat_i in self.get_cats():
            stats_dict[cat_i]+=1
        return stats_dict

    def subset(self,indexes):
        return NameList([self[i] for i in indexes])

    def filtr(self,cond):
        return NameList([name_i for i,name_i in enumerate(self) 
                           if cond(i,name_i)])

    def shuffle(self):
        random.shuffle(self)
        return self

def natural_sort(l):
    return sorted(l,key=natural_keys)

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text)]

def atoi(text):
    return int(text) if text.isdigit() else text

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def find_paths(in_path,cond):
    paths=[]
    for root, subdirs, files in os.walk(in_path):
        if(cond(root)):
            paths.append(root)
        for file_i in files:
            if(cond(file_i)):
                paths.append(f"{root}/{file_i}")            
    return paths

def save(out_path,obj):
    import pickle 
    with open(out_path,"wb") as out_file:   
        pickle.dump(obj,out_file) 
        out_file.close()

def read_csv(in_path):
    import csv
    with open(in_path, newline='\n') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        return [[row.strip() for row in line]
                    for line in csv_reader]

def rec_type(obj):
    type_i=type(obj)
    if(type_i==list or type_i==tuple):
        return [rec_type(obj_i) 
                    for obj_i in obj]
    if(type_i==dict):
        return {name_i:rec_type(data_i)
          for name_i,data_i in obj.items()}
    if(type_i==str):
        return obj
    return str(type(obj))