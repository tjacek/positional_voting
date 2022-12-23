from sklearn import ensemble
import data,exp

class SplitGroup(list):
    def __init__(self, arg=[]):
        super(SplitGroup, self).__init__(arg)

    def save(self,out_path):
    	data.make_dir(out_path)
    	for i,split_i in enumerate(self):
    		split_i.save(f'{out_path}/{i}')

def make_splits(raw_dataset,n_iters=2):
    if(type(raw_dataset)==str):
    	raw_dataset=data.read_data(raw_dataset)
    splits=SplitGroup()
    for i in range(n_iters):
        split_i=raw_dataset.random()#split(None,shuffle=True)
        splits.append(split_i)
    return splits

def read_splits(in_path):
    splits=SplitGroup()
    for path_i in data.top_files(in_path):
        splits.append(data.read_data(path_i))
    return splits

def split_protocol(splits,alg=None):
    if(type(splits)==str):
    	splits=read_splits(splits)
    if(alg is None):
        alg=ensemble.RandomForestClassifier()	
    acc=[exp.simple_exp(split_i,alg) 
             for split_i in splits]
    exp.stats(str(alg),acc)

splits=make_splits('wine.json',n_iters=10)
splits.save('wine')
#splits=read_splits('wine')
split_protocol('wine',alg=None)