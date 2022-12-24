import data

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
        split_i=raw_dataset.random()
        splits.append(split_i)
    return splits

def read_splits(in_path):
    splits=SplitGroup()
    for path_i in data.top_files(in_path):
        splits.append(data.read_data(path_i))
    return splits