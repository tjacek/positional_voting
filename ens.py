import numpy as np
import feats,files,learn

class Votes(object):
    def __init__(self,results):
        self.results=results

    def __len__(self):
        return len(self.results)

    def voting(self):
        names= self.results[0].names()
        preds=[result_i.as_array(names) 
                 for result_i in self.results]
        preds=np.array(preds)
        preds=np.sum(preds,axis=0)  
        print(preds.shape)
        return learn.make_result(preds,names)

    def save(self,out_path):
        files.save(out_path,self)

def make_votes(paths,clf="LR"):
    datasets=read_dataset(paths)
    results=[learn.train_model(data_i,clf_type=clf)
                for data_i in datasets]
    return Votes(results)  

def read_dataset(paths):
    common_path,binary_path=paths
    if(common_path):
        common=feats.read(common_path)
    if(not binary_path):
        return common
    datasets=[]
    for path_i in files.top_files(binary_path):
        binary_i=feats.read(path_i)[0]
        if(common_path):
            binary_i+=common[0]
        datasets.append(binary_i)
    return datasets
     
if __name__ == "__main__":
    in_path="../ml_utils/gen/B/ensembles/wine/0"
    common_path=f"{in_path}/common"
    binary_path=f"{in_path}/binary"
    votes=make_votes((common_path,binary_path))
    result=votes.voting()
    result.report()
    votes.save("test")