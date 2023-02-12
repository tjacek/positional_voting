import learn,data

class Ensemble(object):
    def __init__(self,full,binary,clf_type=None):
        self.binary=binary 
        self.full=full
        self.clf_type=clf_type

    def evaluate(self,as_votes=False):
        results=[]
        for full_i in self.full:
            result_i=learn.fit_lr(full_i,self.clf_type)
            results.append(result_i)
        results=[result_i.split()[1] 
            for result_i in results]
        if(as_votes):
            return learn.Votes(results)
        return learn.voting(results)

class EnsembleFactory(object):
    def __init__(self,clf_type=None):
        self.clf_type=clf_type

    def __call__(self,in_path):
        common_path=f'{in_path}/common'
        binary_path=f'{in_path}/binary'
        common=data.read_data(common_path)
        binary=data.read_data_group(binary_path)
        full=[ common.concat(binary_i) 
            for binary_i in binary]
        return Ensemble(full,binary,self.clf_type)

class RawBinary(object):
    def __init__(self,clf_type=None):
        self.clf_type=clf_type

    def __call__(self,in_path):
        binary_path=f'{in_path}/binary'
        binary=data.read_data_group(binary_path)
        return Ensemble(binary,binary,self.clf_type)