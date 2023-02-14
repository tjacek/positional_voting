import json,gzip
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

    def as_gzip(self,in_path):
        raw_dict={'common':self.full.save(),
            'binary':[binary_i.save() for binary_i in self.binary]}
        with gzip.open(in_path, 'wb') as f:
            json_str = json.dumps(raw_dict) + "\n"               # 2. string (i.e. JSON)
            json_bytes = json_str.encode('utf-8') 
            f.write(json_bytes)

class EnsembleFactory(object):
    def __init__(self,clf_type=None):
        if(clf_type is None):
            import output
            clf_type=output.get_clf('LR')
        self.clf_type=clf_type

    def __call__(self,in_path):
        common_path=f'{in_path}/common'
        binary_path=f'{in_path}/binary'
        common=data.read_data(common_path)
        binary=data.read_data_group(binary_path)
#        full=[ common.concat(binary_i) 
#            for binary_i in binary]
        full=common
        return Ensemble(full,binary,self.clf_type)

class RawBinary(object):
    def __init__(self,clf_type=None):
        self.clf_type=clf_type

    def __call__(self,in_path):
        binary_path=f'{in_path}/binary'
        binary=data.read_data_group(binary_path)
        return Ensemble(binary,binary,self.clf_type)

if __name__ == "__main__":
    in_path='imb/wall-following/0/feats/0'
    ens_factory=EnsembleFactory()
    ens_i=ens_factory(in_path)
    ens_i.as_gzip('test.gzip')