import os.path
import numpy as np
from functools import wraps
import files,ens

def dir_function(clf_decor=False,show=True):
    def decor_fun(fun):
        @wraps(fun)
        def dir_decorator(*args, **kwargs):
            in_path,out_path=get_args(args,clf_decor) 
            if(out_path):
            	files.make_dir(out_path)
            output=[]
            for path_i in files.top_files(in_path): 
                new_args=[path_i]                 
                if(out_path):
                    name_i=path_i.split("/")[-1]
                    out_i=f"{out_path}/{name_i}"
                    new_args.append(out_i)	
                if(clf_decor):
                    new_args=[args[0]]+new_args
                new_args=tuple(new_args)
                if(show):
                    print(new_args)
                output.append(fun(*new_args,**kwargs))
            return output
        return dir_decorator
    return decor_fun

def get_args(args,clf_decor:bool):
    x,y=0+int(clf_decor),1+int(clf_decor)
    in_path=args[x]
    out_path= args[y] if(len(args)>y) else None
    return in_path,out_path

class ResultExp(object):
    def __init__(self,stats_fun=None):
        if(stats_fun is None):
            stats_fun=acc_stats
        self.stats_fun=stats_fun

    def __call__(self,fun):
        @wraps(fun)
        def helper(*args, **kwargs):
            in_path= args[0] if(len(args)==1) else args[1]
            results=[]
            for path_i in files.top_files(in_path):
                if(len(args)==1):
                    results.append(fun(path_i))
                else:
                    results.append(fun(*(args[0],path_i)))
            stats=self.stats_fun(results)
            name_i=in_path.split("/")[-1] 
            return (name_i,stats)
        return helper

def acc_stats(results):
    acc=[result_i.get_acc() for result_i in results]
    stats=[fun(acc) for fun in [np.mean,np.std,np.amax]]
    return stats

def auc_stats(results):
    acc=[result_i.get_auc() for result_i in results]
    stats=[fun(acc) for fun in [np.mean,np.std,np.amax]]
    return stats

def if_exist(fun):
    @wraps(fun)
    def helper(in_path,out_path):
        if(not os.path.exists(out_path)):
            fun(in_path,out_path)
    return helper

def basic_exp(fun):
    @wraps(fun)
    def helper(in_path):
        result=fun(in_path)
        result.report()
        print(result.get_acc())
    return helper

def multi_iter(n_iters=10,in_iter=False):
    def helper(fun):
        @wraps(fun)
        def decorator(*args, **kwargs):
            files.make_dir(args[1])
            output=[]
            for i in range(n_iters):
                if(in_iter):
                    in_i=f"{args[0]}/{i}"  
                else:
                    in_i=args[0]
                out_i=f"{args[1]}/{i}"
                new_args=(in_i,out_i)
                output.append(fun(*new_args,**kwargs))
            return output
        return decorator
    return helper

@dir_function()
@dir_function()
def ens_results(in_path,out_path=None,clf="LR"):
    paths=(f"{in_path}/common",f"{in_path}/binary")
    votes=ens.make_votes(paths,clf=clf)
    votes.save(out_path)

@dir_function()
@ResultExp(auc_stats)
def simple_auc(in_path):
    return ens.read_votes(in_path).voting() 

if __name__ == "__main__":
#    ens_results("A/one_vs_all","A/results")
    acc=simple_auc("A/bag")
    print(acc)   