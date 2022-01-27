import os
import numpy as np
from functools import wraps
import files,ens

def dir_function(recreate=True):
    def decor_fun(fun):
        @wraps(fun)
        def dir_decorator(*args, **kwargs):
            out_dir=(len(args)>1)
            output=[]
            if(out_dir):
            	files.make_dir(args[1])
            for path_i in files.top_files(args[0]): 
                if(out_dir):
                    name_i=path_i.split("/")[-1]
                    out_i=f"{args[1]}/{name_i}"
                    new_args=(path_i,out_i)	
                else:
                	new_args=(path_i,)
                print(new_args)
                output.append( fun(*new_args,**kwargs))
            return output
        return dir_decorator
    return decor_fun

def acc_exp(fun):
    @wraps(fun)
    def dir_decorator(in_path):
        results=[fun(path_i) #ens.read_votes(path_i).voting()
        for path_i in files.top_files(in_path)]
        acc=[result_i.get_acc() for result_i in results]
        mean_acc,std_acc=np.mean(acc),np.std(acc)
        name_i=in_path.split("/")[-1] 
        return (name_i,mean_acc,std_acc)
    return dir_decorator

def basic_exp(fun):
    @wraps(fun)
    def helper(in_path):
        result=fun(in_path)
        result.report()
        print(result.get_acc())
    return helper

def multi_iter(n_iters=10):
    def helper(fun):
        @wraps(fun)
        def decorator(*args, **kwargs):
            files.make_dir(args[1])
            output=[]
            for i in range(n_iters):
                in_i=f"{args[0]}/{i}"  
                out_i=f"{args[1]}/{i}"
                new_args=(in_i,out_i)
                output.append(fun(*new_args,**kwargs))
            return output
        return decorator
    return helper

@dir_function(recreate=False)
@dir_function(recreate=False)
def ens_results(in_path,out_path=None,clf="LR"):
    paths=(f"{in_path}/common",f"{in_path}/binary")
    votes=ens.make_votes(paths,clf=clf)
    votes.save(out_path)

@dir_function(recreate=False)
@acc_exp
def simple_acc(in_path):
    return ens.read_votes(in_path).voting() 

if __name__ == "__main__":
    acc=simple_acc("B/results")
    print(acc)   