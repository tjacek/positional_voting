import os
import numpy as np
from functools import wraps
import files,ens

def dir_function(args=None):
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

@dir_function(args=None)
@dir_function(args=None)
def ens_results(in_path,out_path=None,clf="LR"):
    paths=(f"{in_path}/common",f"{in_path}/binary")
    votes=ens.make_votes(paths,clf=clf)
    votes.save(out_path)

@dir_function(args=None)
def acc_exp(in_path):
    results=[ens.read_votes(path_i).voting()
        for path_i in files.top_files(in_path)]
    acc=[result_i.get_acc() for result_i in results]
    mean_acc,std_acc=np.mean(acc),np.std(acc)
    name_i=in_path.split("/")[-1] 
    return (name_i,mean_acc,std_acc)

if __name__ == "__main__":
    acc=acc_exp("B/results")
    print(acc)   