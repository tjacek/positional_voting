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
                	new_args=(path_i)
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
