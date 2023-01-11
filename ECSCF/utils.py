import data
from functools import wraps

def iter_fun(n_iters=2):
    def decor_fun(fun):
        @wraps(fun)
        def iter_decorator(*args, **kwargs):
            out_path= args[1]
            data.make_dir(out_path)
            for i in range(n_iters):
                new_args=list(args)
                new_args[1]=f'{out_path}/{i}'
                fun(*new_args,**kwargs)
            return None
        return iter_decorator
    return decor_fun

def dir_fun(fun):
    @wraps(fun)
    def dir_decorator(*args, **kwargs):
        in_path= args[0]
        output=[]
        for path_i in data.top_files(in_path):
            new_args=list(args)
            new_args[0]=path_i
            out_i=fun(*new_args,**kwargs)
            output.append(out_i)
        return output
    return dir_decorator
