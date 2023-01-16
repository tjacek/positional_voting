from functools import wraps
import os.path
import data

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

def lazy_dir_fun(fun):
    @wraps(fun)
    def helper(*args, **kwargs):
        in_path,out_path=args[1],args[2]
        data.make_dir(out_path)
        for in_i in data.top_files(in_path):
            id_i=(in_i.split('/')[-1]).split('.')[0]
            out_i=f'{out_path}/{id_i}'
            if(not os.path.exists(out_i)):
                new_args=list(args)
                new_args[1]=in_i
                new_args[2]=out_i
                fun(*new_args,**kwargs)
#                print(out_i)
        return None
    return helper