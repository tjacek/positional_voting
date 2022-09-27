from functools import wraps
import os

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def get_paths(in_path):
    return [f'{in_path}/{path_i}' 
        for path_i in os.listdir(in_path)]

def dir_exp(fun):
    @wraps(fun)
    def helper(in_path,out_path):
        make_dir(out_path)
        for path_i in os.listdir(in_path):
            in_i=f'{in_path}/{path_i}'
            out_i=f'{out_path}/{path_i}'
            fun(in_i,out_i)
    return helper