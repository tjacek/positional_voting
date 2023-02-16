import os
import utils,data

def show_size(in_path):
    total_size=0
    for root, dir, files in os.walk("."):
        for file_i in files:
    	    path_i=f"{root}/{file_i}"
    	    size=os.path.getsize(path_i)
    	    total_size+=size
    	    print(f'{path_i}:{size/1000}')
    print(total_size/1000)

@utils.dir_map(2)
#@utils.unify_cv(dir_path='feats')
def to_gzip(in_path,out_path):
#    print(data.top_files(in_path))
     print(in_path,out_path)

to_gzip('imb','imb_gzip')