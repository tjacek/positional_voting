import os
import utils,data,ens

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
def to_gzip(in_path,out_path):
    factory=ens.EnsembleFactory()
    paths=data.top_files(f'{in_path}/feats')
    for i,path_i in enumerate(paths):
        out_i=f'{out_path}/{i}'
        ens_i=factory(path_i)
        ens_i.as_gzip(out_i)
        print(out_i)

to_gzip('imb','imb_gzip')