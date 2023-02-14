import os

def walk(in_path):
    total_size=0
    for root, dir, files in os.walk("."):
        for file_i in files:
    	    path_i=f"{root}/{file_i}"
    	    size=os.path.getsize(path_i)
    	    total_size+=size
    	    print(f'{path_i}:{size/1000}')
    print(total_size/1000)

walk('imb')