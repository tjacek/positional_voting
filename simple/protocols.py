import cv,data

#class Protocol(object):
#    def __init__(self,n=10):
#        self	

def gen_splits(data_i,n_splits=10):
    if(type(data_i)==str):
        data_i=data.read_data(data_i)
    for k in range(n_splits):
        selector_k=cv.SplitSelector(k,n_splits)
        in_k,out_k=data_i.split(selector_k)
        print(len(in_k),len(out_k))
        yield in_k,out_k

if __name__ == "__main__":
    for in_k,out_k in gen_splits("cleveland"):
        print(len(in_k))
        print(len(out_k))