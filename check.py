import numpy as np
import exp,feats,format

@exp.dir_function(show=False)
def check(in_path):
    data_i=feats.read(in_path)[0]
    name_i=in_path.split('/')[-1]
    name_i=name_i.split('.')[0]
    names=data_i.names()
    ratio=1/class_ratio(names)
    print(f"{name_i},{data_i.n_cats()},{str(data_i)},{ratio:.2f}")

def class_ratio(names):
    stats=names.cats_stats()
    class_size=list(stats.values())
    return min(class_size)/max(class_size)

@exp.dir_function(show=False)
def class_cumul(in_path):
    data_i=feats.read(in_path)[0]
    stats=data_i.names().cats_stats()
    cat_sizes=list(stats.values())
    cat_sizes.sort()
    cat_sizes.reverse()
    cat_sizes=np.array(cat_sizes)/sum(cat_sizes)
    cumul_dist=cat_sizes#np.cumsum(cat_sizes)
    name_i=in_path.split('/')[-1]
    str_cumul=",".join([f"{100*dist:.2f}" for dist in cumul_dist ])    
    line_i=f"{name_i},{str_cumul}"
    print(line_i)
    return line_i

output=class_cumul("A/common")
format.to_csv(output,"imbal")