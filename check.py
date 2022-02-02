import exp,feats

@exp.dir_function()
def check(in_path):
    data_i=feats.read(in_path)[0]
    name_i=in_path.split('/')[-1]
    name_i=name_i.split('.')[0]
    names=data_i.names()
    ratio=class_ratio(names)
    print(f"{name_i},{data_i.n_cats()},{str(data_i)},{ratio}")

def class_ratio(names):
    stats=names.cats_stats()
    class_size=list(stats.values())
    return min(class_size)/max(class_size)

check("A/common")