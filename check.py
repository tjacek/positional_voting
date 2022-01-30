import exp,feats

@exp.dir_function()
def check(in_path):
    data_i=feats.read(in_path)[0]
    name_i=in_path.split('/')[-1]
    names=data_i.names()
    cats=list(names.unique_cats())
    print(f"{data_i.n_cats()},{len(data_i)},{data_i.dim()[0]}")
    print(cats)

check("B/common")