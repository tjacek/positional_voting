import format,exp,ens

@exp.dir_function(clf_decor=False)
@exp.dir_function(clf_decor=False)
def fun(in_path):
    paths=(f"{in_path}/common",f"{in_path}/binary")
    votes=ens.make_votes(paths,clf="LR")
    binary_feats=ens.read_dataset((0,paths[1]))
    print(len(binary_feats))

in_path= "data"
fun(in_path)
#paths=format.find_result(in_path,lambda x:False)
#print(paths)