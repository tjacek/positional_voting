import files,exp

def gen_csv(in_path,cond,out_path):
    stats_dict=get_stats_dict(in_path,cond)
    dataset_dicts=by_dataset(stats_dict)
    with open(out_path, 'w') as f:
        for name_i,dict_i in dataset_dicts.items():
            txt_i=dict_to_txt(name_i,dict_i)
            f.write(txt_i)
            f.write('\n')

def by_dataset(stats_dict):
    names=list(stats_dict.values())[0].keys()
    dataset_dicts={name_i:{} for name_i in names}
    for name_i in names:
        for desc_j,dict_j in stats_dict.items():
        	dataset_dicts[name_i][desc_j]=dict_j[name_i]
    return dataset_dicts

def get_stats_dict(in_path,cond):
    stats_dict={}
    for cond_i in cond:
        for path_j in find_result(in_path,cond_i):
            clf_j=path_j.split('/')[-2]
            desc_ij=f"{clf_j},{cond_i}"
            stats_dict[desc_ij]=dict(exp.simple_exp(path_j))
    return stats_dict

def find_result(in_path,cond):
    def helper(root):
        return (root.split('/')[-1]==cond)
    return files.find_paths(in_path,helper)

def dict_to_txt(name:str,data_dict:dict):
    lines=[]
    for desc_i,stats_i in data_dict.items():
        txt_i=stats_to_txt(stats_i)
        lines.append(f"{name},{desc_i},{txt_i}")
    lines.sort()
    return "\n".join(lines)

def stats_to_txt(stats):
    return ",".join(["%.4f" % stat_j 
                for stat_j in stats])

gen_csv("B",['raw','borda','opv'],"bayes.csv")