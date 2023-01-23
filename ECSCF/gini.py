import numpy as np
import data

def gini_index(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))

def get_gini_index(in_path):
    gini_dict={}
    for path_i in data.top_files(in_path):
        data_i=data.read_data(path_i)
        cats_i=data_i.names().cats_stats()
        print(cats_i)
        name_i=path_i.split('/')[-1]
        values=np.array(list(cats_i.values()))
        gini_dict[name_i]=gini_index(values)
    return gini_dict

if __name__ == "__main__":
    d= get_gini_index('imb_json')
    print(d)