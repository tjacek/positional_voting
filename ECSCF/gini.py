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

def prepare_gini(in_path,imbalance_path):
    stats_df=pd.read_csv(in_path)
    imb_df=pd.read_csv(imbalance_path)
    datasets=imb_df['Dataset'].unique()
    gini_df={}
    for data_i in datasets:
        row_i=imb_df[imb_df['Dataset']==data_i]
        row_i=row_i.to_numpy()[0]
        row_i=[float(c_j) for c_j in row_i
                if(is_number(c_j)) ]
        gini_df[data_i]=gini.gini_index(np.array(row_i))
    gini_df=pd.DataFrame.from_dict(gini_df.items())
    gini_df.columns= ['Dataset','gini index']
    final_df=pd.merge(stats_df,gini_df,on='Dataset',how = 'inner')
    print(final_df)

    return final_df

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

if __name__ == "__main__":
    d= prepare_gini()#get_gini_index('imb_json')
    print(d)