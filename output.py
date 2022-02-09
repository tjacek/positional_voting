import numpy as np
import pandas

def by_voting(paths):
    dataset=[pandas.read_csv(path_i)
                for path_i in paths]
    vote_dicts={}
    key_names=['dataset','clf']
    for data_i in dataset:
        for vote_j in data_i.voting.unique():
            data_ij=data_i[data_i['voting']==vote_j]	
            vote_dicts[vote_j]=to_dict(data_ij,key_names)
    return vote_dicts

def to_dict(data,key_names):
    data = data.reset_index()  
    data_dict={}
    for index, row in data.iterrows():
        key_i="_".join([row[name_i] 
            for name_i in key_names])
        data_dict[key_i]=row
    return data_dict

def compare_output(pair,attr,vote_dicts):
    old,new=vote_dicts[pair[0]],vote_dicts[pair[1]]
    all_diff=[]
    for name_i in old.keys():
    	old_attr=old[name_i][attr]
    	new_attr=new[name_i][attr]
    	diff= new_attr-old_attr
    	print(f"{name_i},{diff/old_attr}")
    	all_diff.append( diff)
    print(np.median(all_diff))

vote_dicts=by_voting(['bayes.csv','auc2.csv'])
print(vote_dicts.keys())
compare_output(['raw','opv_auc2'],'auc_mean',vote_dicts)