import pandas

data=pandas.read_csv("auc.csv")

def to_dict(data,key_names):
    data = data.reset_index()  
    data_dict={}
    for index, row in data.iterrows():
        key_i="_".join([row[name_i] 
            for name_i in key_names])
        data_dict[key_i]=row
    return data_dict

data_dict=to_dict(data,['dataset','clf'])
print(data_dict.keys())