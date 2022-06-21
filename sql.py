import pandas as pd
import exp,feats,files,format

def as_latex(in_path,cols):
    df=pd.read_csv(in_path)
    new_cols={}
    for col_i in cols:
        if(type(col_i)==tuple):
            ts_i=df[col_i[0]]-df[col_i[1]]
            name_i=f"diff_{col_i[0]}"
            new_cols[name_i]= ts_i
        else:
            new_cols[col_i]=df[col_i]
    final = pd.DataFrame(new_cols)
    print(final.to_latex())

def by_voting(in_path):
    df=pd.read_csv(in_path)
    voting_type=['borda','opv_acc','opv_auc','opv_f1','raw']
    df_dict={vote_i:df[df['Voting']==vote_i]
               for vote_i in voting_type}
    print(df_dict['borda']['Dataset'].unique())

def dataset_stats(in_path):
    paths=format.find_result(in_path,"common")
    dataset_dict={}
    for path_i in paths:
        dataset_dict[path_i.split("/")[-3]]=path_i
    raw=[]
    for name_i,path_i in dataset_dict.items():
        data=feats.read(path_i)[0]
        raw.append([name_i,len(data),data.dim()[0],data.n_cats()])
    cols=['Dataset','Samples','Feats','Cats' ]
    return pd.DataFrame(raw,columns=cols)

cols=["Dataset","sucess","neutral","fail"]

#"old_mean","old_std","old_max","new_mean",,"new_max"]
#as_latex("knn/succ.csv",cols)
#by_voting('full.csv')
df=dataset_stats("data")
print(df.to_csv())