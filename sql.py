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
    print(final.to_csv())

def by_voting(in_path):
    df=pd.read_csv(in_path)
    ens_desc=value_dict(df,['Dataset', 'Clf', 'Voting'])
    rows=[]
    for data_i in ens_desc['Dataset']:
        for clf_j in ens_desc['Clf']:
            cond_i=f"Dataset=='{data_i}' and Clf=='{clf_j}' "
            df_i=df.query(cond_i)
            base_acc= df_i[ df_i['Voting']=='raw']['Acc_mean'].to_list()[0]
            pairs=zip(df_i['Voting'].to_list(), df_i['Acc_mean'].to_list())
            row_ij=[]
            for name,acc in pairs:
                if(name=='raw'):
                    row_ij.append(acc)
                else:
                    row_ij.append(acc-base_acc)
            rows.append([data_i,clf_j]+row_ij)
    new_cols=['Dataset', 'Clf']+ ens_desc['Voting']
    return pd.DataFrame(rows,columns=new_cols)

def value_dict(df,cols):
    return { name_i:list(df[name_i].unique()) 
               for name_i in cols }

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

def attr_corel(paths,pairs):
    frame=[pd.read_csv(path_i) for path_i in paths]
    final=frame[0].merge(frame[1],left_on='Dataset', right_on='Dataset')
    main_attr,attrs=pairs
    main_ts=final[main_attr]
    raw=[]
    for attr_i in attrs:
        corl_i=main_ts.corr(final[attr_i])
        raw.append([main_attr,attr_i, corl_i ])
    cols=['Attr X','Attr Y','Corl' ]
    return pd.DataFrame(raw,columns=cols)

#cols=["Dataset","sucess","neutral","fail"]
#cols=["Dataset",("new_mean","old_mean")]
#as_latex("knn/stats.csv",cols)
#df=dataset_stats("data")
df=by_voting("full.csv" )
#pairs=("diff_new_mean",["Samples","Feats","Cats"])
#df=attr_corel(["knn/datasets.csv","knn/diff.csv"],pairs)
print(df.to_latex())