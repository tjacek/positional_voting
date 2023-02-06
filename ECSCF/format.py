import numpy as np
import pandas as pd

def find_best(in_path):
    df=pd.read_csv(in_path)
    datasets=df['Dataset'].unique()
    best=[]
    for data_i in datasets:
        df_i= df[df['Dataset']==data_i]
        best_i=df_i['Acc'].argmax()
        best_acc=df_i.iloc[best_i]['Acc']
        best_std=df_i.iloc[best_i]['Std']
        thres_i= best_acc-best_std
        best+= list(df_i['Acc']>thres_i)
    df['Best']=best
    return df

def format(df,out_path='out.csv'):
    ens_types=df['Ens'].unique()
    clf_types=df['Clf'].unique()
    rows={','.join([data_i,clf_j]):[] 
           for data_i in df['Dataset'].unique()
               for clf_j in clf_types}
    for clf_i in clf_types:
        for ens_j in ens_types:
            df_ij=df[ (df['Clf']==clf_i) & (df['Ens']==ens_j)]
            for k,row_k in df_ij.iterrows():
                row_k=row_k.to_dict()
                id_jk=','.join([row_k['Dataset'],clf_i])
                mean,std=100*row_k['Acc'],100*row_k['Std']
                best=row_k['Best']
                value=f'{mean:.4}±{std:.4}({best})'
                rows[id_jk].append(value)
    final_df=pd.DataFrame.from_dict(rows, orient='index',
    	columns=ens_types)	
    final_df=final_df[['common','binary','ECSCF']]
    print(final_df.to_latex())

def find_algs(in_path):
    df=pd.read_csv(in_path)
    algs=df.columns[1:]
    datasets=df['Dataset'].unique()
    for data_i in datasets:
        row_i=df[df['Dataset']==data_i]
        row_i=row_i.to_dict()
        del row_i['Dataset']
        row_i=[ list(row_i[alg_j].values())[0] 
                   for alg_j in algs]
        mean_i,std_i=[],[]
        for raw_j in row_i:
            mean_j,std_j=to_number(raw_j)
            mean_i.append(mean_j)
            std_i.append(std_j)
        print(mean_i)


def to_number(raw):
    mean,std= raw.split('±')
    return float(mean),float(std)

df=find_algs('wozniak.csv')
#format(df)