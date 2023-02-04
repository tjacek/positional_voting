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
    names=df.columns()


def to_number(raw)
    mean,std= raw.split('±')

df=find_best('result.txt')
format(df)