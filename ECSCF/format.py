import pandas as pd

def find_sign(in_path):
    df=pd.read_csv(in_path)
    datasets=df['Dataset'].unique()
    for data_i in datasets:
        df_i= df[df['Dataset']==data_i]
        best_i=df_i['acc'].argmax()
        best_acc=df_i.iloc[best_i]['acc']
        best_std=df_i.iloc[best_i]['Std']
        thres_i= best_acc-best_std
        best= df_i['acc']>thres_i
        print("*****")
        print(df_i[best])
#        print(df_i.index)

find_sign('result.txt')