import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import gini

def prepare_data(in_path,stats_path,comp):
    result_df=pd.read_csv(in_path)
    stats_df=pd.read_csv(stats_path)
    clf_df= get_clf_df(result_df,stats_df,comp)
    final_df=pd.merge(stats_df,clf_df,on='Dataset',how = 'inner')
    return final_df

def get_clf_df(result_df,stats_df,comp):
    alg_type,alg1,alg2=comp
    ecscf= result_df[result_df[alg_type]==alg1]['acc']
    rf= result_df[result_df[alg_type]==alg2]['acc']
    clf_df=pd.DataFrame({
        'Dataset': stats_df['Dataset'],
        alg1:ecscf.reset_index(drop=True), 
        alg2:rf.reset_index(drop=True)})
    return clf_df

def scatter_plot(df,col='Classes',algs=['ECSCF','RF'],name='title'):
    diff= 100*(df[algs[0]]-df[algs[1]]).to_numpy()
    ind_var=df[col].to_numpy()
    y=df['Dataset']

    plt.figure()
    ax = plt.subplot(111)
    for i,y_i in enumerate(y):    
        plt.text(ind_var[i], diff[i], y_i,
                   fontdict={'weight': 'bold', 'size': 9})
    
    plt.ylim(get_limit(diff)) 
    plt.xlim(get_limit(ind_var))
    
    start, stop = ax.get_ylim()
    ticks = np.arange(int(start), int(stop) + 1, 2)
    ax.set_yticks(ticks)


    start, stop = ax.get_xlim()
    print(stop)
    if(stop>12):
        delta= int((stop- start)/10)
    else:
        delta=1
    print(delta)
    ticks = np.arange(int(start), int(stop) + 3*delta, delta)
    ax.set_xticks(ticks)
    plt.grid()
    plt.ylabel(f"{algs[0]}-{algs[1]} accuracy [%]")
    plt.xlabel(f"{col}")    
    plt.title(f'{algs[0]}-{algs[1]} accuracy as function of {col}')
    plt.show()

def get_limit(series):
    s_max=np.amax(series)
    s_min=np.amin(series)
    if(s_min>0):
        s_min=0
    else:
        s_min-= 1 
    return [s_min,s_max+2]

def prepare_gini(in_path,imbalance_path):
    result_df=pd.read_csv(in_path)
    imb_df=pd.read_csv(imbalance_path)
    datasets=imb_df['Dataset'].unique()
    gini_df={}
    for data_i in datasets:
        row_i=imb_df[imb_df['Dataset']==data_i]
        row_i=row_i.to_numpy()[0]
        row_i=[float(c_j) for c_j in row_i
                if(is_number(c_j)) ]
#        row_i.sort()
        gini_df[data_i]=10* gini.gini_index(np.array(row_i))
    gini_df=pd.DataFrame.from_dict(gini_df.items())
    gini_df.columns= ['Dataset','gini index']
    clf_df= get_clf_df(result_df,gini_df)
    final_df=pd.merge(clf_df,gini_df,on='Dataset',how = 'inner')
    return final_df

def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True

def add_column(df):
    new_col=df['number of samples']/df['number of classes']
    df['samples per class']=new_col
#    print(new_col)

def exp(result_path,stats_path,imb_path):
    var=['classes','samples','features']
    var=[f'number of {var_i}' for var_i in var]
    df=prepare_data(result_path,stats_path)
    for var_i in var:
        scatter_plot(df,col=var_i)
    gini_df=prepare_gini(result_path,imb_path)
    scatter_plot(gini_df,col='gini index')

df= prepare_data('result.txt','stats.csv',['Ens','ECSCF(RF)','ECSCF(LR)'])
#print(df)
add_column(df)
#print(df)
scatter_plot(df,col='samples per class',algs=['ECSCF(LR)','ECSCF(RF)'])


#exp('uci.csv','stats.csv','imbalance.csv')
#print(gini_df)
