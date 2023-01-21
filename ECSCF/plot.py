import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def prepare_data(in_path,stats_path):
    result_df=pd.read_csv(in_path)
    stats_df=pd.read_csv(stats_path)
    ecscf= result_df[result_df['clf']=='ECSCF']['acc']
    rf= result_df[result_df['clf']=='RF']['acc']
    clf_df=pd.DataFrame({
        'Dataset': stats_df['Dataset'],#.reset_index(drop=True),
    	'RF':rf.reset_index(drop=True), 
    	'ECSCF':ecscf.reset_index(drop=True)})
    final_df=pd.merge(stats_df,clf_df,on='Dataset',how = 'inner')
    return final_df


def scatter_plot(df,col='Classes',name='title'):
    diff= 100*(df['ECSCF']-df['RF']).to_numpy()
    ind_var=df[col].to_numpy()
#    if(np.amax(ind_var)>10):
#        ind_var=(10/np.amax(ind_var))*ind_var
    y=df['Dataset']

    print(diff)
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
    plt.ylabel("NECSCF-RF accuracy [%]")
    plt.xlabel(f"{col}")    
    plt.title(f'NECSCF-RF accuracy as function of {col}')
    plt.show()

def get_limit(series):
    s_max=np.amax(series)
    s_min=np.amin(series)
    if(s_min>0):
        s_min=0
    else:
        s_min-=1
    return [s_min,s_max+2]

df=prepare_data('uci.csv','stats.csv')
scatter_plot(df,col='Samples')