import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import utils,protocols,output

class PlotFactory(object):
    def __init__(self,result_dict,stats_df):
        self.result_dict=result_dict
        self.stats_df=stats_df

    def __call__(self,base=('ECSCF','LR'),diff=('ECSCF','RF'),
        x_name='gini index',title_name='Accuracy difference as function of gini',
        x_scale=10,y_scale=100):
        
        y= y_scale * (self.get_df(base)['Acc']-self.get_df(diff)['Acc'])
        x= x_scale * self.stats_df[x_name]
        names=self.stats_df['Dataset']
        y_label=f'{base[0]}({base[1]})-{diff[0]}({diff[1]})'
        scatter_plot(x,y,names,x_name,y_label,title_name)

    def get_df(self,base):
        df_id='_'.join(base)
        return self.result_dict[df_id]

class BoxPlot(object):
    def __init__(self,alg=None):
        if(alg is None):
            alg=output.get_escf_algs(['LR'])['LR']
        self.alg=alg

    def __call__(self,in_path):
        eff_fun=utils.dir_fun(as_dict=True)(self.alg)
        acc_dict=eff_fun(in_path)
        print(acc_dict)

def prepare_data(in_path,stats_path,ens,clf):
    result_df=pd.read_csv(in_path)
    result_dict= by_clf_type(result_df,ens,clf)
    stats_df=pd.read_csv(stats_path)
    return PlotFactory(result_dict,stats_df)

def by_clf_type(result_df,ens,clf):
    result_dict={}
    for ens_i in ens:
        ens_df=result_df[result_df['Ens']==ens_i]
        for clf_j in clf:
            id_ij='_'.join([ens_i,clf_j])
            df_ij=ens_df[result_df['Clf']==clf_j]
            df_ij= df_ij.set_index('Dataset')
            result_dict[id_ij]=df_ij
    return result_dict    

def scatter_plot(x,y,names,x_label,y_label,title_name):

    plt.figure()
    ax = plt.subplot(111)
    for i,name_i in enumerate(names):    
        plt.text(x[i], y[i], name_i,
                   fontdict={'weight': 'bold', 'size': 9})
    
    ax.set_ylim(get_limit(y)) 
    ax.set_xlim(get_limit(x))
    
    start, stop = ax.get_ylim()


    ticks = np.arange(int(start), int(stop) + 1, 1)
    ax.set_yticks(ticks)


    start, stop = ax.get_xlim()
    print(ax.get_xlim())
    if(stop>12):
        delta_x= int((stop- start)/10)
    else:
        delta_x=1
    print(delta_x)
    ticks = np.arange(int(start), int(stop) + 3*delta_x, delta_x)
    
    ax.set_xticks(ticks)
    plt.grid()
    plt.ylabel(y_label)
    plt.xlabel(x_label)   
    plt.title(title_name)
    plt.show()

def get_limit(series):
    s_max=np.amax(series)
    s_min=np.amin(series)
    if(s_min>0):
        s_min=0
    else:
        s_min-= 1 
    return [s_min,s_max+2]

def box_test():
    a=[1,2,3,4,5]
    b=[1,4,9,16,25]
    c=[1,16,81,256,625]
    fig, ax = plt.subplots()
    ax.boxplot([a,b,c])
    plt.show()

def add_column(df):
    new_col=df['number of samples']/df['number of classes']
    df['samples per class']=new_col

def to_latex(in_path):
    df=pd.read_csv(in_path)
    print(df.to_latex())

if __name__ == "__main__":
#    pf= prepare_data('result.txt','full_stats.csv',
#        ['ECSCF','common','binary'],['RF','LR'])
#    pf()
    box_plot=BoxPlot()
    box_plot('imb')