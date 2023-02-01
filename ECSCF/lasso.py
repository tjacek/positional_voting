from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
import pandas as pd

class ExpStats(object):
    def __init__(self,results,stats):
        self.results=results
        self.cols=list(stats.columns)[1:]
        self.X=stats.to_numpy()[:,1:]
        self.X=preprocessing.scale(self.X)

    def __call__(self):
        base=('ECSCF','LR')
        diff=[('common','RF'),('binary','RF'),('ECSCF','RF'),
              ('common','LR'),('binary','LR')]
        new_cols=['base','diff']+self.cols
        raw_dict={col_i:[] for col_i in new_cols} 
        for diff_i in diff:
            raw_dict['base'].append(f'{base[0]}({base[1]})')
            raw_dict['diff'].append(f'{diff_i[0]}({diff_i[1]})')
            y_i=self.get_y(base,diff_i)
            coff_i=self.lasso_coff(y_i)
            for col_j,value_j in zip(self.cols,coff_i):
                raw_dict[col_j].append(f'{value_j:.4}')
        df=pd.DataFrame.from_dict(raw_dict)
        return df
#        print(df)

    def get_y(self,base=('ECSCF','LR'),
    	      diff=('ECSCF','RF')):
        base_df=self.results[(self.results['Ens']==base[0]) &
                             (self.results['Clf']==base[1])]
        diff_df=self.results[(self.results['Ens']==diff[0]) &
                             (self.results['Clf']==diff[1])]
        y=base_df['Acc'].to_numpy()-diff_df['Acc'].to_numpy()
        return y

    def lasso_coff(self,y):
        clf = linear_model.LinearRegression()#Lasso(alpha=0.001)
        clf.fit(self.X,y)
        return clf.coef_/np.sum(np.abs(clf.coef_))

def prepare(in_path,stats_path):
    result_df=pd.read_csv(in_path)
    stats_df=pd.read_csv(stats_path)
    return ExpStats(result_df,stats_df)

es=prepare('result.txt','stats.csv')
df=es()
print(df.to_latex())