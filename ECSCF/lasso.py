from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
import pandas as pd

class ExpStats(object):
    def __init__(self,results,stats):
        self.results=results
        self.cols=stats.columns
        self.X=stats.to_numpy()[:,1:]
        self.X=preprocessing.scale(self.X)

    def __call__(self):
        y=self.get_y()
        coff=self.lasso_coff(y)
        print(self.cols)
        print(coff)

    def get_y(self,base=('ECSCF','LR'),
    	      diff=('ECSCF','RF')):
        base_df=self.results[(self.results['Ens']==base[0]) &
                             (self.results['Clf']==base[1])]
        diff_df=self.results[(self.results['Ens']==diff[0]) &
                             (self.results['Clf']==diff[1])]
        y=base_df['Acc'].to_numpy()-diff_df['Acc'].to_numpy()
        return y

    def lasso_coff(self,y):
        clf = linear_model.Lasso(alpha=0.001)
        clf.fit(self.X,y)
        return clf.coef_/np.sum(np.abs(clf.coef_))

def prepare(in_path,stats_path):
    result_df=pd.read_csv(in_path)
    stats_df=pd.read_csv(stats_path)
    return ExpStats(result_df,stats_df)

es=prepare('result.txt','stats.csv')
es()