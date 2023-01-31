import pandas as pd

class ExpStats(object):
    def __init__(self,results,stats):
        self.results=results
        self.stats=stats

    def get_y(self,base=('ECSCF','LR'),
    	      diff=('ECSCF','RF')):
#    	      var='number of classes'):
        base_df=self.results[(self.results['Ens']==base[0]) &
                             (self.results['Clf']==base[1])]
        diff_df=self.results[(self.results['Ens']==diff[0]) &
                             (self.results['Clf']==diff[1])]
        y=base_df['Acc'].to_numpy()-diff_df['Acc'].to_numpy()
        return y

def prepare(in_path,stats_path):
    result_df=pd.read_csv(in_path)
    stats_df=pd.read_csv(stats_path)
    return ExpStats(result_df,stats_df)

es=prepare('result.txt','stats.csv')
es.get_y()