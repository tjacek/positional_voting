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

prepare_data('uci.csv','stats.csv')