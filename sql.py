import pandas as pd
import exp,feats

def by_voting(in_path):
    df=pd.read_csv(in_path)
    voting_type=['borda','opv_acc','opv_auc','opv_f1','raw']
    df_dict={vote_i:df[df['Voting']==vote_i]
               for vote_i in voting_type}
    print(df_dict)


def dataset_stats(in_path):
    @exp.dir_function()
    def helper(in_path):
	    name=in_path.split("/")[-1]
	    data=feats.read(in_path)[0]
	    return [name,len(data),data.dim()[0],data.n_cats()]
    raw=helper(in_path)
    cols=['Dataset','Samples','Feats','Cats' ]
    return pd.DataFrame(raw,columns=cols)

#by_voting('full.csv')
df=dataset_stats("A/datasets")
print(df)