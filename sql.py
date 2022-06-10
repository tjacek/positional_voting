import pandas as pd

def by_voting(in_path):
    df=pd.read_csv(in_path)
    voting_type=['borda','opv_acc','opv_auc','opv_f1','raw']
    df_dict={vote_i:df[df['Voting']==vote_i]
               for vote_i in voting_type}
    print(df_dict)

by_voting('full.csv')

#print(df[df['Voting']=='opv_acc'])