import numpy as np
import pandas

def by_voting(paths):
    dataset=[pandas.read_csv(path_i)
                for path_i in paths]
    vote_dicts={}
    key_names=['dataset','clf']
    for data_i in dataset:
        for vote_j in data_i.voting.unique():
            data_ij=data_i[data_i['voting']==vote_j]	
            vote_dicts[vote_j]=to_dict(data_ij,key_names)
    return vote_dicts

def to_dict(data,key_names):
    data = data.reset_index()  
    data_dict={}
    for index, row in data.iterrows():
        key_i="_".join([row[name_i] 
            for name_i in key_names])
        data_dict[key_i]=row
    return data_dict

def compare_output(pair,attr,vote_dicts):
    old,new=vote_dicts[pair[0]],vote_dicts[pair[1]]
    all_diff=[]
    for name_i in old.keys():
    	old_attr=old[name_i][attr]
    	new_attr=new[name_i][attr]
    	diff= new_attr-old_attr
    	print(f"{name_i},{diff/old_attr}")
    	all_diff.append( diff)
    print(np.median(all_diff))

def to_doc(in_path,out_path):
    from docx import Document
    exp_output = pandas.read_csv(in_path)
    col_names= exp_output.columns
    document = Document()
    dataset_dict={ name_i:[]
       for name_i in exp_output.dataset.unique()}
    for i, row_i in exp_output.iterrows():
        dataset_dict[row_i['dataset']].append(row_i)
    for name_i,lines_i in dataset_dict.items():
        table_i=document.add_table(rows=len(lines_i)+1,
            cols=len(col_names))
        cells = table_i.rows[0].cells
        for i,name_i in enumerate(col_names):
            cells[i].text=name_i
        for line_j in lines_i:
            print(line_j[0])
        document.add_paragraph()
    print(col_names)
    document.save(out_path)

to_doc('bayes.csv','test.doc')
#vote_dicts=by_voting(['bayes.csv','auc2.csv'])
#print(vote_dicts.keys())
#compare_output(['raw','opv_auc2'],'auc_mean',vote_dicts)