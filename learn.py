import numpy as np
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.metrics import roc_auc_score
import data_dict,feats,clf,files

class Result(data_dict.DataDict):
    def get_pred(self):
        train,test=self.split()
        names=test.names()
        y_true=names.get_cats()
        y_pred=[np.argmax(self[name_i]) for name_i in names]
        return y_true,y_pred,names

    def as_array(self,names=None):
        if(names is None):
            names=self.names()
        return np.array([self[name_i] for name_i in names]) 
    
    def true_one_hot(self):
        names=self.names()
        y_true=names.get_cats()
        n_cats=names.n_cats()
        return to_one_hot(y_true,n_cats)

    def pred_one_hot(self):
        names=self.names()
        y_pred=[np.argmax(self[name_i]) for name_i in names]
        n_cats=names.n_cats()
        return to_one_hot(y_pred,n_cats)

    def dim(self):
        return list(self.values())[0].shape

    def get_acc(self):
        y_true,y_pred,names=self.get_pred()
        return accuracy_score(y_true,y_pred)

    def get_auc(self):
        train,test=self.split()
        y_true=test.true_one_hot()
        y_pred=test.pred_one_hot()
        return roc_auc_score(y_true,y_pred,multi_class="ovo")
    
    def get_f1(self):
        y_true,y_pred,names=self.get_pred()
        return f1_score(y_true,y_pred,average='macro')

    def report(self):
        y_true,y_pred,names=self.get_pred()
        print(classification_report(y_true, y_pred,digits=4))

    def metrics(self):
        y_true,y_pred,names=self.get_pred()
        return precision_recall_fscore_support(y_true,y_pred,average='weighted')

    def add_column(self):
        for name_i,value_i in self.items():
            vect_i=self[name_i]
            self[name_i]=np.insert(value_i,value_i.shape[0],0,axis=0)
    
    def save(self,out_path):
        files.save(out_path,self)

def make_result(y_pred,names):
    if(type(names)!=files.NameList):
        names=files.NameList(names)
    if(type(y_pred)!=np.ndarray):
        y_pred=np.array(y_pred)
    if(len(y_pred.shape)==1):
        n_cats=names.n_cats()#np.amax(y_pred)+1
        y_pred=to_one_hot(y_pred,n_cats)
    result=[(name_i,pred_i) 
            for name_i,pred_i in zip(names,y_pred)]
    return Result(result)    

def train_model(data,clf_type="LR",selector=None,
                model_only=False):
    if(type(data)==str or type(data)==list):    
        data=feats.read(data)[0]
    data.norm()
    print(data.dim())
    print(len(data))
    train,test=data.split(selector)
    model=make_model(train,clf_type)
    if(model_only):
        return model
    names=data.names()
    X_test,y_true=data.get_X(),data.get_labels()
    y_pred=model.predict_proba(X_test)
    return make_result(y_pred,names)

def make_model(train,clf_type):
    if(type(clf_type)==str):
        model= clf.get_cls(clf_type)
    else:
        model=clf_type
    X_train,y_train= train.get_X(),train.get_labels()
    model.fit(X_train,y_train)
    return model

def to_one_hot(y,n_cats):
    one_hot=[]
    for y_i in y:
        one_hot.append(np.zeros((n_cats,)))
        one_hot[-1][y_i]=1.0
    return np.array(one_hot)

if __name__ == "__main__":
    in_path="../ml_utils/gen/B/common/wine"#dermatology"
    result= train_model(in_path)
    print(result.get_acc())
