import numpy as np
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
import data_dict,feats,clf

class Result(data_dict.DataDict):
    def get_pred(self):
        train,test=self.split()
        names=test.names()
        y_true=names.get_cats()
        y_pred=[np.argmax(self[name_i]) for name_i in names]
        return y_true,y_pred,names

    def as_array(self,names):
        return np.array([self[name_i] for name_i in names])
    
    def get_acc(self):
        y_true,y_pred,names=self.get_pred()
        return accuracy_score(y_true,y_pred)

    def report(self):
        y_true,y_pred,names=self.get_pred()
        print(classification_report(y_true, y_pred,digits=4))

    def metrics(self):
        y_true,y_pred,names=self.get_pred()
        return precision_recall_fscore_support(y_true,y_pred,average='weighted')

def make_result(y_pred,names):
    result=[ (name_i,pred_i) 
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
    model= clf.get_cls(clf_type)
    X_train,y_train= train.get_X(),train.get_labels()
    model.fit(X_train,y_train)
    return model

if __name__ == "__main__":
    in_path="../ml_utils/gen/B/common/wine"#dermatology"
    result= train_model(in_path)
    print(result.get_acc())
