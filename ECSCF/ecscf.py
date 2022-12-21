import numpy as np 
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import nn

class ECSCF(BaseEstimator, ClassifierMixin):
    def __init__(self,n_hidden=25,n_epochs=100):
        self.n_hidden=n_hidden
        self.n_epochs=n_epochs
        self.batch_size=32
        self.extractors=[]
        self.estimators_=[]

    def fit(self,X,targets):
        self.make_extractor(X,targets)
        binary= self.binary_features(X)
        LR=LogisticRegression(solver='liblinear')
        for binary_i,extractor_i in zip(binary,self.extractors):
            clf_i=LR.fit(binary_i,targets)
            facade_i=MulticlassFacade(clf_i,extractor_i)
            self.estimators_.append(facade_i)
        return self

    def make_extractor(self,X,targets):
        n_cats=max(targets)+1
        for cat_i in range(n_cats):
            y_i=binarize(cat_i,targets)
            params={'dims':X.shape[1],'n_cats':2}
            model_i=nn.SimpleNN(n_hidden=self.n_hidden)(params)
            model_i.fit(X,y_i,epochs=self.n_epochs,batch_size=self.batch_size)
            extractor_i= nn.get_extractor(model_i)
#            extractor_i=Model(inputs=model_i.input,
#                outputs=model_i.get_layer('hidden').output)  
            self.extractors.append(extractor_i)
        return self.extractors        

    def predict(self,X):
        y=[]
        for model_i in self.estimators_:
            y_i=model_i.predict_proba(X)
            y.append(y_i)
        y=np.array(y)
        target=np.sum(y,axis=0)
        return np.argmax(target,axis=1)

    def binary_features(self,X):
        binary=[]
        for extractor_i in self.extractors:
            binary_i=extractor_i.predict(X)
            concat_i=np.concatenate([X,binary_i],axis=1)
            binary.append(concat_i)
        return binary

class MulticlassFacade(object):
    def __init__(self,raw_clf,extractor):
        self.raw_clf=raw_clf
        self.extractor=extractor

    def predict_proba(self,X):
        binary_i=self.extractor.predict(X)
        concat_i=np.concatenate([X,binary_i],axis=1)
        return self.raw_clf.predict_proba(concat_i)

def binarize(cat_i,targets):
    y_i=np.zeros((len(targets),2))
    for j,target_j in enumerate(targets):
        y_i[j][int(target_j==cat_i)]=1
    return y_i