import numpy as np
import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import protocols,data,clfs,cv,opv

class BinaryEnsemble(BaseEstimator, ClassifierMixin):
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
            model_i=SimpleNN(n_hidden=self.n_hidden)(params)
            model_i.fit(X,y_i,epochs=self.n_epochs,batch_size=self.batch_size)
            extractor_i=Model(inputs=model_i.input,
                outputs=model_i.get_layer('hidden').output)  
            self.extractors.append(extractor_i)
        return self.extractors        

    def predict(self,X):
        y=[]
        for model_i in self.estimators_:
#            raise Exception(X.shape)
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

def binarize(cat_i,targets):
    y_i=np.zeros((len(targets),2))
    for j,target_j in enumerate(targets):
        y_i[j][int(target_j==cat_i)]=1
    return y_i

class SimpleNN(object):
    def __init__(self,n_hidden=10):
        self.n_hidden=n_hidden
        self.optim=optimizers.RMSprop(learning_rate=0.00001)

    def __call__(self,params):
        model = Sequential()
        model.add(Dense(self.n_hidden, input_dim=params['dims'], activation='relu',name="hidden",
            kernel_regularizer=regularizers.l1(0.001)))
        model.add(BatchNormalization())
        model.add(Dense(params['n_cats'], activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=self.optim, metrics=['accuracy'])
        model.summary()
        return model

class MulticlassFacade(object):
    def __init__(self,raw_clf,extractor):
        self.raw_clf=raw_clf
        self.extractor=extractor

    def predict_proba(self,X):
#        raise Exception(X.shape)
        binary_i=self.extractor.predict(X)
        concat_i=np.concatenate([X,binary_i],axis=1)
        return self.raw_clf.predict_proba(concat_i)

def binary_clf():
    clf = BinaryEnsemble   
    params={'n_hidden': [25,50,100],
            'n_epochs': [100,200,300]}
    return clfs.ClfAlg(clf,params,"binary")

def test_bf(d):
    clf_alg=BinaryEnsemble()
    train,test=d.split()
    X,y,names=train.as_dataset()
    clf_alg.fit(X,y)
    X,y,names=test.as_dataset()
    y=clf_alg.predict(X)
    print(y)

def test_cv(d):
    train=d.split()[0]
    clf_alg=binary_clf()
    best_params=protocols.find_hyperparams(train,clf_alg)
    print(best_params)

def binary_protocol():
    selector=cv.BalancedSelector(0,4)
    opv_exp=protocols.OPVExp(limit=12,selector=selector)
    clf_algs=[binary_clf()]#clfs.rf_clf(),clfs.bag_clf()]
    metrics=[opv.acc_metric]#,opv.auc_metric,opv.f1_metric]
    return protocols.Protocol(clf_algs,metrics,opv_exp=opv_exp)

#d=data.read_data("wine.json")
protocol= binary_protocol()
protocol("wine.json",'wine_',n_iters=2)
#test_cv(d)


#d=d.subsample(100)
#print(len(d))
