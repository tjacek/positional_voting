import numpy as np
import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
from sklearn.base import BaseEstimator, ClassifierMixin
import protocols,data,clfs

class BinaryEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self,n_hidden=25):
        self.n_hidden=n_hidden
        self.models=[]

    def fit(self,data_i,targets):
        params={'dims':data_i.shape[1],'n_cats':2}#data_i.shape[1]}
        for cat_i in range(params['n_cats']):
            model_i=SimpleNN(n_hidden=self.n_hidden)(params)
            self.models.append(model_i)
        return self

    def predict(self,X):
        y=[]
        for model_i in self.models:
            y_i=model_i.predict(X)
            y.append(y_i)
        y=np.array(y)
        target=np.sum(y,axis=0)
        return np.argmax(target,axis=1)


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

def binary_clf():
    clf = BinaryEnsemble   
    params={'n_hidden': [25,50,100]}
    return clfs.ClfAlg(clf,params,"binary")

d=data.read_data("wine.json")
#d=d.subsample(100)
#print(len(d))
train=d.split()[0]
#clf_alg=BinaryEnsemble()
clf_alg=binary_clf()
#X,y,names=train.as_dataset()
#clf_alg.fit(X,y)
#y=clf_alg.predict(X)
#print(y)
protocols.find_hyperparams(train,clf_alg)