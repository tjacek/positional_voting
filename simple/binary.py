from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
import protocols,data,clfs

class SimpleBinary(object):
    def __init__(self,n_hidden=25):
        self.n_hidden=n_hidden
        	
    def fit(self,data_i):
        return None


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
    clf = SimpleBinary   
    params={'n_hidden': [25,50,100]}
    return clfs.ClfAlg(clf,params,"binary")

d=data.read_data("cleveland.json")
train=d.split()[0]
clf_alg=binary_clf()
protocols.find_hyperparams(train,clf_alg)