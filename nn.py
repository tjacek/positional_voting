import sys
sys.path.append("..")
import tensorflow.keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import Input, Model
import os.path
import exp,feats,files,learn

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

def simple_ensemble(dataset,out_path,
    n_hidden=30,batch_size=1,n_epochs=5):
    print(len(dataset))
    train,test=dataset.split()
    X,y,names= train.as_dataset()
    n_cats=names.n_cats()
    params={'dims':train.dim()[0],'n_cats':2}
    models,accuracy=[],[]
    for cat_i in range(n_cats):
        model_i=SimpleNN(n_hidden=n_hidden)(params)
        y_i=names.binarize(cat_i)
        y_i=learn.to_one_hot(y_i,2)
        history = model_i.fit(X,y_i,
            epochs=n_epochs,batch_size=batch_size)
        models.append(model_i)
        acc_i=history.history['accuracy'][-1]
        accuracy.append(acc_i)
    files.make_dir(out_path)
    for i,model_i in enumerate(models):
        extractor_i=Model(inputs=model_i.input,
                outputs=model_i.get_layer('hidden').output)        
        X,y,names=dataset.as_dataset()
        new_X=extractor_i.predict(X)
        feats_i=feats.Feats({name_j:new_X[j] 
                for j,name_j in enumerate( names)})
        out_i=f"{out_path}/{i}"
        print(out_i)
        feats_i.save(out_i)
    return accuracy

@exp.dir_function(recreate=False)
@exp.multi_iter(n_iters=10,in_iter=False)
def random_ensemble(in_path,out_path,n_epochs=10):
    print(in_path)
    print(out_path)
    if(not os.path.exists(out_path)):
        files.make_dir(out_path)
        data_i=feats.read(in_path)[0]
        random_data=data_i.balanced_split()
        random_data.save(f"{out_path}/common")
        n_hidden=30#min( (30,random_data.dim()[0]))   
        simple_ensemble(random_data,f"{out_path}/binary",
            n_hidden=n_hidden,batch_size=1,n_epochs=n_epochs)

if __name__ == "__main__":
#    dataset=convert.txt_dataset("penglung/raw.data")
    random_ensemble("A/common","A/one_vs_all")