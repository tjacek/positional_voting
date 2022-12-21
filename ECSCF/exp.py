from sklearn.metrics import classification_report,accuracy_score,f1_score
import data,ecscf

d=data.read_data('wine.json')

clf_alg=ecscf.ECSCF()
train,test=d.split()
X,y,names=train.as_dataset()
clf_alg.fit(X,y)
X,y_true,names=test.as_dataset()
y_pred=clf_alg.predict(X)
print(accuracy_score(y_true,y_pred))
