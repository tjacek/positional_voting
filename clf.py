from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def get_cls(clf_type):
    if(clf_type=="SVC"):
        print("SVC")
        return make_SVC()
    elif(clf_type=="SVC_simple"):
        return SVC(probability=True)
    elif(clf_type=="RF"):
        print("RF")
        return RandomForestClassifier(max_depth=None, random_state=0)
    elif(clf_type=="Tree"):
        print(clf_type)
        return DecisionTreeClassifier(criterion='entropy',random_state=0)
    elif(clf_type=="bag"):
        print("bag")
        base_clf=get_cls("LR")
        return BaggingClassifier(base_estimator=base_clf,n_estimators=10)
    else:
        print("LR")
        return LogisticRegression(solver='liblinear')

def make_SVC():
    params=[{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 50,110, 1000]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]    
    clf = GridSearchCV(SVC(C=1,probability=True),params, cv=5,scoring='accuracy')
    return clf