from sklearn.metrics import classification_report,accuracy_score,f1_score
import data

class Result(data.DataDict):
    def get_acc(self):
        y_pred,y_true,names=self.as_dataset()
        return accuracy_score(y_pred,y_true)

    def report(self):
        y_pred,y_true,names=self.as_dataset()
        print(classification_report(y_true, y_pred,digits=4))
	

def make_result(names,y_pred):
    result=[(name_i,pred_i) 
            for name_i,pred_i in zip(names,y_pred)]
    return Result(result)  