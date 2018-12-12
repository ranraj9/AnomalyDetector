import hdf5storage
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,classification_report,roc_curve,auc


def separate(X,y):
    data=pd.concat([X,y],axis=1)
    data.columns=[0,1,2,3]
    normal=data.loc[data[3]==0]
    abnormal=data.loc[data[3]==1]
    return normal,abnormal

def prepare_sets(X,y):
    X=np.array(X)
    y=np.array(y)
    X,X_test,y,y_test=train_test_split(X,y,test_size=0.2)
    X_train,X_dev,y_train,y_dev=train_test_split(X,y,test_size=0.25)
    return X_train,y_train,X_dev,y_dev,X_test,y_test

def model(X):
    ifor = IsolationForest(n_estimators=100, max_samples=256,contamination=0.0235)
    ifor.fit(X)
    return ifor

def convert(y):
    for i in range(y.shape[0]):
        if y[i]==-1:
            y[i]=1;
        else:
            y[i]=0
    return y


def model_evaluate(ifor,X,y):
    y_pred=ifor.predict(X)
    y_pred=convert(y_pred)
    s= ifor.decision_function(X)
    z=roc_auc_score(y,-s)
    print('Report= ',classification_report(y,y_pred))
    print('AUCROC= ',z)
    print('F1= ',f1_score(y,y_pred))
    print('Confusion matrix= ',confusion_matrix(y,y_pred))


def roc_plot(model,X,y):
    preds=-model.decision_function(X)
    fpr, tpr, threshold = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
if __name__=='__main__':
    file='http.mat'
    mat = hdf5storage.loadmat(file)
    X=pd.DataFrame(mat['X'])
    y=pd.DataFrame(mat['y'])
    
    normal,abnormal=separate(X,y)
    X_train,y_train,X_dev,y_dev,X_test,y_test=prepare_sets(X,y)
    
    
    ifor=model(X_train)
    
    scores_n=ifor.decision_function(normal)
    scores_ab=ifor.decision_function(abnormal)
    
    for i in range(1500):
        plt.scatter(scores_n[i],0,color='blue',marker='*')
    for i in range(1500):
        plt.scatter(scores_ab[i],1,color='red',marker='*')
    
    print('TRAIN')
    model_evaluate(ifor,X_train,y_train)
    
    print('DEV')
    model_evaluate(ifor,X_dev,y_dev)
    
    print('TEST')
    model_evaluate(ifor,X_test,y_test)
    

    