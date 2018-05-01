import pandas as pd
import csv
import glob
import tables
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# creates a list of all csv files
#mypath = "\\\\egr-1l11qd2\\CLS_lab\\Junya Zhao\\GWAS SNPs_2018\\6439_SNPs_file\\"
globbed_files = glob.glob("*.csv")
for csv in globbed_files:
    frame = pd.read_csv(csv)
    del frame['Unnamed: 0']
    data = frame[["allele1","allele2"]]
    X=data.values
    target = frame[["label"]]
    y=target.values
    kf = KFold(n_splits=20)
    a = kf.get_n_splits(X)
    accoo = []
    for train_index, test_index in kf.split(X):    
       # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3,random_state= 0)    
       # print("==================train set ==================================")
        #print(X_train)
        #print("==================test set ==================================")
       #print(X_test)
        #print("==================result ==================================")
        mlp = MLPClassifier(hidden_layer_sizes=(10))
        result = mlp.fit(X_train,y_train)
        predictions = mlp.predict(X_test)
        #print(predictions)
        acc = mlp.score(X_test, y_test, sample_weight=None)
        acc2 = accuracy_score(y_test,predictions, sample_weight=None)
        #print(confusion_matrix(y_test,predictions))
        #print("==================accuracy ==================================")
       #print("The mean accuracy on the given test data and labels:", acc)
        #print("The accuracy on the ", acc2)
        #print("============================================================")
        #print(classification_report(y_test,predictions))
        accoo.append(acc)
    mean_accuracy  = (sum(accoo) / float(len(accoo)))*100
    print(csv,"KFold_accuracy is ",mean_accuracy,"%")
