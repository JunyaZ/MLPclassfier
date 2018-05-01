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
    loo = LeaveOneOut()
    a = loo.get_n_splits(X)
    mlp = MLPClassifier(hidden_layer_sizes=(50,20), learning_rate='constant',learning_rate_init=0.01)
    accuracy =[]
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        result = mlp.fit(X_train,y_train)
        predictions = mlp.predict(X_test)
        matrix = confusion_matrix(y_test,predictions)
        report = classification_report(y_test,predictions)
        acc = accuracy_score(y_test,predictions)
        accuracy.append(acc)
        with open(f"\\\\egr-1l11qd2\\CLS_lab\\Junya Zhao\\\GWAS SNPs_2018\\6439_SNPs_file\\leaveOneout_report\\{'leaveOneout_'+csv}.txt","a") as output:
            output.write(csv)
            output.write("The prediction label is ")
            output.write(str(predictions))
            output.write(str(matrix))
            output.write(report)
            output.write("Accuracy Score -> ")
            output.write(str(acc))
                
