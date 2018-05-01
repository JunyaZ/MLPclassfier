import pandas as pd
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
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        mlp = MLPClassifier(hidden_layer_sizes=(50,30))
        result = mlp.fit(X_train,y_train)
        predictions = mlp.predict(X_test)
        matrix = confusion_matrix(y_test,predictions)
        report = classification_report(y_test,predictions)
        acc = mlp.score(X_test, y_test, sample_weight=None)
        print(str(acc))
        with open(f"\\\\egr-1l11qd2\\CLS_lab\\Junya Zhao\\\GWAS SNPs_2018\\6439_SNPs_file\\report\\{'20_fold_'+csv}.txt","a") as output:
            output.write(csv)
            output.write("The prediction label is ")
            output.write(str(predictions))
            output.write(str(matrix))
            output.write(report)
            output.write("Accuracy Score -> ")
            output.write(str(acc))
