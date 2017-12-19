import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import svm
import re
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from numpy import random
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt 
import pickle
### Set path to the data set
dataset_path = "/home/ubuntu/Desktop/datapredict/77_cancer_proteomes_CPTAC_itraq.csv"
clinical_info = "/home/ubuntu/Desktop/datapredict/clinical_data_breast_cancer.csv"
pam50_proteins = "/home/ubuntu/Desktop/datapredict/PAM50_proteins.csv"

#Load csv file
data = pd.read_csv(dataset_path,header=0,index_col=0)
clinical = pd.read_csv(clinical_info,header=0,index_col=0)## holds clinical information about each patient/sample
pam50 = pd.read_csv(pam50_proteins,header=0)

#Drop unused information columns
data.drop(['gene_symbol','gene_name'],axis=1,inplace=True)

#Change the protein data sample names to match the clinical data format
data.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)

#Transpose data to match the formal in clinical
data = data.transpose()

#Only choose data about PAM50 gene
data_p50 = data.loc[:,data.columns.isin(pam50['RefSeqProteinID'])]

#Drop clinical entries for samples not in our protein data set
clinical = clinical.loc[[x for x in clinical.index.tolist() if x in data_p50.index],:]

#Merge the data together
merged = data.merge(clinical,left_index=True,right_index=True)

#Choose proper columns of data, train the model and do prediction and get scores

#Collect training data
x_train1 = merged.loc[:,[x for x in merged.columns if bool(re.search("NP_|XP_",x)) == True]]
y_train1 = merged.loc[:,[x for x in merged.columns if bool(re.search("Tumor",x)) == True]]
y_train1.drop(['Tumor--T1 Coded'],axis=1,inplace=True)
y_train1 = y_train1.replace({'T1' : 1, 'T2' : 2, 'T3' : 3, 'T4' : 4})

#Clean the data
imputer = Imputer(missing_values='NaN', strategy='median', axis=1)
imputer = imputer.fit(x_train1)
x_train1 = imputer.transform(x_train1)
y_train1 = imputer.transform(y_train1)
y_train1 = y_train1.astype(int)

#Split training and predicting part
test_size = 0.9
seed = 1
X_train1, X_test1, Y_train1, Y_test1 = model_selection.train_test_split(x_train1, y_train1, test_size=test_size, random_state=seed)

#Train the model and print score
model1 = svm.SVC(kernel='linear')
model1.fit(X_train1, np.ravel(Y_train1))
result1 = model1.score(X_test1, Y_test1)
print 'result 1 is:', result1

model2 = svm.SVC(kernel='poly')
model2.fit(X_train1, np.ravel(Y_train1))
result2 = model2.score(X_test1, Y_test1)
print 'result 2 is:', result2

model3 = svm.SVC(kernel='sigmoid')
model3.fit(X_train1, np.ravel(Y_train1))
result3 = model3.score(X_test1, Y_test1)
print 'result 3 is:', result3

model4 = svm.SVC()
model4.fit(X_train1, np.ravel(Y_train1))
result4 = model4.score(X_test1, Y_test1)
print 'result 4 is:', result4

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train1, Y_train1)
print 'Naive Bayes result is:',clf.score(X_test1, Y_test1)

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier() 
knn.fit(X_train1, Y_train1)
print 'kNN result is:',knn.score(X_test1, Y_test1)

from sklearn import tree
model1 = tree.DecisionTreeClassifier()
model1.fit(X_train1, Y_train1)
print 'Decision Tree result is:'model1.score(X_test1, Y_test1)