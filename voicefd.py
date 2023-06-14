# -*- coding: utf-8 -*-
"""
**********Voice Recognition Data Analysis**********
**********@author: gulsen************
"""
""" *********************************************************************** """
""" 
Libraries
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


""" *********************************************************************** """
"""
Data Import
"""
data = pd.read_csv(r'C:\Users\LENOVO\Desktop\data_science\voice.csv')

""" *********************************************************************** """


"""
Data Preparation
"""

data.label= [1 if each == "female" else 0 for each in data.label] 

y= data.label.values

x_data= data.drop(["label"], axis=1)

"""
Normalization
"""

x= (x_data - np.min(x_data)) / (np.max(x_data)).values 

"""
Train and Test
"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

method_names=[]

method_scores=[]

"""
LOGISTIC REGRESSION
"""

from sklearn.linear_model import LogisticRegression
log_reg= LogisticRegression()
log_reg.fit(x_train,y_train)
print("Logistic Regression Classification Test Accuracy {}".format(log_reg.score(x_test,y_test)))
method_names.append("Logistic Regression")
method_scores.append(log_reg.score(x_test, y_test))

"""
CONFUSION MATRIX
"""

y_pred= log_reg.predict(x_test)
conf_mat= confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red",fmt=".0f", ax=ax)
plt.xlabel("Predict Values")
plt.ylabel("True Values")
plt.title("Logistic Regression Test Accuracy")
plt.show()

"""
KNN (K-Nearest Neigbour) CLASSIFICATION
"""

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
print("KNN (K-Nearest Neigbour) CLASSIFICATION Test Accuracy {}".format(knn.score(x_test,y_test)))
method_names.append("KNN CLASSIFICATION")
method_scores.append(knn.score(x_test,y_test))

y_pred= knn.predict(x_test)
conf_mat= confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red",fmt=".0f", ax=ax)
plt.xlabel("Predict Values")
plt.ylabel("True Values")
plt.title("KNN CLASSIFICATION Test Accuracy")
plt.show()

score_list=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,15), score_list)
plt.xlabel("N Values")
plt.ylabel("Scores")
plt.title("KNN SCORES")
plt.show()

"""
SVM (Support Vector Machine)
"""

from sklearn.svm import SVC
svm= SVC()
svm.fit(x_train,y_train)
print("SVM Test Accuracy {}".format(svm.score(x_test,y_test)))
method_names.append("SVM")
method_scores.append(svm.score(x_test,y_test))

y_pred= svm.predict(x_test)
conf_mat= confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red",fmt=".0f", ax=ax)
plt.xlabel("Predict Values")
plt.ylabel("True Values")
plt.title("SVM Test Accuracy")
plt.show()

"""
Naive Bayes
"""

from sklearn.naive_bayes import GaussianNB
naive_bayes= GaussianNB()
naive_bayes.fit(x_train,y_train)
print("Naive Bayes Test Accuracy {}".format(naive_bayes.score(x_test,y_test)))
method_names.append("Naive Bayes")
method_scores.append(naive_bayes.score(x_test,y_test))

y_pred= naive_bayes.predict(x_test)
conf_mat= confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red",fmt=".0f", ax=ax)
plt.xlabel("Predict Values")
plt.ylabel("True Values")
plt.title("Naive Bayes Test Accuracy")
plt.show()

"""
DESICION TREE
"""
from sklearn.tree import DecisionTreeClassifier
dec_tree= DecisionTreeClassifier()
dec_tree.fit(x_train,y_train)
print("Decision Tree Accuracy {}".format(dec_tree.score(x_test,y_test)))
method_names.append("Decision Tree")
method_scores.append(dec_tree.score(x_test,y_test))

y_pred= dec_tree.predict(x_test)
conf_mat= confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red",fmt=".0f", ax=ax)
plt.xlabel("Predict Values")
plt.ylabel("True Values")
plt.title("Decision Tree Test Accuracy")
plt.show()

"""
RANDOM FOREST
"""

from sklearn.ensemble import RandomForestClassifier
rand_forest= RandomForestClassifier(n_estimators=100)
rand_forest.fit(x_train,y_train)
print("Random Forest Accuracy {}".format(rand_forest.score(x_test,y_test)))
method_names.append("Decision Tree")
method_scores.append(rand_forest.score(x_test,y_test))

y_pred= rand_forest.predict(x_test)
conf_mat= confusion_matrix(y_test, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5, linecolor="red",fmt=".0f", ax=ax)
plt.xlabel("Predict Values")
plt.ylabel("True Values")
plt.title("Random Forest Test Accuracy")
plt.show()

"""
CONCLUSION
"""

plt.figure(figsize=(15,10))
plt.ylim([0.85,1])
plt.bar(method_names, method_scores, width=0.5)
plt.xlabel('Method Name')
plt.ylabel('Method Score')