# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset\iris.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.333, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression()
#classifierLR = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', random_state=0)
classifierLR.fit(X_train, y_train)

#Fitting Naive Bayes Regression to the Training Set
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifierDT.fit(X_train, y_train)

# Predicting the Test set results
y_predLR = classifierLR.predict(X_test)
y_predNB = classifierNB.predict(X_test)
y_predDT = classifierDT.predict(X_test)

# Printing Results
from sklearn.metrics import confusion_matrix
print("Linear Regression Confusion matrix" )
cmLR = confusion_matrix(y_test, y_predLR)
for i in range(3) :
    print("{:3d} {:3d} {:3d}".format(cmLR[i][0], cmLR[i][1], cmLR[i][2]))
accuracyLR = (cmLR[0][0] + cmLR[1][1] + cmLR[2][2])/len(y_test)
print("Accuracy: {0:6.4f} \n\n".format(accuracyLR))
cmNB = confusion_matrix(y_test, y_predNB)
print("Naive Bayes Confusion matrix" )
for i in range(3) :
    print("{:3d} {:3d} {:3d}".format(cmNB[i][0], cmNB[i][1], cmNB[i][2]))
accuracyNB = (cmNB[0][0] + cmNB[1][1] + cmNB[2][2])/len(y_test)
print("Accuracy: {0:6.4f}\n\n".format(accuracyNB))
cmDT = confusion_matrix(y_test, y_predDT)
print("Decision Tree Confusion matrix" )
for i in range(3) :
    print("{:3d} {:3d} {:3d}".format(cmDT[i][0], cmDT[i][1], cmDT[i][2]))

accuracyDT = (cmDT[0][0] + cmDT[1][1] + cmDT[2][2])/len(y_test)
print("Accuracy: {0:6.4f}\n\n".format(accuracyDT))

