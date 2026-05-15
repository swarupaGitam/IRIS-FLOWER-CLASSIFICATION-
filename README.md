import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# %matplotlib inline

#loading dataset

columns = ['sepal_length','sepal_width','petal_length','petal_width','species']
#load the data
df = pd.read_csv('iris.csv',names=columns, skiprows=1)
# Convert numerical columns to float, coercing errors
for col in columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.head()

#Visualize the dataset

df.describe()

sns.pairplot(df,hue='species')

#Separate input and output

data = df.values
X = data[:,0:4]
Y = data[:,4]

#Split data to train and to test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#Model 1: Support vector machine algorithm

from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train,y_train)

prediction1 = model_svc.predict(X_test)

#Calculate accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction1)*100)

#Split data to train and to test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.metrics import confusion_matrix, classification_report, f1_score

cm1 = confusion_matrix(y_test, prediction1, labels=model_svc.classes_)
print("SVM Confusion Matrix:\n", cm1)

sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=model_svc.classes_, yticklabels=model_svc.classes_)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("SVM Classification Report:\n", classification_report(y_test, prediction1))
print("SVM F1 Score (Macro):", f1_score(y_test, prediction1, average='macro'))
print("SVM F1 Score (Weighted):", f1_score(y_test, prediction1, average='weighted'))

# Model 2: Logistic Regression

from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()
model_LR.fit(X_train,y_train)

prediction2 = model_LR.predict(X_test)

#Calculate accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction2)*100)

cm2 = confusion_matrix(y_test, prediction2, labels=model_LR.classes_)
print("Logistic Regression Confusion Matrix:\n", cm2)

sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens',
            xticklabels=model_LR.classes_, yticklabels=model_LR.classes_)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Logistic Regression Classification Report:\n", classification_report(y_test, prediction2))
print("Logistic Regression F1 Score (Macro):", f1_score(y_test, prediction2, average='macro'))
print("Logistic Regression F1 Score (Weighted):", f1_score(y_test, prediction2, average='weighted'))

# Model 3: Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train,y_train)

prediction3 = model_DTC.predict(X_test)

#Calculate accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction3)*100)

cm3 = confusion_matrix(y_test, prediction3, labels=model_DTC.classes_)
print("Decision Tree Confusion Matrix:\n", cm3)

sns.heatmap(cm3, annot=True, fmt='d', cmap='Oranges',
            xticklabels=model_DTC.classes_, yticklabels=model_DTC.classes_)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Decision Tree Classification Report:\n", classification_report(y_test, prediction3))
print("Decision Tree F1 Score (Macro):", f1_score(y_test, prediction3, average='macro'))
print("Decision Tree F1 Score (Weighted):", f1_score(y_test, prediction3, average='weighted'))

# Detailed Classification report

from sklearn.metrics import classification_report
print(classification_report(y_test,prediction2))

X_new = np.array([[3,2,1,0.2],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])

#Prediction of species from the input vector
prediction = model_svc.predict(X_new)
print("Prediction of Species: {}".format(prediction))
