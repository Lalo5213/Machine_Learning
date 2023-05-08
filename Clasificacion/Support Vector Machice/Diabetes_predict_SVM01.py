# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 08:35:29 2023
the target of this program to do is make to algorithm to predict if a pacient have diabetic 
of not have diabetic implementing  Support Vector Machine
@author: Eduardo Castillo Garcia 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Data collection

#Loading the diabetes dataset
df = pd.read_csv('D:/MachineLearnig/diabetes1.csv')


#Getting the statistical measures of the data
s=df.describe()
#c

NoD=df['Outcome'].value_counts()
#0 .----> No Diabetics
#1 .----> Diabetics

mean=df.groupby('Outcome').mean()

#Separating the Data Labels

X= df.drop(columns='Outcome',axis=1)
Y=df['Outcome']

#Data Standarization (range 0 to 1)
scaler=StandardScaler()
#Fit to data, then transform it.
scaler.fit(X)
stand_Data=scaler.transform(X)

X=stand_Data

#Train test split
#test_size=0.2 represents to 20% the Data Data_test
#stratify=Y  If not None, data is split in a stratified fashion, using this as the class labels. 
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
#Proportion of the data
print(X.shape,X_train.shape,X_test.shape)

#Training the Model

classifier=svm.SVC(kernel='linear')

#Training the support vector Machine Classifier

classifier.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score on the training Data
X_train_prediction=classifier.predict(X_train)
training_data_acurracy=accuracy_score(X_train_prediction, Y_train)

#Accuracy Score on the test Data
X_test_prediction=classifier.predict(X_test)
test_data_acurracy=accuracy_score(X_test_prediction, Y_test)


print('Accuracy score of the training Data: ', training_data_acurracy)
print('Accuracy score of the test Data: ', test_data_acurracy)

#We to see the model is not overfit

#Making Predictive system
input_data=(5,116,74,0,0,25.6,0.201,30)

#Transform the input_data to Numpy array
input_data_np=np.asarray(input_data)

#reshape the array predicting
input_data_np_rs=input_data_np.reshape(1,-1)
#We need to standarized teh Data
std_data=scaler.transform(input_data_np_rs)

prediction= classifier.predict(std_data)
#Remeber if de data predicction
#0 .----> No Diabetics
#1 .----> Diabetics
if (prediction ==0):
    print('The pacient not have diabetic ')
else:
    print('The pacient have diabetic ')
