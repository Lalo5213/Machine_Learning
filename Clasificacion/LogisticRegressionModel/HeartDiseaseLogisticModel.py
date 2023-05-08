# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:37:40 2023

Heart Disease predcition with python 
@author: Eduardo Castillo Garcia 

Heart Data ---> Data Processing ---> Train Test split
---> Logistic Regresion Model


We use This model because in this issue is a binary classification 
either yes or no.

We are going to classify wheter a person has a deceased or no 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#DATA COLLECTION AND PROCESSING
df=pd.read_csv('D:/MachineLearnig/Heart.csv')


#getting info about the data set
info_data=df.info()

#Checking for missing values
missing_values=df.isnull().sum()

#stadistical measures about the data

smd=df.describe()

#checking the distribution of target variarble
tar=df['target'].value_counts()
# 1 -----> Defective Heart
# 0 -----> Healthy Heart


#Splitting the target 

X=df.drop(columns='target',axis=1)
Y=df['target']

#Splitting the Data into Training data & Test Data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,stratify=Y,random_state=2)
'''
 stratify nos permite generar los bloques de entrenamiento y pruebas preservando en ambos el porcentaje de las muestras del dataset original.
 Separa las clase 1 y 0
'''
print('The distribution of the Data:',X.shape,X_train.shape,X_test.shape)

#MODEL TRAINING
model=LogisticRegression()
#Training logistic Regresion
model.fit(X_train, Y_train)

#Model Evaluation

#Accuracy score of training Data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy of the training data is: ', training_data_accuracy)

#Accuracy score of test Data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy of the test data is: ', test_data_accuracy)

"""  PREDICTED SYSTEM """
input_data=df.sample(n=1).drop(columns='target',axis=1)

prediction=model.predict(input_data)

if prediction==1:
    print('You have defeactive heart')
else:
    print('You have health hearth')
    

