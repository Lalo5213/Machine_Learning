# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 07:14:16 2023
the target of this program to do is make to algorithm to predict if a customer is selected 
for a credit bank  implementing  Support Vector Machine
@author: Eduardo Castillo Garcia
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Data Collection
df=pd.read_csv('D:/MachineLearnig/train_u6lujuX_CVtuZ9i.csv')

#Statistical measures
mean_df=df.describe()

#Number fo missing values in each column
missin_values_df=df.isnull().sum()
#Dropping the missing values
dp_drop=df.dropna()
#Label Encoding
dp_drop.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)
#0 -----> Not obtain a credit
#1 -----> Obtain the credit
#Analyzing Dependents column value
depen=dp_drop['Dependents'].value_counts()
#Replacing the value of 3+ to 4
dp_drop=dp_drop.replace(to_replace='3+', value=4)

#DATA VISUALIZATION

#Education vs Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=dp_drop)

#Marital Status vs Loan Status
#sns.countplot(x='Married',hue='Loan_Status',data=dp_drop)

#Gender vs Loan Status
#sns.countplot(x='Gender',hue='Loan_Status',data=dp_drop)

gen=dp_drop['Gender'].value_counts()

#Convert categorical columns to numerical values
dp_drop.replace({"Married":{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Education':{'Not Graduate':0,'Graduate':1},'Self_Employed':{'No':0,'Yes':1},
                 'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)

#Separating the data and label
X=dp_drop.drop(columns=['Loan_ID','Loan_Status'])
Y=dp_drop['Loan_Status']
#Train Test Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

#TRAINING THE MODEL:SUPPORT VECTOR MODEL

classifier=svm.SVC(kernel='linear')

#Training the support Vector Machine
classifier.fit(X_train,Y_train)

#MODEL EVALUATION
#Accuracy score on the training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

#Accuracy score on the test data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

#MAKING A PREDICTIVE SYSTEM
input_data=dp_drop.sample(n=1).drop(columns=['Loan_ID','Loan_Status' ])

prediction= classifier.predict(input_data)
#Remeber if de data predicction
#0 .----> No credits aprove
#1 .----> Credit aprove
if (prediction ==0):
    print('The custemer not aprove the credit ')
else:
    print('The custumer aprove the credit ')