# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:01:23 2023

@author: Eduardo Castillo Garcia 

Credit Card Data  -------> Data Pre Procesing --------> Data Analyst ----->
----> Train Test Split -----> Logistic Regresion
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#Load Dataset 
df= pd.read_csv('D:/MachineLearnig/creditcard.csv')

#Obtain information of the Dataset

df.info()

nullvalues=df.isnull().sum()
#Dont have missing values

#
values= df['Class'].value_counts()
 # 0 -----> Normal Transaction
 # 1 -----> Fraudulent Transaction
 
#Separating Data for Analysis
legit =df[df.Class==0]
fraud=df[df.Class==1]

#Stadistical measures of the Data

stmd_legit=legit.Amount.describe()
stmd_fraud=fraud.Amount.describe()

#Compare the values for both transactions
comparation=df.groupby('Class').mean()

#Under Sampling

#Build a sample dataset containg similar distribution of normal
#transactions and Frauduent transactions
#Number of fraudulent Transactions --->492

legit_sample = legit.sample(n=492)

#Concatenating two DataFrame
new_dataset=pd.concat([legit_sample,fraud],axis=0)

values_new=new_dataset['Class'].value_counts()

#Compare the values for both transactions
comparation_new=new_dataset.groupby('Class').mean()

#Splitting the data into features & target

X=new_dataset.drop(columns='Class',axis=1)
Y= new_dataset['Class']

#Split the Data into Training Data & testing Data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

#Model Regresion

model= LogisticRegression()

#Training the Logistic Regresion with Training Data 
model.fit(X_train, Y_train)


#Model Evaluation
#Accuracy Score on train_data
X_train_prediction=model.predict(X_train)
training_data_acc=accuracy_score(X_train_prediction, Y_train)
print("Accuracy of the model on training Data: ",training_data_acc)

#Accuracy Score on test_data
X_test_prediction=model.predict(X_test)
test_data_acc=accuracy_score(X_test_prediction, Y_test)
print("Accuracy of the model on test Data: ",test_data_acc)


