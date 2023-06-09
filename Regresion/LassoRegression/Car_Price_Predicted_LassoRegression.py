# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 14:09:11 2023
Car Price Prediction with Python
@author: Eduardo Castillo Garcia

Car Data ---->Data Processing ---->Train Test Split ---> Linear & Lasso Regression

"""

#importing the dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.pipeline import Pipeline

#Data Collection and Processing

car_dataset=pd.read_csv('D:/MachineLearnig/daTA.csv')

#Note: Selling Price and Present price is representes by k 
#We have 301 files and 9 columns

#Getting some information about dataset
info=car_dataset.info()
#In this DataSet not have null values 

#Checking the number of missing values

missing=car_dataset.isnull().sum()

#Checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

#Encoding the categorical Data
#"Fuel_type column" 

car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

#Petrol --->0
#Diesel --->1
#Petrol --->2

#"Seller_type"
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
#Dealer --->0
#Individual  --->1

#"Transmission
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
#Manual --->0
#Automatic --->1

#SPLITTING THE DATA INTO TRAINING DATA AND TEST DATA

X=car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y=car_dataset['Selling_Price']

X_train,X_test, y_train,y_test =  train_test_split(X,Y,test_size=0.1,random_state=2)


#Linear Regression
lr=LinearRegression()
lr.fit(X_train,y_train)


#Model evaluation

#prediction on training data

data_predict=lr.predict(X_train)

#R square
error_square=metrics.r2_score(y_train, data_predict)

#Visualization 
#Actual prices vs Predicted prices

plt.scatter(y_train,data_predict)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual prices vs Predicted prices')
plt.show()

#Predicting of Test Data:
t_data_predict=lr.predict(X_test)
#R square
error_square=metrics.r2_score(y_test, t_data_predict)

#Visualization 
#Actual prices vs Predicted prices

plt.scatter(y_train,data_predict)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual prices vs Predicted prices')
plt.show()
    
#LASSO REGRESSION

ls=Lasso()
ls.fit(X_train,y_train)


#prediction on training data

data_predict_ls=ls.predict(X_train)

#R square
error_square_ls=metrics.r2_score(y_train, data_predict_ls)