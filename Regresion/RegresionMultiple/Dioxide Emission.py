# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:27:26 2023
FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon 
dioxide emissions for new light-duty vehicles for retail sale in Canada
@author: casti
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv('D:/MachineLearnig/fuelConsumption.csv')

#Lets some features that we want to use fore regression
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

#PLOT EMISSION RESPECT TO ENGINE SIZE
plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS,color='blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.show()


#MULTIPLE REGRESSION MODEL
#ENTENDIENDO LOS DATOS
x_multiple=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y_multiple=df['CO2EMISSIONS']

#test_zize utilizamos el 20 porciento de los datos como prueba

x_train, x_test, y_train, y_test = train_test_split(x_multiple,y_multiple,test_size=0.2)
#There are multiple variables that impact the CO2EMISSIONS. 

regr = linear_model.LinearRegression()
#Entrenamos el modelo
regr.fit (x_train, y_train)
print ('Coefficients: ', regr.coef_)
#Varianza:Cuadrado de la desviacion estandar debe aproximarse a 1 

print('Calculamos la presicion del algoritmo:',regr.score(x_train,y_train))
#La varianza se aproxima a 1 es igual a 0.8631 por lo tanto el modelo puede dar un resultado aproximado 
