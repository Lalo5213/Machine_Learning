# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:58:57 2023

El objetivo del programa es predicir el precio de las casas en bostons
de acuerdo al numero de habitaciones que cuenta la vivienda,el tiempo de ocupacion
y la distancia que se encuentra la misma de los centros de trabajo de boston

@author: casti
"""
#Regresion Lineal Multiple
#y=a1x1+a2x2+anxn.....

import numpy as np
from  sklearn import datasets,linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#importamos los dataset
boston=datasets.load_boston()
#Verificamos la informacion del data set
#print(boston.keys())
#verificamos las caracteristicas del dataset
#print(boston.DESCR)

#informacion de las columnas
#print(boston.feature_names)

#Son nuestras variables de interes
#Seleccionamos la columna 5 que corresponde al numero de cuartos
#Seleccioneamos la columan AGE, DIS Y 
X_MULTIPLE=boston.data[:,5:8]

print(X_MULTIPLE)
Y_MULTIPLE=boston.target


####!@@@IMPLEMENTACION DE LA REGRESION LINEAL MULTIPLE###############
#test_zize utilizamos el 20 porciento de los datos como prueba
x_train, x_test, y_train, y_test = train_test_split(X_MULTIPLE,Y_MULTIPLE,test_size=0.2)

#Definimos el algoritmo a utiliza
lr_multiple=linear_model.LinearRegression()

#Entrenamos el algoritmo
lr_multiple.fit(x_train,y_train)

#Realizamos la prediccion del comportamiento
Y_pred_mult=lr_multiple.predict(x_test)

print('Datos del modelo de Regresion Multiple')
print('Valor de las pendientes o coeficientes a:', lr_multiple.coef_)
print('Valor de la interseccion B o coefciente b',lr_multiple.intercept_)
print('Calculamos la presicion del algoritmo:',lr_multiple.score(x_train,y_train))

#Como la presicion se aleja de 1 podemos establecer que este modelo no es adecuado para estos
#tipos de datos
#Graficamos 
#plt.scatter(X_MULTIPLE, Y)
#plt.xlabel('Numero de Habitaciones')
#plt.ylabel('Valor medio')
#plt.show()


