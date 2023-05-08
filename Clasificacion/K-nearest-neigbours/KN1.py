# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:17:38 2023

@author: Eduardo Castillo
"""
##Librerias###
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
##Preparamos la data###
dataset=datasets.load_breast_cancer()

#verificamos la informacion del data 

print('Informacion de la data')
print (dataset.keys())

#Caracteristicas de la DATA

print('Caracteristicas de la DATA')
print(datasets.descr)

#Selecionamos todas las columnas 

X=dataset.data

#Definimos los datos correspondientes a las etiquetas
y=dataset.target
#If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include
#Separamos los datos de train en entrenamiento y prueba 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.70)

#Definimos el algoritmo a utilizar
#n_neighbors= 5 los datos predeterminados del algortimo
#pint, default=2
#Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used
algoritmo =KNeighborsClassifier(n_neighbors= 4, metric='minkowski',p=2)

#Entrenamos el modelo
algoritmo.fit(X_train, y_train)
#Realizo una prediccion
y_pred = algoritmo.predict(X_test)

#Verificamos la matriz de confusion
matriz = confusion_matrix(y_test, y_pred)
 
 
print('La matriz de confusiones es: ')
print(matriz)

#Verificamos la presicion del modelo

precision = precision_score(y_test, y_pred)

print('La presicion del modelo es:')
print(precision)

"""La presicion es del 0.96 es muy buena este algoritmo para 
este conjunto de datos
"""
