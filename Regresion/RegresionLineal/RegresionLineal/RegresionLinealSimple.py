# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:15:41 2023
//Revisar los datos de una compania de venta de equipos que se observa
//que parece haber una relacion entre el numero de llamadas de ventas y el numero
//de unidades vendidas. Es decir los vendedores que hicieron mas llamadas de ventas 
//Sin embargo la relacion no es perfecta o exacta hay algunos casos donde no se mantuvo
@author: Eduardo Castillo Garcia
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df=pd.read_csv('dataset.csv')
df.head(12)

#Graficamos el data set

plt.scatter(df['llamadas'],df['ventas'])
plt.xlabel('Llamadas')
plt.ylabel('Ventas')
plt.title('Datos')
plt.show()

#Modelo a Ajustar
#Como es la relacion de ventas con respecto a llamadas
# No Ventas = Bo+B1(No llamadas)

#Tomamos una muestra del Dataset
#0.7 corresponde al 70% del Data
frac=0.7 
#Random_state: Random_state se usa para establecer la semilla 
#para el generador aleatorio para que podamos asegurarnos de que los resultados que obtengamos se puedan reproducir.
datos_train=df.sample(frac=frac,random_state=42)
#Obtenemos una muestra aleatoria para comprobar
df1=datos_train.head(10)
datos_test=df[~df.index.isin(datos_train.index)]
#Definimos nuestras variables de interes
#Llamadas
X=np.array(datos_train['llamadas'])
#Ventas
Y=np.array(datos_train['ventas'])
#np.newaxis Aumenta la dimension de la matriz en 1
X=X[:,np.newaxis]
Y=Y[:,np.newaxis]
#Recordando que es una regresion Lineal Simple

#Bo=Y-B1X
#B1=(X'X))^-1X'y
#Mean()Calcula la media 
B1=float(np.linalg.inv(X.T@X)@X.T@Y)
Bo=float(Y.mean()-B1*X.mean())
#Graficamos la recta de regresion
u=np.linspace(10, 40)
v=Bo+B1*u

plt.plot(u,v,'r',label='Recta_regresion')
plt.scatter(datos_train['llamadas'],datos_train['ventas'],label='Puntos_Train')
plt.scatter(datos_test['llamadas'],datos_test['ventas'],label='Puntos_Test')
plt.xlabel('Llamadas')
plt.ylabel('Ventas')
plt.title('Datos')
plt.show()

#Prediccion y Error
#
def prediccion(val):
       return Bo+B1*val
res1=[]
#Llamada a la funcion prediccion 
for datos in datos_test['llamadas']:
    preposicion=prediccion(datos)
    res1.append(preposicion)
  
    
resultado={'valor_real':datos_test['ventas'],'prediccion':res1}

R=pd.DataFrame(data=resultado)

#Funcion error 
def error(real,pred):
    return np.sum((real-pred)**2/len(real))

error_cuadratico=error(R['valor_real'],R['prediccion'])

print("Error Cuadratico Medio",format(error_cuadratico))
#Estimacion de coeficientes de regresion unando scikit-learn

from sklearn.linear_model import LinearRegression

#El método 'fit' entrena el algoritmo en los datos de entrenamiento
modelo=LinearRegression()
modelo.fit(X, Y)

print('Bo:  ',format(float(modelo.coef_)))
print('B1:  ',format(float(modelo.intercept_)))


#Funcion de impresion
def lin_regre(X,y,model):
    plt.scatter(X, y, c='blue')
    plt.plot(X,model.predict(X), color='red')


lin_regre(X,Y,modelo)
plt.xlabel('Llamadas')
plt.ylabel('Ventas')
plt.title('Modelo de Regresion')
plt.show()

X_test=np.array(datos_test['llamadas'])
X_test=X_test[:,np.newaxis]

predicion2=modelo.predict(X_test)

resultado2={'Valor_real':datos_test['ventas'],'prediccion2':res1}
R2=pd.DataFrame(data=resultado2)
def error(real,pred):
    return np.sum((real-pred)**2)/len(real)

error_cuadratico2=error(R2['Valor_real'],R2['prediccion2'])
print('Error Cuadratico Medio: ',format(error_cuadratico2))
