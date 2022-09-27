#Librerias a utlizar
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Primero se leen los datos del dataset y los pasamos a un dataframe para poder manipularlos.
datos = pd.read_csv("./Datos/ProblemaRegresion.csv")

#Se imprime la cantidad de filas y columnas que tiene el datasframe
print(datos.shape)
print('\n'*3)

#Se imprime el dataframe
print(datos)
print('\n'*3)

#Se imprime la media de todas las columnas del dataframe
print(datos.mean())
print('\n'*3)

x1 = datos['x1'].values
x2 = datos['x2'].values
x3 = datos['x3'].values
x4 = datos['x4'].values
x5 = datos['x5'].values
x6 = datos['x6'].values
x7 = datos['x7'].values
x8 = datos['x8'].values
x9 = datos['x9'].values
x10 = datos['x10'].values
x11 = datos['x11'].values
x12 = datos['x12'].values
x13 = datos['x13'].values
x14 = datos['x14'].values
x15 = datos['x15'].values
x16 = datos['x16'].values
x17 = datos['x17'].values
x18 = datos['x18'].values
x19 = datos['x19'].values
x20 = datos['x20'].values
x21 = datos['x21'].values
x22 = datos['x22'].values
x23 = datos['x23'].values
x24 = datos['x24'].values
x25 = datos['x25'].values
x26 = datos['x26'].values
x27 = datos['x27'].values
x28 = datos['x28'].values
x29 = datos['x29'].values
x30 = datos['x30'].values
x31 = datos['x31'].values
x32 = datos['x32'].values
x33 = datos['x33'].values
x34 = datos['x34'].values

#Se establecen las caracteristicas
Y = np.array(datos[['y']])
X = np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34]).T #Caracteristicas del conjunto

print(Y.shape)
print(X.shape)

regresion = LinearRegression()

regresion = regresion.fit(X,Y)


### Train a neural net for regression 
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=100, max_iter=1500, tol=0.0001 , verbose=True)
#mlp = MLPRegressor(solver='adam', hidden_layer_sizes=400, max_iter=15000, tol=1e-20 , verbose=True)
print(len(X), len(Y))  
#xtr = xtr[0:len(xtr)-5000]            
tmodel = mlp.fit(X,Y)

y_prediccion = tmodel.predict(X)

error = np.sqrt(mean_squared_error(Y,y_prediccion))
r2 = regresion.score(X,Y)




print("El valor de error: ",error)
print("El valor de R cuadrada: ",r2)
print("Los coeficientes son: " , regresion.coef_)

print("Tiempo esperado de la prediccion:" ,regresion.predict(datos[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34']]))