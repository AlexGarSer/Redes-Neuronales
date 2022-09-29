#Librerias a utlizar
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


import os

def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)


#Primero se leen los datos del dataset y los pasamos a un dataframe para poder manipularlos.
datos = pd.read_csv("./Proyecto1/Datos/ProblemaRegresion.csv")

#g=sns.pairplot(data=datos['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20','x21','x22','x23','x24','x25','x26','x27','x28','x29','x30','x31','x32','x33','x34'],hue=datos['y'])

#Se imprime la cantidad de filas y columnas que tiene el datasframe
#print(datos.shape)
#print('\n'*3)

#Se imprime el dataframe
#print(datos)
#print('\n'*3)

#Se imprime la media de todas las columnas del dataframe
print(datos.mean())
print('\n'*3)

#Se establecen las caracteristicas
Y = datos.iloc[:,:-34].values # Las etiquetas
X = datos.iloc[:,1:].values #Caracteristicas del conjunto
print(Y.shape)
print(X.shape,'\n'*2)
print(X)

#Division en conjunto de entrenamiento y conjunto de prueba
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

# Se imprime la cantidad de pruebas por cada set
print(x_train.shape) #Se imprime la cantidade columnas y filas a utilizar
print(x_test.shape) #Se imprime la cantidade columnas y filas a utilizar
print(y_train.shape)
print(y_test.shape)
print('\n'*3)
regresion = LinearRegression()
print(x_train)
print('\n'*3)
regresion = regresion.fit(x_train,y_train)

y_prediccion2 = regresion.predict(x_test)

print(y_prediccion2)

error = np.sqrt(mean_squared_error(y_test,y_prediccion2))
r2 = regresion.score(x_train,y_train)

print("El valor de error: ",error)
print('\n'*3)
print("El valor de R cuadrada: ",r2)
print('\n'*3)
print("Los coeficientes son: " , regresion.coef_)
print('\n'*3)

if(r2 > 0.75 and error < 50.0):
    input('El resultado de este entrenamiento se considera aceptable para predecir una nueva estimacion. \n presione cualquier tecla para continuar: ')
    clearConsole()
    xnueva = []
    nueva_prediccion = input("Ingrese las caracteristicas de la toma desde X1 hasta X34 separado por comas: \n").split(",")
    for n in nueva_prediccion:
        xnueva.append(n)
    new_x = [float(i) for i in xnueva]
    print("Tiempo esperado de la prediccion: ",regresion.predict([new_x]))
else:
    print('La prediccion no fue lo suficiente precisa vuelva a interntarlo')