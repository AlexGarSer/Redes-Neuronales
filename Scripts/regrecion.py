#Librerias a utlizar
import pandas as pd

#Primero se leen los datos del dataset y los pasamos a un dataframe para poder manipularlos.
datos = pd.read_csv("ProblemaRegresion.csv")
#Se imprime la cantidad de filas y columnas que tiene el datasframe
print(datos.shape)
print('\n'*3)

#Se imprime el dataframe
#print(datos)
#print('\n'*3)

#Se imprime la media de todas las columnas del dataframe
print(datos.mean())
print('\n'*3)

#Se imprime la columna Y de salida
Y = datos[:766]
X = datos[:766]


print(Y[:5])
print(X[:5])

