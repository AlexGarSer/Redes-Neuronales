# %% [markdown]
# Mini proyecto de regresion lineal | Examen Medio Curso | Redes Neuronales Artificiales
# 
# Autor: Alex Garcia Serna
# Matricula:  1725239
# Correo: alex.garciase@uanl.edu.mx
# Github: github.com/AlexGarSer
# 
# Objetivo:
# El mini proyecto consiste en desarrollar un algoritmo que recibe como entrada valores dados por el usuario de las variables de control seleccionadas como relevantes y devuelve al usuario el tiempo de producción esperado si se ajustan los controles con dichos valores dados.
# 
# La informacion utilizada para este proyecto al ser de caracter sensible se eliminaron los nombres de las columnas por razones de confidencialidad y se sustituyeron por "X" + "# de la columna".

# %% [markdown]
# Para comenzar utilizaremos las siguientes librerias

# %%
#Librerias a utlizar
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# %% [markdown]
# El dataset utilizado en este proyecto se encuentra nombrado como ProblemaRegresion.csv, 
# este dataset fue otorgado por el instructor y para uso exclusivo del examen.

# %%
# Primero se leen los datos del dataset y los pasamos a un dataframe para poder manipularlos.
datos = pd.read_csv("./Datos/ProblemaRegresion.csv")

# %% [markdown]
# Al darnos cuenta que faltaba informacion en el dataset y marcaba errores al querer utilizarla decidimos utilizar los datos de media que teniamos actualmente para rellenar los espacios vacios, ya que no senti la necesidad de eliminar columnas por que al no saber cual de todas realmente influyen de manera correcta existe la posibilidad de eliminar una que realmente sea importante.

# %%
#Se imprime la media de todas las columnas del dataframe
print(datos.mean())
print('\n'*3)

# %% [markdown]
# El procedimiento fue manual directamente en el CSV por medio de la herramienta: Edit CSV de janisdd id de extencion: janisdd.vscode-edit-csv

# %% [markdown]
# Una vez completados todos los datos, procedo a seccionar las salidas y las caracteristicas del dataset
# donde Y representa las salidas esperadas y X son las caracteristicas

# %%
Y = datos.iloc[:,:-34].values # Las etiquetas
X = datos.iloc[:,1:].values #Caracteristicas del conjunto

# %% [markdown]
# Una vez que ya tenemos los datos es hora de seccionar cuales de ellos seran utilizados para el entrenamiento de la red neuronal y cuales seran la comprobacion de que la red neuronal predice de manera correcta.
# 
# En este caso utilizaremos una proporcion 70% : 30% para otorgar mas ejemplos de entrenamiento a la red neuronal que la que va a predecir.
# 
# Al ser 766 datos utilizaremos 536 que es el equivalente al 70% de los datos para el conjunto de entrenamiento y solo 230 para el conjunto de pruebas para corroborar el funcionamiento de la red neuronal.

# %%
#Division en conjunto de entrenamiento y conjunto de prueba
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)

# Se imprime la cantidad de pruebas por cada set
print('Cantidad a utilizar para el entrenamiento: ',x_train.shape) #Se imprime la cantidade columnas y filas a utilizar
print('Cantidad a utilizar para la prueba: ',x_test.shape) #Se imprime la cantidade columnas y filas a utilizar

# %% [markdown]
# Continuaremos aplicando ahora si la regresion lineal utilizando el set de entrenamiento ya anterior mente establecido

# %%
regresion = LinearRegression() #Funcion de regresion
regresion = regresion.fit(x_train,y_train) #Funcion de entrenamiento con sus respecitvas caracteristicas y etiquetas

# %% [markdown]
# Una vez entrenado ya nuestra funcion de regresion con esta informacion pasaremos a realizar una comprobacion del aprendizaje, comparando una nueva prediccion con los datos de prueba y comparandolos con las salidas esperadas de la misma seccion de pruebas.

# %%
y_prediccion = regresion.predict(x_test)
print(y_prediccion)

# %% [markdown]
# Ya una vez obtenida hay que corroborarla con el marco de error y con nuestra r^2 para llevar acabo una interpretacion mejor.
# 
# Esta informacion la podemos interpretar de la siguiente manera:
# 
#     Si el margen de error es mayor a 50 la prediccion deja de ser buena
# 
#     Si el valor cualculado de R^2 es el mas cercano a 1 sin llegar a ser 1 entonces el resultado de prediccione es muy bueno. 

# %%
error = np.sqrt(mean_squared_error(y_test,y_prediccion))
r2 = regresion.score(x_train,y_train)

print("El valor de error: ",error)
print("El valor de R cuadrada: ",r2)


# %% [markdown]
# 

# %% [markdown]
# Para este algoritmo decidi utilizar un limpiador de consola que podemos ver en esta funcion

# %%
import os

def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)

# %% [markdown]
# Una vez evaluado y analizado el resultado de este algoritmo podemos calcular un nuevo tiempo en base a las nuevas caracteristicas que ingrese el usuario en este caso deberian ingresar las 34 caracteristicas para poder calcular el tiempo que tomara el proceso.

# %%
if(r2 > 0.75 and error < 50.0):
    input('El resultado de este entrenamiento se considera aceptable para predecir una nueva estimacion. \n presione cualquier tecla para continuar: ')
    clearConsole()
    xnueva = []
    nueva_prediccion = input("Ingrese las caracteristicas de la toma desde X1 hasta X34 separado por comas: \n").split(",")
    for n in nueva_prediccion:
        xnueva.append(n)
    new_x = [float(i) for i in xnueva]
    resultado = regresion.predict([new_x])
    print("Tiempo esperado es de: ",float(resultado)," minutos.")
else:
    print('La prediccion no fue lo suficiente precisa vuelva a interntarlo')


