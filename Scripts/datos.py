#Librerias a utlizar
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


#Primero se leen los datos del dataset y los pasamos a un dataframe para poder manipularlos.
datos = pd.read_csv("./Datos/ProblemaRegresion.csv")

g=sns.pairplot(datos)