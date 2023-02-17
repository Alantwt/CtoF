#MODELO DE RED NEURONAL CON MAS DE UNA CAPA

import os
from tkinter.filedialog import SaveFileDialog
from traceback import FrameSummary
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#ENTRADAS DE GRADOS CELCIUS
celsius = np.array([-40, -10, 0, 8, 15, 22, 30],dtype=float)
#SALIDAS EN FARENHEIT
fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 86],dtype=float)
plt.xlabel("Celcius")
plt.ylabel("fahrenheit")
plt.plot(celsius,fahrenheit)
plt.show()

oculta1 = tf.keras.layers.Dense(units = 3, input_shape = [1])
oculta2 = tf.keras.layers.Dense(units = 3)
salida = tf.keras.layers.Dense(units= 1)

#MODELO DE KERAS PARA DARLE LAS CAPAS 
modelo = tf.keras.Sequential([oculta1,oculta2,salida])
 
#COMPILAR EL MODELO PARA EMPEZAR A ENTRENARLO CON DATOS
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),#algoritmo adam, que permite saber como ajustar los sesgos entre las neuronas de forma eficiente, para que vaya aprendiendo, parametro: tasa de aprendizaje
    loss = "mean_squared_error",#funcion de perdida: error cuadratico medio, poca cantidad de errores grandes es peor que mucha cantidad de errores pequeños
)

#EMPEZAR A ENTRENAR
print("Entrenando....")
histotial = modelo.fit(celsius,fahrenheit,epochs = 500, verbose=False)# parametros: datos de entrada, datos de salida, vualtas de intentos, que no imprima tantos datos
print("Modelo Entrenado")

#RESULTADO DE LA FUNCION DE PERDIDA
#verifica que tan mal estan los resultados  de la red en cada vuelta
plt.xlabel("# Epoca")
plt.ylabel("# Magnitud de Perdida")
plt.plot(histotial.history["loss"]) 
plt.show()

#PRIMERA PREDICCION
print("Primera prediccin")
resultado = modelo.predict([100.0])
print("El Resultado es: ",resultado)

print("Prediccion...")
while True:
    print("Introdusca los celcius")
    inp = input()
    if inp == "exit":
        break
    resultado = modelo.predict([float(inp)])
    print(f"Fahrenheit: {resultado}")