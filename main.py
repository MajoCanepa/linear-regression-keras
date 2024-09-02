import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import SGD

#? Leemos los datos y los guardamos en un DataFrame.
datos = pd.read_csv('altura_peso.csv')
print(datos)
x = datos['Altura'].values
y = datos['Peso'].values

#? Se normalizan los datos para que el modelo pueda aprender más rápido.
x_media = np.mean(x)                     # media de x 
x_std = np.std(x)                        # desviación estándar de x 
x_normalizada = (x - x_media) / x_std    # se normaliza x

y_media = np.mean(y)                     # media de y
y_std = np.std(y)                        # desviación estándar de y
y_normalizada = (y - y_media) / y_std    # se normaliza y 

#? Implementación del modelo de regresión lineal.
np.random.seed(2)
modelo = Sequential()

#? uso de dense().
modelo.add(Dense(1, input_dim=1, activation='linear'))

#? uso de sgd().
sgd = SGD(learning_rate=0.0004)
modelo.compile(loss='mse', optimizer=sgd)

#? Entrenar el modelo.
epochs = 5000
batch_size = x_normalizada.shape[0]
historia = modelo.fit(x_normalizada, y_normalizada, epochs = epochs, batch_size = batch_size, verbose=1)

#? Resultado.
capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros del modelo: w = {:.4f}, b = {:.4f}'.format(w[0][0], b[0]))

#? visualización.
plt.subplot(1, 2, 1)
plt.plot(historia.history['loss'])
plt.xlabel('Épocas')
plt.ylabel('ECM')
plt.title('ECM vs. Épocas')

#? Superposición de datos y modelo y visualización.
regr_norm = modelo.predict(x_normalizada)
regr = regr_norm * y_std + y_media  # Desnormalizar las predicciones
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Datos Originales')
plt.plot(x, regr, color='red', label='Regresión lineal')
plt.title('Datos Originales y Regresión lineal')
plt.show()

#? Predicción.
altura = 170
altura_norm = (altura - x_media) / x_std
peso = modelo.predict(np.array([altura_norm]))
peso_pred = peso * y_std + y_media  
print('La predicción del peso será de {:.1f} kg para una altura de {} cm'.format(peso_pred[0][0], altura))