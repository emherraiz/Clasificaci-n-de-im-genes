#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 12 - Clasificación de imágenes
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   KERAS 2.2.4
#   PILLOW 6.0.0
#   SCIKIT-LEARN 0.20.3
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------




#************************************************************************************
#
# REDES NEURONALES CON 1 CAPA DE CONVOLUCIONES Y UNA CANTIDAD DE IMAGENES AUMENTADA
#
#************************************************************************************

import pandas as pnd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Definición del largo y ancho de la imagen
LARGO_IMAGEN = 28
ANCHO_IMAGEN = 28

#Carga de los datos de entrenamiento
observaciones_entrenamiento = pnd.read_csv('datas/zalando/fashion-mnist_train.csv')

#Solo se guardan las características "píxeles"
X = np.array(observaciones_entrenamiento.iloc[:, 1:])

#Se crea una tabla de categorías con ayuda del módulo Keras
y = to_categorical(np.array(observaciones_entrenamiento.iloc[:, 0]))

#Distribución de los datos de entrenamiento en datos de aprendizaje y datos de validación
#80 % de datos de aprendizaje y 20 % de datos de validación
X_aprendizaje, X_validacion, y_aprendizaje, y_validacion = train_test_split(X, y, test_size=0.2, random_state=13)


# Se redimensionan las imágenes al formato 28*28 y se realiza una adaptación de escala en los datos de los píxeles
X_aprendizaje = X_aprendizaje.reshape(X_aprendizaje.shape[0], ANCHO_IMAGEN, LARGO_IMAGEN, 1)
X_aprendizaje = X_aprendizaje.astype('float32')
X_aprendizaje /= 255

# Se hace lo mismo con los datos de validación
X_validacion = X_validacion.reshape(X_validacion.shape[0], ANCHO_IMAGEN, LARGO_IMAGEN, 1)
X_validacion = X_validacion.astype('float32')
X_validacion /= 255

#Preparación de los datos de prueba
observaciones_test = pnd.read_csv('datas/zalando/fashion-mnist_test.csv')

X_test = np.array(observaciones_test.iloc[:, 1:])
y_test = to_categorical(np.array(observaciones_test.iloc[:, 0]))

X_test = X_test.reshape(X_test.shape[0], ANCHO_IMAGEN, LARGO_IMAGEN, 1)
X_test = X_test.astype('float32')
X_test /= 255


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#Se especifican las dimensiones de la imagen de entrada
dimensionImagen = (ANCHO_IMAGEN, LARGO_IMAGEN, 1)

#Se crea la red neuronal capa por capa
redNeurona1Convolucion = Sequential()

#1- Adición de la capa de convolución que contiene
#  Capa oculta de 32 neuronas
#  Un filtro de 3x3 (Kernel) recorriendo la imagen
#  Una función de activación de tipo ReLU (Rectified Linear Activation)
#  Una imagen de entrada de 28px * 28 px
redNeurona1Convolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=dimensionImagen))

#2- Definición de la función de pooling con un filtro de 2px por 2 px
redNeurona1Convolucion.add(MaxPooling2D(pool_size=(2, 2)))

#3- Adición de una función de ignorancia
redNeurona1Convolucion.add(Dropout(0.2))

#5 - Se transforma en una sola línea
redNeurona1Convolucion.add(Flatten())

#6 - Adición de una red neuronal compuesta por 128 neuronas con una función de activación de tipo Relu
redNeurona1Convolucion.add(Dense(128, activation='relu'))

#7 - Adición de una red neuronal compuesta por 10 neuronas con una función de activación de tipo softmax
redNeurona1Convolucion.add(Dense(10, activation='softmax'))

#8 - Compilación del modelo
import keras
redNeurona1Convolucion.compile(loss=keras.losses.categorical_crossentropy,
                                  optimizer=keras.optimizers.Adam(),
                                   metrics=['accuracy'])


#9 - Aumento de la cantidad de imágenes
from keras.preprocessing.image import ImageDataGenerator
generador_imagenes = ImageDataGenerator(rotation_range=8,
                         width_shift_range=0.08,
                         shear_range=0.3,
                         height_shift_range=0.08,
                         zoom_range=0.08)


nuevas_imagenes_aprendizaje = generador_imagenes.flow(X_aprendizaje, y_aprendizaje, batch_size=256)
nuevas_imagenes_validacion = generador_imagenes.flow(X_validacion, y_validacion, batch_size=256)


#10 - Aprendizaje
historico_aprendizaje = redNeurona1Convolucion.fit_generator(nuevas_imagenes_aprendizaje,
                                                   steps_per_epoch=48000//256,
                                                   epochs=50,
                                                   validation_data=nuevas_imagenes_validacion,
                                                   validation_steps=12000//256,
                                                   use_multiprocessing=False,
                                                   verbose=1 )



#11 - Evaluación del modelo
evaluacion = redNeurona1Convolucion.evaluate(X_test, y_test, verbose=0)
print('Error:', evaluacion[0])
print('Precisión:', evaluacion[1])


#12 - Visualización de la fase de aprendizaje
import matplotlib.pyplot as plt

#Datos de precisión (accuracy)
plt.plot(historico_aprendizaje.history['accuracy'])
plt.plot(historico_aprendizaje.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Epoch')
plt.legend(['Aprendizaje', 'Test'], loc='upper left')
plt.show()

#Datos de validación y error
plt.plot(historico_aprendizaje.history['loss'])
plt.plot(historico_aprendizaje.history['val_loss'])
plt.title('Error')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['Aprendizaje', 'Test'], loc='upper left')
plt.show()


#Guardado del modelo
# serializar modelo a JSON
modelo_json = redNeurona1Convolucion.to_json()
with open("modelo/modelo.json", "w") as json_file:
    json_file.write(modelo_json)

# serializar pesos a HDF5
redNeurona1Convolucion.save_weights("modelo/modelo.h5")
print("¡Modelo guardado!")