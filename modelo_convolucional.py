# Importamos las librerias que vamos a utilizar
import helpers
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
import time


class Convolucion():
    '''Clase que contiene los métodos necesarios para la creación de un modelo de aprendizaje profundo

    Raises:
        ValueError: Si el largo y el ancho no se corresponden con los datos recogidos
    '''

    CLASES = ['Camiseta/top', 'Pantalones', 'Jerseys', 'Vestidos', 'Abrigos', 'Sandalias', 'Camisas', 'Zapatillas', 'Bolsos', 'Botines']
    NUMERO_DE_CLASES = len(CLASES)


    def __init__(self, LARGO_IMAGEN, ANCHO_IMAGEN, datos_entrenamiento):


        if LARGO_IMAGEN != ANCHO_IMAGEN:
            print('Para un aprendizaje optimo es preferible utilizar imagenes cuadradas')

        self.LARGO_IMAGEN = LARGO_IMAGEN
        self.ANCHO_IMAGEN = ANCHO_IMAGEN

        #Solo se guardan las características "píxeles"
        self.X = np.array(datos_entrenamiento.iloc[:, 1:])

        #Se crea una tabla de categorías con la ayuda del módulo Keras
        self.y = to_categorical(np.array(datos_entrenamiento.iloc[:, 0]))

        if LARGO_IMAGEN * ANCHO_IMAGEN != self.X.shape[1]:
            raise ValueError('El largo y el ancho no se corresponden con los datos recogidos')


    def preparacion_de_los_datos(self):
        '''Método que prepara los datos para el entrenamiento del modelo
        '''
        #Distribución de los datos de entrenamiento en datos de aprendizaje y datos de validación
        #80 % de datos de aprendizaje y 20 % de datos de validación
        X_aprendizaje, X_validacion, self.y_aprendizaje, self.y_validacion = train_test_split(self.X, self.y, test_size=0.2, random_state=13)


        # Se redimensionan las imágenes al formato 28*28 y se realiza una adaptación de escala en los datos de los píxeles
        X_aprendizaje = X_aprendizaje.reshape(X_aprendizaje.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_aprendizaje = X_aprendizaje.astype('float32')
        self.X_aprendizaje /= 255

        # Se hace lo mismo con los datos de validación
        X_validacion = X_validacion.reshape(X_validacion.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_validacion = X_validacion.astype('float32')
        self.X_validacion /= 255

        #Preparación de los datos de prueba
        X_test = self.X
        self.y_test = self.y

        # Al ser una imagen en escala de grises solo tiene un canal
        X_test = X_test.reshape(X_test.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_test = X_test.astype('float32')
        self.X_test /= 255


    def definir_modelo(self, n_capas_no_lineales, n_capas_lineales):
        '''Método que define el modelo de aprendizaje profundo'''

        #Se especifican las dimensiones de la imagen de entrada
        dimensionImagen = (self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)

        #Se crea la red neuronal capa por capa
        redNeurona1Convolucion = Sequential()

        for i in range(n_capas_no_lineales):
            print('Capa convolucional', i+1)
            n_neuronas = int(input('Número de neuronas: '))
            tamano_filtro = list()
            for j in range(2):
                dim = int(input('Tamaño de filtro: '))
                tamano_filtro.append(dim)
            tamano_filtro = tuple(tamano_filtro)
            f_activacion = input('Función de activación: ')
            # Se añade la capa de convolución
            redNeurona1Convolucion.add(Conv2D(n_neuronas, tamano_filtro, activation=f_activacion, input_shape=dimensionImagen))

            datos_normalizados = input('¿Desea normalizar los datos? (True/False): ')
            if datos_normalizados.lower() == 'true':
                # Se añade la capa de normalización
                redNeurona1Convolucion.add(BatchNormalization())

            activacion_pooling = input('¿Desea utilizar MaxPooling? (True/False): ')
            if activacion_pooling.lower() == 'true':
                # Introducimos el pool size
                pool_size = list()
                for j in range(2):
                    dim = int(input('Tamaño de pool: '))
                    pool_size.append(dim)

                pool_size = tuple(pool_size)
                # Se añade la capa de Pooling
                redNeurona1Convolucion.add(MaxPooling2D(pool_size=pool_size))

            activacion_dropout = input('¿Desea utilizar Dropout? (True/False): ')
            if activacion_dropout.lower() == 'true':
                # Introducimos el dropout
                dropout = float(input('Dropout: '))
                # Se añade la capa de Dropout
                redNeurona1Convolucion.add(Dropout(dropout))

            helpers.limpiar_pantalla()

            print('Capa', i+1, 'añadida correctamente')

            input('\nPulse cualquier teclas para continuar...')

            helpers.limpiar_pantalla()

        # Se añade la capas de aplanamiento
        redNeurona1Convolucion.add(Flatten())

        for i in range(n_capas_lineales):
            print('Capa lineal', i+1)
            n_neuronas = int(input('Número de neuronas: '))
            f_activacion = input('Función de activación: ')
            # Se añade la capa de aplanamiento
            redNeurona1Convolucion.add(Dense(n_neuronas, activation=f_activacion))

            datos_normalizados = input('¿Desea normalizar los datos? (True/False): ')
            if datos_normalizados.lower() == 'true':
                # Se añade la capa de normalización
                redNeurona1Convolucion.add(BatchNormalization())

            activacion_dropout = input('¿Desea utilizar Dropout? (True/False): ')
            if activacion_dropout.lower() == 'true':
                # Introducimos el dropout
                dropout = float(input('Dropout: '))
                # Se añade la capa de Dropout
                redNeurona1Convolucion.add(Dropout(dropout))

            helpers.limpiar_pantalla()

            print('Capa', i+1, 'añadida correctamente')

            input('\nPulse cualquier teclas para continuar...')

            helpers.limpiar_pantalla()

        # Se añade la capa de salida
        redNeurona1Convolucion.add(Dense(self.NUMERO_DE_CLASES, activation='softmax'))

        # Se compila el modelo
        redNeurona1Convolucion.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Se guarda el modelo como atributo de la clase
        self.modelo = redNeurona1Convolucion

    def aprendizaje(self, generar_imagenes, batch_size = 256, epochs = 10, verbose = 1):
        '''Método que permite realizar el aprendizaje de la red neuronal

        Args:
            generar_imagenes (bool): Generar imágenes de entrenamiento
            batch_size (int, optional): Tamaño del lote. Defaults to 256.
            epochs (int, optional): Cantidad de veces que pasamos el conjunto de datos por la red. Defaults to 10.
            verbose (int, optional): Visualizacion de los datos. Defaults to 1.
        '''
        if generar_imagenes:
            # Se genera un generador de imágenes
            # Con el fin de aumentar el número de imágenes de entrenamiento
            # Nota: Este generador se puede modificar
            generadorImagenes = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, height_shift_range=0.08, shear_range=0.3, zoom_range=0.08)

            nuevas_imagenes_aprendizaje = generadorImagenes.flow(self.X_aprendizaje, self.y_aprendizaje, batch_size=batch_size)
            nuevas_imagenes_validacion = generadorImagenes.flow(self.X_validacion, self.y_validacion, batch_size=batch_size)


            #10 - Aprendizaje
            start = time.perf_counter()
            self.historico_aprendizaje = self.modelo.fit_generator(nuevas_imagenes_aprendizaje,
                                                            steps_per_epoch=self.X_aprendizaje.shape[0]//batch_size,
                                                            epochs=epochs,
                                                            validation_data=nuevas_imagenes_validacion,
                                                            validation_steps=self.y_aprendizaje.shape[0]//batch_size,
                                                            use_multiprocessing=False,
                                                            verbose=verbose )

            stop = time.perf_counter()

            print("Tiempo de aprendizaje = "+str(stop-start))

        else:
            self.historico_aprendizaje  = self.modelo.fit(self.X_aprendizaje, self.y_aprendizaje,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(self.X_validacion, self.y_validacion))


    def grafico_aprendizaje(self):
        '''Muestra los gráficos de aprendizaje y error
        '''
        #Datos de precisión (accuracy)
        plt.plot(self.historico_aprendizaje.history['accuracy'])
        plt.plot(self.historico_aprendizaje.history['val_accuracy'])
        plt.title('Precisión del modelo')
        plt.ylabel('Precisión')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()

        #Datos de validación y error
        plt.plot(self.historico_aprendizaje.history['loss'])
        plt.plot(self.historico_aprendizaje.history['val_loss'])
        plt.title('Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()


    def guardar_modelo(self, ruta_modelo = "modelo.json", ruta_pesos = "pesos.h5"):
        '''Guarda el modelo en formato JSON y los pesos en formato HDF5

        Args:
            ruta_modelo (str, optional): Defaults to "modelo.json".
            ruta_pesos (str, optional): Defaults to "pesos.h5".
        '''
        #Guardado del modelo
        # serializar modelo a JSON
        modelo_json = self.modelo.to_json()
        with open(ruta_modelo, "w") as json_file:
            json_file.write(modelo_json)

        # serializar pesos a HDF5
        self.modelo.save_weights(ruta_pesos)
        print("¡Modelo guardado!")

    def get_modelo(self):
        return self.modelo


    def __str__(self):
        # Evaluación del modelo
        self.puntuacion = self.modelo.evaluate(self.X_test, self.y_test, verbose=0)
        print('Error:', self.puntuacion[0])
        print('Precisión:', self.puntuacion[1])
        return ''


class Clasificacion(Convolucion):
    def __init__(self, modelo):
        self.modelo = modelo

    def clasificar(self, imagen):
        imagen = Image.open(imagen).convert('L')


        # Dimensión de la imagen
        largo = float(imagen.size[0])
        alto = float(imagen.size[1])

        # Creación de una imagen nueva
        nuevaImagen = Image.new('L', (28, 28), (255))


        #Redimensionamiento de la imagen
        #La imagen es más larga que alta, la ponemos a 20 píxeles
        if largo > alto:
                #Se calcula la relación de ampliación entre la altura y el largo
                relacionAltura = int(round((20.0 / largo * alto), 0))
                if (relacionAltura == 0):
                    nAltura = 1

                #Redimensionamiento
                img = imagen.resize((20, relacionAltura), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
                #Posición horizontal
                posicion_alto = int(round(((28 - relacionAltura) / 2), 0))

                nuevaImagen.paste(img, (4, posicion_alto))  # pegar imagen redimensionada en lienzo en blanco
        else:

            relacionAltura = int(round((20.0 / alto * largo), 0))  # redimensionar anchura según relación altura
            if (relacionAltura == 0):  # caso raro pero el mínimo es 1 píxel
                relacionAltura = 1

            #Redimensionamiento
            img = imagen.resize((relacionAltura, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

            #Cálculo de la posición vertical
            altura_izquierda = int(round(((28 - relacionAltura) / 2), 0))
            nuevaImagen.paste(img, (altura_izquierda, 4))

            #Recuperación de los píxeles
            pixeles = list(nuevaImagen.getdata())

            #Normalización de los píxeles
            tabla = [(255 - x) * 1.0 / 255.0 for x in pixeles]

            import numpy as np
            #Transformación de la tabla en tabla numpy
            img = np.array(tabla)

            #Se transforma la tabla lineal en imagen 28x20
            imagen_test = img.reshape(1, 28, 28, 1)

            prediccion = self.modelo(imagen_test)
            prediccion = np.argmax(prediccion, axis=-1)
            print()
            print("La imagen es: " + self.CLASES[prediccion[0]])
            print()

            #Extracción de las probabilidades
            probabilidades = self.modelo.predict(imagen_test)

            i=0
            for clase in self.CLASES:
                print(clase + ": "+str((probabilidades[0][i]*100))+"%")
                i=i+1

