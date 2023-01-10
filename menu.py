import helpers
import keras
from keras.models import model_from_json, Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
from modelo_convolucional import Convolucion, Clasificacion
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time



def lanzar():

    helpers.limpiar_pantalla()
    # Cambiará cuando carguemos el modelo
    modelo = True
    while True:

        helpers.limpiar_pantalla()
        print("========================")
        print(" BIENVENIDO AL Manager ")
        print("========================")
        print("\nNOTA: Este entrenamiento se plantea con 10 tipos de prendas:")
        print('\n - Camiseta/top\n - Pantalones\n - Jerseys\n - Vestidos\n - Abrigos\n - Sandalias\n - Camisas\n - Zapatillas\n - Bolsos\n - Botines')
        print('\nOPCIONES:')
        print("\n[1] Cargar modelo ")
        print("[2] Entrenar un nuevo modelo ")
        if type(modelo) == bool:
            print('[3] ¿? - Se desbloquea cuando dispongas de un modelo')
        else:
            print(f"[3] Datos de nuestro modelo {modelo.name}")

        if type(modelo) == bool:
            print('[4] ¿? - Se desbloquea cuando dispongas de un modelo')
        else:
            print("[4] Clasificar imagen ")

        print("[5] Salir")
        print("========================")

        opcion = input("\n> ")
        helpers.limpiar_pantalla()

        if opcion == '1':
            #----------------------------
            # CARGA DEL MODELO
            #----------------------------

            #Carga de la descripción del modelo
            direccion = input('Introduce la ruta del modelo json. Ej: Codigo Dado/modelo/modelo_4convoluciones.json\n> ')
            archivo_json = open(direccion, 'r')
            modelo_json = archivo_json.read()
            archivo_json.close()

            #Carga de la descripción de los pesos del modelo
            modelo = model_from_json(modelo_json)

            # Cargar pesos en el modelo nuevo
            direccion = input('\nIntroduce la ruta de los pesos. Ej: Codigo Dado/modelo/modelo_4convoluciones.h5\n> ')
            modelo.load_weights(direccion)

            print('¡¡¡DATOS CARGADOS CORRECTAMENTE!!!')

            input('\nPulse cualquier teclas para continuar...')

        if opcion == '2':
            direccion = input('Introduce la ruta del csv:\n> ')
            df = pd.read_csv(direccion)
            helpers.limpiar_pantalla()
            print('¡¡¡DATOS CARGADOS CORRECTAMENTE!!!\n')

            LARGO = int(input('Dime el largo de la imagen:\n> '))
            ANCHO = int(input('Dime el ancho de la imagen:\n> '))

            helpers.limpiar_pantalla()
            modelo = Convolucion(LARGO, ANCHO, df)
            modelo.preparacion_de_los_datos()
            print('¡¡¡DATOS PREPARADOS CORRECTAMENTE!!!')

            print('NOTA: No añadir muchas capas, con cada capa aumenta el tiempo de entrenamiento y el tiempo de clasificación')
            n_capas_conv = int(input('Dime el número de capas convolucionales:\n> '))
            n_capas_lineales = int(input('Dime el número de capas lineales:\n> '))


            helpers.limpiar_pantalla()
            modelo.definir_modelo(n_capas_conv, n_capas_lineales)


            epochs = int(input('¿Cuantas veces quieres pasar el conjunto de datos por la red?:\n> '))

            generar_imagenes = input('¿Quieres usar un generador de imágenes? (True/False):\n> ')
            helpers.limpiar_pantalla()
            if generar_imagenes.lower() == 'true':
                modelo.aprendizaje(True, epochs=epochs)
            else:
                modelo.aprendizaje(False, epochs=epochs)

            print('¡¡¡MODELO ENTRENADO CORRECTAMENTE!!!')
            print('Ahora le mostraremos una gráfica con la precisión y la pérdida del modelo')

            modelo.grafico_aprendizaje()

            helpers.limpiar_pantalla()

            guardar = input('¿Quieres guardar el modelo? (True/False):\n> ')
            helpers.limpiar_pantalla()
            if guardar.lower() == 'true':
                modelo.guardar_modelo()


            print('¡¡¡Fin del entrenamiento!!!\n')
            print(modelo)

            modelo = modelo.get_modelo()

            input('\nPulse cualquier teclas para continuar...')


        if opcion == '3':

            print('\n\nDATOS DEL MODELO\n\n')
            print(modelo.summary()  )

            input('\nPulse cualquier teclas para continuar...')


        if opcion == '4':

            Clasificarcion_modelo = Clasificacion(modelo)

            continuar = 'true'
            while continuar.lower() == 'true':
                imagen = input('Introduce la ruta de la imagen. Ej: Codigo Dado/imagenes/pantalon.jpg\n> ')

                helpers.limpiar_pantalla()
                Clasificarcion_modelo.clasificar(imagen)

                continuar = input('¿Quieres clasificar otra imagen? (True/False):\n> ')

                helpers.limpiar_pantalla()


            input('\nPulse cualquier teclas para continuar...')


        if opcion == '5':
            print("Saliendo...\n")
            break


