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


#----------------------------
# CARGA DEL MODELO
#----------------------------


#Carga de la descripción del modelo
archivo_json = open('modelo/modelo_4convoluciones.json', 'r')
modelo_json = archivo_json.read()
archivo_json.close()

#Carga de la descripción de los pesos del modelo
from keras.models import model_from_json
modelo = model_from_json(modelo_json)
# Cargar pesos en el modelo nuevo
modelo.load_weights("modelo/modelo_4convoluciones.h5")


#Definición de las categorías de clasificación
clases = ["Una camiseta/top","Un pantalón","Un jersey","Un vestido","Un abrigo","Una sandalia","Una camisa","Zapatillas","Un bolso","Botines"]

#---------------------------------------------
# CARGA Y TRANSFORMACIÓN DE UNA IMAGEN
#---------------------------------------------

from PIL import Image, ImageFilter

#Carga de la imagen
imagen = Image.open("imagenes/zapatilla.jpg").convert('L')

#Dimensión de la imagen
largo = float(imagen.size[0])
alto = float(imagen.size[1])

#Creación de una imagen nueva
nuevaImagen = Image.new('L', (28, 28), (255))

#Redimensionamiento de la imagen
#La imagen es más larga que alta, la ponemos a 20 píxeles
if largo > alto:
        #Se calcula la relación de ampliación entre la altura y el largo
        relacionAltura = int(round((20.0 / largo * altura), 0))
        if (relacionAltura == 0):
            nAltura = 1

        #Redimensionamiento
        img = image.resize((20, relacionAltura), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
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

prediccion = modelo.predict_classes(imagen_test)
print()
print("La imagen es: "+clases[prediccion[0]])
print()

#Extracción de las probabilidades
probabilidades = modelo.predict_proba(imagen_test)

i=0
for clase in clases:
    print(clase + ": "+str((probabilidades[0][i]*100))+"%")
    i=i+1


