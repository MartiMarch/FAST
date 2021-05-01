"""
    Algoritmo FAST. No se va a drtectar movimiento, simplemente se va a encontrar los puntos clave.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Cámara desde la que se leen las imágenes
camara = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Objeto Fast
objetoFast = cv2.FastFeatureDetector_create()

while True:
    # Imagen leída de la cámara
    _, imagen = camara.read()

    # Se transforma la imagen a gris
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Se obtienen los puntos clave
    puntosClave = objetoFast.detect(imagenGris, None)

    # Se dibujan los puntos clave
    imagenDibujada = cv2.drawKeypoints(imagen, puntosClave, None, color = (255, 0, 0))

    # Se muestran los puntos clave
    cv2.imshow("FAST", imagenDibujada)

    # Pulsar 'q' para salir
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break