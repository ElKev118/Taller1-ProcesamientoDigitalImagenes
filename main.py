import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# ============================================================
# CONFIGURACIÓN
# ============================================================
VENTANA_SUAVIZADO = 15     # ventana para suavizado de posición (aumentado)
VENTANA_VELOCIDAD = 11     # ventana para suavizado de velocidad
VENTANA_ACELERACION = 11   # ventana para suavizado de aceleración
FRAME_DIFF = 5             # frames para diferencias finitas (aumentado)
VIDEO_PATH = "video_PDI.mp4"
RESULTADOS_DIR = "resultados"
VIDEO_SALIDA = os.path.join(RESULTADOS_DIR, "video_procesado.mp4")

# puntos A y B fijos
A = (1839, 320)
B = (9, 228)

# Distancia real entre A y B en metros
DISTANCIA_AB_METROS = 41.1

# Calcular escala fija basada en A y B
dist_pixeles_AB = np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
ESCALA_FIJA = DISTANCIA_AB_METROS / dist_pixeles_AB  # metros por pixel

print(f"Distancia A-B en píxeles: {dist_pixeles_AB:.2f}")
print(f"Escala calculada: {ESCALA_FIJA:.6f} m/pixel")

if not os.path.exists(RESULTADOS_DIR):
    os.makedirs(RESULTADOS_DIR)

# ============================================================
# FUNCIONES DE SUAVIZADO
# ============================================================


def suavizar_datos(datos, ventana):
    """Aplica media móvil para suavizar datos"""
    if len(datos) < ventana:
        return datos
    return uniform_filter1d(datos, size=ventana, mode='nearest')


def suavizar_savgol(datos, ventana, orden=3):
    """Aplica filtro Savitzky-Golay (mejor para derivadas)"""
    if len(datos) < ventana:
        return datos
    # Asegurar que ventana sea impar
    if ventana % 2 == 0:
        ventana += 1
    return savgol_filter(datos, ventana, orden)
