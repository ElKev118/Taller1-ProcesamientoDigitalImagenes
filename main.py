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


# ============================================================
# PROCESAMIENTO DE VIDEO
# ============================================================

def procesar_video(cap):

    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_SALIDA, fourcc, fps, (width, height))

    kernel = np.ones((5, 5), np.uint8)
    kernel_grande = np.ones((7, 7), np.uint8)  # Para mejor limpieza

    centroides_x = []      # Posición x del centroide
    centroides_y = []      # Posición y del centroide
    tiempos = []

    trayectoria = []

    velocidades_display = []  # Para mostrar en tiempo real
    aceleraciones_display = []

    frame_index = 0

    paused = False
    frame = None

    while True:

        if not paused:

            ret, frame = cap.read()

            if not ret:
                break

            tiempo = frame_index * dt

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # =============================
            # DETECCIÓN ROJO
            # =============================

            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([15, 255, 255])

            lower_red2 = np.array([165, 70, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            mask = mask1 + mask2

            # Operaciones morfológicas mejoradas
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_grande)
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_DILATE, kernel, iterations=1)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            vehiculo = None
            area_max = 0

            for c in contours:

                area = cv2.contourArea(c)

                if area > 800 and area > area_max:
                    area_max = area
                    vehiculo = c

            if vehiculo is not None:

                M = cv2.moments(vehiculo)

                if M["m00"] != 0:

                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    centroides_x.append(cx)
                    centroides_y.append(cy)
                    tiempos.append(tiempo)
                    trayectoria.append((cx, cy))

                    cv2.drawContours(frame, [vehiculo], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)

            # =========================
            # CÁLCULO EN TIEMPO REAL (suavizado)
            # =========================
            vel = 0
            acel = 0

            if len(centroides_x) > FRAME_DIFF * 2:
                # Suavizar posiciones para cálculo en tiempo real
                pos_suave = suavizar_datos(np.array(centroides_x), min(
                    VENTANA_SUAVIZADO, len(centroides_x)))

                # Velocidad con diferencias finitas centradas (más estable)
                dx = (pos_suave[-1] - pos_suave[-1-FRAME_DIFF]) * ESCALA_FIJA
                dt_total = FRAME_DIFF * dt
                vel = dx / dt_total  # Puede ser negativo si va de A a B

                velocidades_display.append(vel)

            if len(velocidades_display) > FRAME_DIFF * 2:
                # Suavizar velocidades para aceleración
                vel_suave = suavizar_datos(np.array(velocidades_display), min(
                    VENTANA_VELOCIDAD, len(velocidades_display)))

                dv = vel_suave[-1] - vel_suave[-1-FRAME_DIFF]
                dt_total = FRAME_DIFF * dt
                acel = dv / dt_total

                aceleraciones_display.append(acel)

            for i in range(1, len(trayectoria)):
                cv2.line(frame, trayectoria[i-1],
                         trayectoria[i], (255, 0, 0), 2)

            cv2.circle(frame, A, 10, (0, 255, 255), -1)
            cv2.circle(frame, B, 10, (0, 255, 255), -1)
            cv2.putText(frame, "A", (A[0]+15, A[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "B", (B[0]+15, B[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Mostrar velocidad con signo (negativo = hacia B)
            vel_kmh = abs(vel) * 3.6  # Convertir a km/h para referencia
            cv2.putText(frame, f"Velocidad: {abs(vel):.2f} m/s ({vel_kmh:.1f} km/h)", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Aceleracion: {acel:.2f} m/s^2", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Tiempo: {tiempo:.2f}s", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(frame, f"Escala: {ESCALA_FIJA:.4f} m/px", (50, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            frame_index += 1

            out.write(frame)

        display = frame.copy()

        if paused:
            cv2.putText(display, "PAUSADO", (800, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        cv2.putText(display, "SPACE = Pausar | ESC = Salir", (50, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Procesamiento", display)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:   # ESC
            break

        if key == 32:   # SPACE
            paused = not paused

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return np.array(centroides_x), np.array(centroides_y), np.array(tiempos)

# ============================================================
# ANÁLISIS FINAL (con filtro Savitzky-Golay)
# ============================================================


def analisis_cinematico(pos_pixeles, tiempos):
    """Análisis cinemático con suavizado Savitzky-Golay"""

    if len(pos_pixeles) < VENTANA_SUAVIZADO:
        print("Advertencia: pocos datos para análisis robusto")
        return None, None, None

    dt = tiempos[1] - tiempos[0]

    # 1. Suavizar posición en píxeles
    pos_suave_px = suavizar_savgol(pos_pixeles, VENTANA_SUAVIZADO)

    # 2. Convertir a metros
    pos_m = pos_suave_px * ESCALA_FIJA

    # 3. Calcular velocidad con diferencias finitas (usando más frames)
    # Usar gradiente numérico que es más estable que diff
    vel = np.gradient(pos_m, dt)

    # 4. Suavizar velocidad
    vel_suave = suavizar_savgol(vel, VENTANA_VELOCIDAD)

    # 5. Calcular aceleración
    acel = np.gradient(vel_suave, dt)

    # 6. Suavizar aceleración
    acel_suave = suavizar_savgol(acel, VENTANA_ACELERACION)

    return pos_m, vel_suave, acel_suave

# ============================================================
# GRÁFICAS
# ============================================================

def graficas(t, pos, v, a):
    """Genera gráficas de posición, velocidad y aceleración"""
    
    _, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Gráfica de posición
    axes[0].plot(t, pos, 'b-', linewidth=1.5)
    axes[0].set_title("Posición vs Tiempo")
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Posición (m)")
    axes[0].grid(True, alpha=0.3)
    
    # Gráfica de velocidad
    axes[1].plot(t, v, 'g-', linewidth=1.5)
    axes[1].set_title("Velocidad vs Tiempo")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Velocidad (m/s)")
    axes[1].grid(True, alpha=0.3)
    
    # Añadir línea de velocidad promedio
    vel_promedio = np.mean(np.abs(v))
    axes[1].axhline(y=-vel_promedio, color='r', linestyle='--', 
                    label=f'Promedio: {vel_promedio:.2f} m/s ({vel_promedio*3.6:.1f} km/h)')
    axes[1].legend()
    
    # Gráfica de aceleración
    axes[2].plot(t, a, 'r-', linewidth=1.5)
    axes[2].set_title("Aceleración vs Tiempo")
    axes[2].set_xlabel("Tiempo (s)")
    axes[2].set_ylabel("Aceleración (m/s²)")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS_DIR, "cinematica_completa.png"), dpi=150)
    
    # También guardar gráficas individuales
    plt.figure(figsize=(8, 5))
    plt.plot(t, pos, 'b-', linewidth=1.5)
    plt.title("Posición vs Tiempo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Posición (m)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTADOS_DIR, "posicion.png"), dpi=150)
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, v, 'g-', linewidth=1.5)
    plt.title("Velocidad vs Tiempo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad (m/s)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=-vel_promedio, color='r', linestyle='--', 
                label=f'Promedio: {vel_promedio:.2f} m/s')
    plt.legend()
    plt.savefig(os.path.join(RESULTADOS_DIR, "velocidad.png"), dpi=150)
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, a, 'r-', linewidth=1.5)
    plt.title("Aceleración vs Tiempo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Aceleración (m/s²)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTADOS_DIR, "aceleracion.png"), dpi=150)
    
    plt.close('all')
    
    # Imprimir estadísticas
    print("\n" + "="*50)
    print("ESTADÍSTICAS DEL MOVIMIENTO")
    print("="*50)
    print(f"Duración total: {t[-1]:.2f} s")
    print(f"Distancia recorrida: {abs(pos[-1] - pos[0]):.2f} m")
    print(f"Velocidad promedio: {vel_promedio:.2f} m/s ({vel_promedio*3.6:.1f} km/h)")
    print(f"Velocidad máxima: {np.max(np.abs(v)):.2f} m/s ({np.max(np.abs(v))*3.6:.1f} km/h)")
    print(f"Aceleración promedio: {np.mean(np.abs(a)):.2f} m/s²")
    print(f"Aceleración máxima: {np.max(np.abs(a)):.2f} m/s²")
    print("="*50)


# ============================================================
# MAIN
# ============================================================

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("No se pudo abrir el video")
        return

    centroides_x, _, tiempos = procesar_video(cap)
    
    if len(centroides_x) < VENTANA_SUAVIZADO:
        print("Error: No se detectaron suficientes puntos del vehículo")
        return

    pos, vel, acel = analisis_cinematico(centroides_x, tiempos)
    
    if pos is None:
        print("Error en el análisis cinemático")
        return

    graficas(tiempos, pos, vel, acel)

    print("\nProceso terminado.")
    print("Video procesado guardado en:", VIDEO_SALIDA)
    print("Gráficas guardadas en:", RESULTADOS_DIR)


if __name__ == "__main__":
    main()
