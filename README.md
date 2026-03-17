# Análisis de Movimiento de un Vehículo Mediante Visión por Computadora 🚗🎥

**Universidad de Antioquia** | **Facultad de Ingeniería** | **Departamento de Ingeniería de Sistemas** **Asignatura:** Procesamiento Digital de Imágenes (2026-1)  
**Docente:** Sebastián Guzmán Obando  

---

## 📋 Descripción del Proyecto

El presente proyecto detalla el proceso de estimación de los parámetros cinemáticos de un vehículo (automóvil Toyota Yaris) a partir de un archivo de video. El objetivo principal es extraer la posición, velocidad y aceleración en cada instante de tiempo utilizando técnicas de procesamiento digital de imágenes y visión por computadora. 

Para lograrlo, se aplicaron conceptos de:
* Segmentación por color (espacio HSV).
* Operaciones morfológicas (apertura, cierre, dilatación).
* Detección de contornos y cálculo de centroides espaciales.
* Derivación numérica y suavizado de señales (Filtro Savitzky-Golay).

Los resultados empíricos obtenidos se compararon con modelos cinemáticos teóricos, identificando exitosamente un Movimiento Rectilíneo Uniforme (MRU) a una velocidad promedio de 47.8 km/h.

## 🛠️ Tecnologías y Librerías Utilizadas

El proyecto fue desarrollado en Python y hace uso de las siguientes herramientas:

* **OpenCV (`cv2`)**: Para la captura, procesamiento de cada fotograma del video, transformaciones de color espacial y operaciones morfológicas.
* **NumPy (`numpy`)**: Para el manejo de matrices y cálculos matemáticos vectorizados.
* **SciPy (`scipy`)**: Para la implementación del filtro Savitzky-Golay aplicado al suavizado de la señal espacial y temporal.
* **Matplotlib (`matplotlib`)**: Para la generación y visualización de las gráficas cinemáticas.

## 📂 Estructura del Repositorio

* `main.py` / `Informe_PDI_Tarea1.pdf`: Código fuente con el algoritmo de procesamiento y el informe estructurado en formato IEEE.
* `video_PDI.mp4`: Video original capturado por el equipo para el análisis.
* `resultados/`: Directorio que contiene los archivos generados por el script:
  * `video_procesado.mp4`: Video final con la trayectoria, variables en tiempo real y marcadores superpuestos.
  * `posicion.png`, `velocidad.png`, `aceleracion.png`: Gráficas resultantes del análisis cinemático.
* `Informe_PDI_Tarea1.pdf`: Informe académico detallado del proyecto en formato IEEE.

## 🚀 Instrucciones de Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/ElKev118/Taller1-ProcesamientoDigitalImagenes.git
cd Taller1-ProcesamientoDigitalImagenes
```
### 2. Instalar dependencias
```bash
pip install opencv-python numpy matplotlib scipy jupyterlab
```

### 3. Ejecutar el análisis
Asegúrate de que el archivo de video (video_PDI.mp4) se encuentre en la raíz del proyecto.

Si usas Jupyter, abre el cuaderno y ejecuta las celdas:

```bash
jupyter lab Informe_PDI_Tarea1.ipynb
```

Si prefieres correr el script directamente desde consola (si cuentas con el archivo .py):

```bash
python main.py
```

## Autores
- Kevin Esteban Ruda Gómez

- Brandon Duque García

- Simón Berrio Gaviria

