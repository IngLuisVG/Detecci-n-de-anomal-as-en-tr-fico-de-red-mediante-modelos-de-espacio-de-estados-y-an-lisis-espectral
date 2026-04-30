# Detección de Anomalías en Tráfico de Red (SSM + Filtro de Kalman)

Este repositorio contiene la implementación en PyTorch de un sistema avanzado para la detección de anomalías en tráfico de red. El modelo combina **Modelos de Espacio de Estados Selectivos (Selective SSM)**, **Bloques Espectrales (FFT)** y un **Filtro de Kalman** para el suavizado de predicciones temporales. 

El código está ajustado para trabajar con el dataset **UNSW-NB15**.

---

## 📌 Requisitos Previos

El código está diseñado para ejecutarse directamente en **Google Colab** para evitar problemas de dependencias y aprovechar la aceleración por hardware (GPU) de forma gratuita. 

Las librerías principales utilizadas son:
- `torch`
- `pandas` & `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`

*(Todas se instalan automáticamente en la primera celda del notebook).*

---

## 📂 Preparación de los Datos

Antes de correr el notebook, asegúrate de tener los archivos de datos listos:
1. `UNSW_NB15_training-set.csv`
2. `UNSW_NB15_testing-set.csv`

**Para cargarlos en Colab tienes dos opciones:**
* **Opción A (Recomendada):** Sube los archivos a tu Google Drive, monta el Drive en Colab y ajusta las rutas en la celda de carga de datos (ej. `/content/drive/MyDrive/ruta-a-tus-datos/`).
* **Opción B (Temporal):** Usa el menú de la izquierda en Colab (icono de carpeta) y arrastra los archivos allí, o ejecuta la celda de carga manual. Se borrarán al cerrar la sesión.

---

## 🚀 Cómo ejecutar el código (Paso a Paso)

El notebook es **completamente secuencial** y modular. Para ejecutarlo:

1. **Abrir el Notebook:** Abre el archivo `.ipynb` en Google Colab.
2. **Activar GPU (Opcional pero recomendado):** En el menú superior, ve a `Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` y selecciona **T4 GPU** (o cualquier acelerador disponible).
3. **Ejecución Completa:** * Ve al menú superior y haz clic en **`Entorno de ejecución` > `Ejecutar todo`** (o usa el atajo `Ctrl + F9`).
   * *Nota:* Si usaste carga manual de datos, asegúrate de ejecutar primero las celdas de subida, esperar a que carguen los CSV, y luego ejecutar el resto.

---

## 📊 ¿Qué hace el notebook exactamente?

Al ejecutar de inicio a fin, el pipeline realiza lo siguiente automáticamente:

1. **Preprocesamiento:** Limpia columnas innecesarias, codifica variables categóricas y normaliza (MinMaxScaler).
2. **Balanceo:** Aplica SMOTE + Undersampling a la clase minoritaria en entrenamiento.
3. **Secuenciación:** Crea ventanas de tiempo superpuestas (sliding windows de longitud 20).
4. **Entrenamiento:** Entrena la red neuronal profunda (Temporal + Spectral + Fusion) y guarda automáticamente los pesos del mejor modelo en un archivo llamado `checkpoint.pt`.
5. **Evaluación Base:** Calcula las métricas crudas de la red (Accuracy, Recall, F1, MAE, MSE) sobre el set de prueba.
6. **Aplicación de Kalman:** Evalúa las predicciones en el tiempo pasándolas por un suavizador de Kalman para mitigar el ruido.
7. **Reporte Final:** Genera gráficos visuales y una **tabla comparativa final** enfrentando el rendimiento del modelo base contra 3 configuraciones distintas del filtro de Kalman (ligero, balanceado y fuerte).
