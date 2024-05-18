# PROYECTO IA - T02 (William Gomez & Luis Enrique)

# Prediccion de Enfermedades de Hojas de Planta utilizando Deep Learning

Este proyecto es un sistema de detección de enfermedades en hojas de planta que utiliza técnicas de aprendizaje profundo, incluyendo el aprendizaje por transferencia (para trasladar el modelo pre-entrenado) para identificar y clasificar 33 tipos diferentes de enfermedades en hojas. El modelo ha sido entrenado en un gran conjunto de datos de imágenes y está diseñado para ayudar a profesionales y entusiastas de la agricultura a diagnosticar enfermedades de plantas de manera mas rápida y precisa.

## Detalles del Modelo
El modelo de detección de enfermedades en hojas de planta se ha construido utilizando técnicas de aprendizaje profundo y emplea el aprendizaje por transferencia para aprovechar el conocimiento preentrenado de un modelo base. El modelo se entrena con un conjunto de datos que contiene imágenes de 33 tipos diferentes de enfermedades en hojas.

## Ejecutar Localmente

Para utilizar el modelo para la detección de enfermedades de hojas, sigue estos pasos:

1. Asegúrate de tener configurado un entorno de Python con las bibliotecas necesarias instaladas. Puedes utilizar el archivo `requirements.txt` proporcionado para configurar las dependencias necesarias.

```
pip install -r requirements.txt
```

2. Correr app.py

```
streamlit run app.py --server.enableXsrfProtection false
```
