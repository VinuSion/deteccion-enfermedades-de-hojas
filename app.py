import streamlit as st
import cv2 as cv
import numpy as np
import keras

label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

st.write("""El modelo de detección de enfermedades de hojas de planta se ha construido utilizando técnicas de aprendizaje profundo y emplea el aprendizaje por transferencia para aprovechar el conocimiento preentrenado de un modelo base. El modelo se entrena con un conjunto de datos que contiene imágenes de 33 tipos diferentes de enfermedades en hojas.""")

st.write("Por favor, introduzca solo imágenes de hojas de Manzana, Cereza, Maíz, Uva, Durazno, Pimienta, Papa, Fresa y Tomate. De lo contrario, el modelo no funcionará perfectamente.")

# Load the model
try:
    model = keras.models.load_model('Entrenamiento/modelo/Enfermedades_de_hoja.h5')
    st.success("Modelo cargado exitosamente!")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

uploaded_file = st.file_uploader("Subir Una Imagen")
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        if img is None:
            st.error("Error al decodificar la imagen. Asegúrese de subir un archivo de imagen válido.")
        else:
            normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)

            # Debug: check the shape of the input image
            st.write(f"Shape of the input image: {normalized_image.shape}")

            predictions = model.predict(normalized_image)
            st.image(image_bytes)
            if predictions[0][np.argmax(predictions)]*100 >= 80:
                st.write(f"El resultado es: {label_name[np.argmax(predictions)]}")
            else:
                st.write("La prediccion de esta imagen es menor a 80%. Pruebe otra imagen con un tipo de las mencionadas anteriormente.")
    except Exception as e:
        st.error(f"Error procesando la imagen o realizando la predicción: {e}")