import streamlit as st
import numpy as np
import keras
from PIL import Image

label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

st.write("""El modelo de detección de enfermedades de hojas de planta se ha construido utilizando técnicas de aprendizaje profundo y emplea el aprendizaje por transferencia para aprovechar el conocimiento preentrenado de un modelo base. El modelo se entrena con un conjunto de datos que contiene imágenes de 33 tipos diferentes de enfermedades en hojas.""")

st.write("Por favor, introduzca solo imágenes de hojas de Manzana, Cereza, Maíz, Uva, Durazno, Pimienta, Papa, Fresa y Tomate. De lo contrario, el modelo no funcionará perfectamente.")

model = keras.models.load_model('Entrenamiento/modelo/Enfermedades_de_hoja.h5')

uploaded_file = st.file_uploader("Subir Una Imagen")
if uploaded_file is not None:
    try:
        # Use PIL to open the uploaded image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption='Imagen Adjuntada', use_column_width=True)
        
        # Resize and preprocess the image
        resized_image = image.resize((150, 150))
        normalized_image = np.array(resized_image) / 255.0  # Normalize pixel values to [0, 1]
        normalized_image = np.expand_dims(normalized_image, axis=0)
        
        # Make predictions
        predictions = model.predict(normalized_image)
        
        # Display results
        if predictions[0][np.argmax(predictions)] * 100 >= 80:
            st.write(f"El resultado es: {label_name[np.argmax(predictions)]}")
        else:
            st.write("La predicción de esta imagen es menor al 80%. Pruebe otra imagen con un tipo de las mencionadas anteriormente.")
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")