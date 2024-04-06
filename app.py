import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

model = tf.keras.models.load_model("cnn1.h5")
def predictImage(image):
    try:
        # preprocess the image for the model
        
        # resize to the model's input shape
        img = image.resize((150, 150))  
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # normalize pixel values to [0, 1]
        img_array = img_array / 255.0  

        # make prediction using the trained model
        prediction = model.predict(img_array)[0][0]

        if prediction >= 0.5:
            label = "No Fire"
        else:
            label = "Fire"

        return label
    except Exception as e:
        return str(e)
    
    
def main():
    st.title("Fire Detection using Deep Learning")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        prediction = predictImage(image)
        st.markdown(f"<h1 style='text-align: center; color: black;'>{prediction}</h1>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()