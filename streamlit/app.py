#inspiration for this app was used from https://blog.streamlit.io/deep-learning-apps-for-image-processing-made-easy-a-step-by-step-guide/
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

st.title('Brain Tumor Classifier')

def load_model():
    model = tf.keras.models.load_model('../models/model4.h5')
    return model

with st.sidebar:
    st.write("hi")

file = st.file_uploader('', type=["jpg", "png", "jfif", 'jpeg'])

if file is None:
    st.text('Please upload an image of your brain scan:')
else:
    image_file = file.getvalue()
    st.image(image_file)

    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) #
    img_array = img_array / 255
    with st.spinner('Model is being loaded..'):
        model = load_model()
    st.success('Model loaded successfully!')

    predictions = np.round(model.predict(img_array) * 100, 2)

    with st.container():
        st.write("Accuracy:", f"{predictions[0, np.argmax(predictions)]:.2f}%")

    #0 = glioma, 1 = meningioma, 2 = no tumor, 3 = pituitary
    tumors = [0, 1, 2, 3]

    if tumors[np.argmax(predictions)] == 0:
        st.write("You most likely have a glioma.")
        st.info("information here")
    elif tumors[np.argmax(predictions)] == 1:
        st.write("You most likely have a meningioma.")
        st.info("information here")
    elif tumors[np.argmax(predictions)] == 2:
        #st.balloons()
        st.write("You most likely do not have a tumor!")
        st.info("information here")
    elif tumors[np.argmax(predictions)] == 3:
        st.write("You most likely have a pituitary tumor.")
        st.info("information here")

