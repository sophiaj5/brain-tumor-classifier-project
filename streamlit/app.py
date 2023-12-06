#inspiration for this app was used from https://blog.streamlit.io/deep-learning-apps-for-image-processing-made-easy-a-step-by-step-guide/
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title('Brain Tumor Classifier')

#function to load in my model4 to make predictions
def loading_model():
    model = tf.keras.models.load_model('../models/model4.h5')
    return model

with st.sidebar:
    st.header("Welcome to the Brain Tumor Classifier!")
    st.info("Experience the power of advanced medical imaging with our Brain Tumor Analyzer app. Simply upload your brain scan, and our cutting-edge machine learning model will analyze it to provide insights into potential conditions such as glioma, pituitary tumor, meningioma, or no tumor, along with the accuracy level of each diagnosis.")
    st.write("Upload your brain scan to get started!")

file = st.file_uploader('', type=["jpg", "png", "jfif", 'jpeg'])

if file is None:
    st.text('Please upload an image of your brain scan:')
else:
    image_file = file.getvalue()
    st.image(image_file)

    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) #expands the shape of the array on a new dimension
    img_array = img_array / 255

    #load in and call the model
    with st.spinner('Model is loading...'):
        model = loading_model()
    st.success('Model loaded successfully!')

    #make and show predictions
    predictions = np.round(model.predict(img_array) * 100, 2)
    with st.container():
        st.write("Accuracy:", f"{predictions[0, np.argmax(predictions)]:.2f}%")

    #0 = glioma, 1 = meningioma, 2 = no tumor, 3 = pituitary
    tumors = [0, 1, 2, 3]

    if tumors[np.argmax(predictions)] == 0:
        st.write("You most likely have a glioma.")
        st.info("A glioma is a type of cancerous tumor that originates in the glial cells, which play a crucial role in supporting the brain and spinal cord. Gliomas encompass a diverse range of tumor types, each characterized by distinct appearances and requiring tailored treatment approaches. Please consult with your doctor for further treatment options.")
    elif tumors[np.argmax(predictions)] == 1:
        st.write("You most likely have a meningioma.")
        st.info("A meningioma is a tumor that develops within the protective membranes surrounding the brain and spinal cord. Predominantly found in adults, meningiomas come in various types, each distinguished by its appearance. The size and location of the tumor significantly influences symptoms and treatment options. Please consult with your doctor for further treatment options.")
    elif tumors[np.argmax(predictions)] == 2:
        st.write("You most likely do not have a tumor!")
        st.info("Congratulations! You most likely do not have a tumor. Be sure to follow up with your doctor and attend any future check-ups.")
    elif tumors[np.argmax(predictions)] == 3:
        st.write("You most likely have a pituitary tumor.")
        st.info("A pituitary tumor is a growth located in the pituitary gland, situated behind the nose. This unique gland plays a crucial role in regulating various hormones that control essential bodily functions. In certain cases, a pituitary tumor can lead to the overproduction or disruption or these horomoes, which could cause imbalances that affect other parts of the body. Please consult with your doctor for further treatment options.")


#info used from: 
#https://www.cancerresearchuk.org/about-cancer/brain-tumours/types/glioma-adults#:~:text=Gliomas%20are%20cancerous%20brain%20tumours,gliomas%20grow%20faster%20than%20others.
#https://www.brighamandwomens.org/neurosurgery/meningioma#:~:text=Meningiomas%20are%20tumors%20that%20develop,or%20malignant%20meningioma%20(cancerous).
#https://www.hopkinsmedicine.org/health/conditions-and-diseases/pituitary-tumors#:~:text=A%20pituitary%20tumor%20is%20an,are%20not%20cancerous%20(benign).