import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title('Brain Tumor Analyzer')

st.text('Upload an image of your brain scan here:')

f = st.file_uploader("", type=["jpg", "png"])

if f is not None:
    image = Image.open(f)

