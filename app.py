import re
import os
import pickle
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import streamlit as st
import tensorflow as tf
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Input

st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the ResNet50 model pre-trained on ImageNet, excluding the top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False 
inputs = Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = x
model = tf.keras.Model(inputs, outputs)

# Load the data
try:
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as k:
        filenames = pickle.load(k)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()
    
st.markdown("<h3 style='text-align: center; color: blue;'>FURNITURE RECOMMENDATION SYSTEM BY LUKE CHUGH</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: black;'>Note: This WebApp can only recommend chairs, couches/sofa, beds, and tables</h5>", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)  
    preprocessed_img = tf.keras.applications.resnet.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result) 
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

uploaded_file = st.file_uploader("Upload an image and wait for a few seconds.", type=["jpg", "png", "jpeg", "webp"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the file
        display_image = Image.open(uploaded_file)
        resized_image = display_image.resize((300, 300))
        st.write('#### **Uploaded Image:**')
        st.image(resized_image)
        st.write('#### **Recommendations:**')
        with st.spinner('Preparing Recommendations..'):
            # Feature extraction
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
            # Recommendation
            indices = recommend(features, feature_list)
            # Show Recommendations
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(Image.open(filenames[indices[0][1]]).resize((300, 300)))
            with col2:
                st.image(Image.open(filenames[indices[0][2]]).resize((300, 300)))
            with col3:
                st.image(Image.open(filenames[indices[0][3]]).resize((300, 300)))
            with col4:
                st.image(Image.open(filenames[indices[0][4]]).resize((300, 300)))
            with col5:
                st.image(Image.open(filenames[indices[0][5]]).resize((300, 300)))
        # Clean up uploaded files
        for file in os.listdir('uploads'):
            os.remove(os.path.join('uploads', file))
    else:
        st.header("Some error occurred in file upload")