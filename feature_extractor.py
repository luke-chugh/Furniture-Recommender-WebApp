import tensorflow as tf
import numpy as np
import re
import os
import pickle
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from numpy.linalg import norm

# Load the ResNet50 model pre-trained on ImageNet, excluding the top (classification) layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze the base model layers to prevent them from being trained

# Create a new model using the Functional API, adding a GlobalAveragePooling2D layer on top of the base model
inputs = Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = x
model = tf.keras.Model(inputs, outputs)

# Function to extract features from an image using the model
def extract_features(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    preprocessed_img = tf.keras.applications.resnet.preprocess_input(expanded_img_array)
    
    # Get the feature vector by passing the image through the model
    result = model.predict(preprocessed_img).flatten()
    
    # Normalize the feature vector to have unit length
    normalized_result = result / norm(result)
    
    return normalized_result

# Function to sort filenames numerically (to ensure correct order)
def numerical_sort(value):
    match = re.search(r'\d+', value)  # Extract the numeric part of the filename
    return int(match.group()) if match else 0  # Return the number as the sorting key

# Collect all image filenames from the 'images' directory
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# Sort the filenames numerically (so 'img_2' comes before 'img_10')
filenames.sort(key=numerical_sort)
filenames = [path.replace('\\', '/') for path in filenames]

# Extract features for each image and store them in a list
feature_list = []
for file in tqdm(filenames):  # tqdm provides a progress bar for the loop
    feature_list.append(extract_features(file, model))

# Save the extracted features and corresponding filenames using pickle
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
