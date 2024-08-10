# Furniture Recommender System [[WebApp Link]](https://lukechugh-furniture-recommender-webapp.streamlit.app/)

![](https://img.shields.io/badge/python-3.7-blueviolet)
![](https://img.shields.io/badge/tensorflow-2.9.0-fuchsia)
![](https://img.shields.io/badge/scikit--learn-0.24.1-blue)
![](https://img.shields.io/badge/streamlit-1.9.1-brightgreen)

This WebApp can recommend chairs, couches/sofa, beds, and tables. The dataset can be found [here](https://www.kaggle.com/competitions/day-3-kaggle-competition/data). 

Docker Container Registry: lukechugh/furniture_recommender_webapp

# How it works ?

Here's an expanded version:

- **Dataset Collection:** 
  - Web-scraped approximately 8,650 images using Selenium Chrome WebDriver.
  - Images categorized into folders: chairs, couches, sofas, beds, and tables.

- **Model Selection:**
  - Tested various CNN architectures (Inception-V3, ResNet50, Xception, MobileNet-V2, VGG-19) to determine the best feature extractor.
  - ResNet50 outperformed others in balanced accuracy and F1 score.

- **Feature Extraction:**
  - Removed the top layer (dense/output layer) of ResNet50 to access convolutional features.
  - Added a GlobalMaxPooling2D layer to capture high-level semantic features across entire spatial dimensions.
  - Converted these features into a 2D vector for each image.

- **Data Preparation:**
  - Merged images from all categories into a single "images" folder.
  - Pickled file paths of all images.
  - Extracted features for each image using the refined ResNet50 architecture.
  - Pickled both the filenames list and the resulting feature embeddings matrix for later use.

- **App Workflow:**
  - When a user uploads an image, ResNet50 extracts features from the uploaded image.
  - KNN algorithm uses Euclidean distance to identify the 6 nearest neighbors among all stored feature embeddings.
  - The first index (potentially the uploaded image itself) is ignored; the remaining 5 images are displayed as recommendations.

- **Deployment:**
  - The web app was built using Streamlit and deployed on the cloud within a Docker container.

# Below are the screenshots of this WebApp:

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/bed.png)

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/chair.png)

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/table.png)

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/couch.png)

# Installation:
To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```
To run this app in your local machine open a command prompt or terminal in the cloned directory and run:
```bash
streamlit run app.py
```
# Author:
[Luke Chugh](https://www.linkedin.com/in/luke-chugh-2b2043181/)
