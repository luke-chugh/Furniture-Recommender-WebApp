# Furniture Recommender System [[WebApp Link]](https://lukechugh-furniture-recommender-webapp.streamlit.app/)

![](https://img.shields.io/badge/python-3.7-blueviolet)
![](https://img.shields.io/badge/tensorflow-2.9.0-fuchsia)
![](https://img.shields.io/badge/scikit--learn-0.24.1-blue)
![](https://img.shields.io/badge/streamlit-1.9.1-brightgreen)

This WebApp can recommend chairs, couches/sofa, beds, and tables.

Docker Container Registry: lukechugh/furniture_recommender_webapp

# How it works ?
Pretrained ResNet-50 model in **generator.py** was used to extract features of each image in the **images** folder. These features were saved in **embeddings.pkl**. Then these features were fed to **K-Nearest Neighbors** model in **app.py** which by bruteforcing the **euclidean distance** between the image uploaded by the user and all the images in the **images** folder returned the indexes of 5 nearest neighbors of the image inputted by the user on the WebApp. At last the images from the **images** folder corresponding to these 5 indexes were shown as recommendations on the WebApp. The dataset can be found [here](https://www.kaggle.com/competitions/day-3-kaggle-competition/data). 

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
