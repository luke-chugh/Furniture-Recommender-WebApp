# Furniture Recommender System

This WebApp can recommend chairs, couches/sofa, beds, and tables.

# How it works ?
Pretrained ResNet-50 model in **generator.py** was used to extract features of each image in the **images** folder. These features were saved in **embeddings.pkl**. Then these features were fed to **K-Nearest Neighbors** model in **app.py** which by bruteforcing the **euclidean distance** between the image uploaded by the user and all the images in the **images** folder returned the indexes of 5 nearest neighbors of the image inputted by the user on the WebApp. At last the images from the **images** folder corresponding to these 5 indexes were shown as recommendations on the WebApp. The dataset can be found [here](https://www.kaggle.com/competitions/day-3-kaggle-competition/data). 

# Below are the screenshots of this WebApp:

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/bed.png)

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/chair.png)

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/table.png)

![Capture](https://github.com/luke-chugh/Furniture-Recommender-WebApp/blob/main/screenshots/couch.png)
