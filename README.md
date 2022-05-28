# Furniture Recommender System

#### How it works ?
Pretrained ResNet-50 model in **generator.py** was used to extract features of all the images in the **images** folder and these features were saved in **embeddings.pkl**. Then these features were fed to **K-Nearest Neighbors** model in **app.py** which returned the indexes of 5 nearest neighbors of the image inputted by the user on the WebApp. At last the images from the **images** folder corresponding to these 5 indexes were shown as recommendations on the WebApp
