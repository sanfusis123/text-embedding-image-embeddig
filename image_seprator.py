from glob import glob
from image_embeddings import get_image_embeddings
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random


all_images = glob('data/images/unsplash-images-collection/*.jpg')
random.shuffle(all_images)
embeds = []

for image in tqdm(all_images[:10]):
    embeds.append(get_image_embeddings(image))

cluster =  KMeans(n_clusters=3)

cluster.fit(np.array(embeds))

clusters_assignment = cluster.labels_
df = pd.DataFrame({'file_name': all_images[:10], 'cluster': clusters_assignment})

df.to_csv('image_classes.csv', index=False)
