from glob import glob

import numpy as np

from helper_func import get_the_file_encoding
from sklearn.cluster import KMeans
import random
import pandas as pd
from tqdm import  tqdm

all_files = glob('data/data/data/*/*.pdf')
random.shuffle(all_files)

cluster = KMeans(n_clusters=3)
embeds = []
for f_name in tqdm(all_files[:10]):
    embed = get_the_file_encoding(f_name)
    embeds.append(embed)

df = pd.DataFrame({'file_name': all_files[:10], 'embeddings': embeds})

df.to_csv('data.csv', index=False)

cluster.fit(np.array(embeds))
clusters_assignment = cluster.labels_
df1 = pd.DataFrame({'file_name': all_files[:10], 'cluster': clusters_assignment})
df1.to_csv('clusterwith_id.csv', index = False)

