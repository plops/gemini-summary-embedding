import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import umap.plot
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

reducer = None
reducer_fn = 'reducer.pkl'
try:
    with open(reducer_fn, 'rb') as f:
        f.seek(0)
        reducer = pickle.load(f)
        print('Loaded existing reducer from file')
except FileNotFoundError:
    exit(-1)

# 2019 Aurelien Geron p.256
scan = DBSCAN(eps=.15, min_samples=5)
scan.fit(reducer.embedding_)


embedding = reducer.embedding_
plt.scatter(embedding[:, 0], embedding[:, 1], c=scan.labels_, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.savefig('cluster.png')

dft= pd.read_csv('parts.csv')

p = umap.plot.interactive(reducer, labels=scan.labels_, hover_data=dft, point_size=4, width=1800, height=900)
umap.plot.show(p)

# >>> scan.labels_
# array([ 0,  1,  1, ..., -1,  2, 11], shape=(4118,))
# >>> scan.core_sample_indices_
# array([   0,    1,    2, ..., 4114, 4116, 4117], shape=(3528,))
# >>> scan.components_
# array([[ 6.9252043, -2.7925692],
#        [ 6.6870832, -3.2808669],
#        [ 6.6851816, -3.292168 ],
#        ...,
#        [ 6.534326 ,  5.427366 ],
#        [10.893396 , 10.08514  ],
#        [ 6.2154083,  3.4041855]], shape=(3528, 2), dtype=float32)

df = pd.read_csv('fulltext.csv')

# scan.labels_
# scan.core_sample_indices_
# scan.components_


# >>> df[scan.labels_==2]
#                                                 summary
# 4     **The Real Reasons Behind Stefan Raab's Comeba...
# 220   **Escaping the "Woke" Mindset: An Interview wi...
# 375   **Exploring Potential Political Shifts: Nigel ...
# 394   **The UK's Overqualification Crisis: A Summary...
# 435   **"Eat the Rich!": A Critical Look at Social I...
# ...                                                 ...
# 3909  **Abstract:**\n\nThis video presents a detaile...
# 3951  **Abstract:**\n\nThis video is a deep-dive inv...
# 3983  **Abstract:**\n\nThis video explains the key m...
# 4023  **Abstract:**\n\nThis transcript consists of a...
# 4116  **Abstract:**\n\nThis video by Gary's Economic...



