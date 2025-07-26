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

