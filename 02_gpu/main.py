import cudf
from cuml import UMAP
from cuml.cluster import DBSCAN
import numpy as np
import time
import umap.plot
import pandas as pd
from matplotlib import pyplot as plt

# export CUDA_PATH=/opt/cuda/


# load a from file
a = np.load('../01_start/embeddings.npy')
reducer3 = UMAP(n_neighbors=5, min_dist=.1, n_components=4)
reducer2 = UMAP(n_neighbors=12, min_dist=.13, n_components=2)
print('Will compute UMAP embedding')
# measure time for fitting
start_time = time.time()
reducer3.fit(a)
reducer2.fit(a)
end_time = time.time()
print(f"UMAP fitting time: {end_time - start_time:.3f} seconds")

# Setup and fit clusters
start_time = time.time()
print('Will compute DBSCAN clustering')
scan = DBSCAN(eps=.3, min_samples=5)
scan.fit(reducer3.embedding_)
end_time = time.time()
print(f"DBSCAN clustering time: {end_time - start_time:.3f} seconds")

dft= pd.read_csv('../01_start/parts.csv')
dff= pd.read_csv('../01_start/fulltext.csv')
embedding = reducer2.embedding_
plt.scatter(embedding[:,0], embedding[:,1], c=scan.labels_, cmap='Spectral', s=5)
plt.gca().set_aspect('equal','datalim')
plt.colorbar() #boundaries=np.arange(11)-.5).set_ticks(np.arange(10))
plt.savefig('umap_dbscan.png', dpi=300)

dft2 = dft.drop(columns=['Unnamed: 0'])
dft2['cluster'] = scan.labels_

p = umap.plot.interactive(reducer2, labels=scan.labels_, hover_data=dft2, point_size=4, width=800, height=800)
umap.plot.show(p)

# Print information about the clusters
print("Number of clusters found:", len(set(scan.labels_)) - (1 if -1 in scan.labels_ else 0))

# Print clusters sorted by size
clusters, counts = np.unique(scan.labels_, return_counts=True)
for cluster, count in sorted(zip(clusters, counts), key=lambda x: x[1], reverse=True):
    if cluster != -1:  # Exclude noise points
        print(f"Cluster {cluster}: {count} points")

# For each cluster (again sorted by number of entries) print 3 random samples of the fulltext
for cluster, count in sorted(zip(clusters, counts), key=lambda x: x[1], reverse=True):
    if cluster != -1:  # Exclude noise points
        print(f"Cluster {cluster}: {count} points")
        samples = dff[dft2['cluster'] == cluster].sample(n=3, random_state=42)
        for i, row in samples.iterrows():
            print(f"Sample {i}: {row['summary'][:200]}...")
        print()


def main():
    print("Hello from 02-gpu!")


if __name__ == "__main__":
    main()
