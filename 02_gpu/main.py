import cudf
from cuml import UMAP
from cuml.cluster import DBSCAN
import numpy as np
# export CUDA_PATH=/opt/cuda/


# load a from file
a = np.load('../01_start/embeddings.npy')
reducer = UMAP(n_neighbors=4, min_dist=.1, n_components=3)
print('Will compute UMAP embedding')
# measure time for fitting
import time
start_time = time.time()
reducer.fit(a)
end_time = time.time()
print(f"UMAP fitting time: {end_time - start_time:.3f} seconds")

# Setup and fit clusters
start_time = time.time()
print('Will compute DBSCAN clustering')
scan = DBSCAN(eps=1.0, min_samples=1)
scan.fit(reducer.embedding_)
end_time = time.time()
print(f"DBSCAN clustering time: {end_time - start_time:.3f} seconds")

def main():
    print("Hello from 02-gpu!")


if __name__ == "__main__":
    main()
