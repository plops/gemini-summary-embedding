# export CUDA_PATH=/opt/cuda/
#
# import cudf
from cuml import UMAP
from cuml.cluster import DBSCAN
import numpy as np
import time
import umap.plot
import pandas as pd
from matplotlib import pyplot as plt
import pickle

from google import genai
from google.genai import types
import pydantic

try:
    with open("../01_start/api_key.txt") as f:
        api_key = f.read().strip()
    client = genai.Client(api_key=api_key)
except FileNotFoundError:
    print("Error: api_key.txt not found. Please create this file with your Gemini API key.")
    exit()



# load embeddings `a` from file. This has been extracted from the sqlite3 database with text and embeddings using
# 01_start/loaddb.py. This python script also creates the files `parts.csv` and `fulltext.csv` which contain
# the first 100 characters and full summaries of the videos.

a = np.load('../01_start/embeddings.npy')
dft= pd.read_csv('../01_start/parts.csv')
dff= pd.read_csv('../01_start/fulltext.csv')

reducers = None
reducer_fn = 'reducer.pkl'
try:
    with open(reducer_fn, 'rb') as f:
        f.seek(0)
        reducers = pickle.load(f)
        reducer2, reducer3, scan = reducers
        print('Loaded existing UMAP reducer objects and DBSCAN clustering object from file')
except FileNotFoundError:
    pass

if reducers is None:
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

    with open(reducer_fn, 'wb') as f:
        reducers = [reducer2, reducer3, scan]
        pickle.dump(reducers, f) # 98MB



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
# for cluster, count in sorted(zip(clusters, counts), key=lambda x: x[1], reverse=True):
#     if cluster != -1:  # Exclude noise points
#          print(f"Cluster {cluster}: {count} points")

# For each cluster (again sorted by number of entries) print 3 random samples of the fulltext
res = []
for cluster, count in sorted(zip(clusters, counts), key=lambda x: x[1], reverse=True):
    if cluster != -1:  # Exclude noise points
        # print(f"Cluster {cluster}: {count} points")
        samples = dff[dft2['cluster'] == cluster].sample(n=3, random_state=42)
        for i, row in samples.iterrows():
            res.append({'cluster': cluster, 'summary': row['summary']})
            # print(f"Sample {i}: {row['summary'][:200]}...")
        # print()

df = pd.DataFrame(res)
# >>> df
#      cluster                                            summary
# 0          5  **Abstract:**\n\nThis video explores the persi...
# 1          5  **Abstract:**\n\nThis video critically examine...
# 2          5  **Abstract:**\n\nThis video features Sergey, a...
# 3         14  **Abstract:**\n\nThis video delves into the pr...
# 4         14  **Key Points from the Discussion on the Gaza C...
# ..       ...                                                ...
# 508      168  **Abstract:**\n\nThis video compiles heartfelt...
# 509      168  **Abstract:**\n\nThis report highlights the pe...
# 510      170  **Abstract:**\n\nThis lecture, Part 2 of a two...
# 511      170  **Abstract:**\n\nThis lecture, the second part...
# 512      170  **Abstract:**\n\nThis lecture, Part 2 of a two...

# Add cluster to the DataFrame index
df.set_index('cluster', inplace=True)

# Iterate over the DataFrame and construct a prompt for each cluster according to this pattern:
# "Cluster 5: [summary1] [summary2] [summary3]"
examples = []
for cluster, group in df.groupby(df.index):
    summaries = " ".join(group['summary'].tolist())
    examples.append(f"Cluster {cluster}: {summaries}")


prompt0 = "I have a embedding visualization of Youtube video summaries. The embeddings are displayed as a 2D map where every video is one point. Clusters of points were identified using DBSCAN. You will see three examples of summaries that were randomly taken from a cluster. Generate a title for each cluster that can be shown in the diagram. Make sure that the response contains only one title for each cluster that describes all three examples reasonably well."
prompts = []

# We will call the Gemini API multiple times. Each time with a different prompt from `prompts`.
# Construct each prompt by starting with `prompt0` and adding data from prompts[]. Stop adding to a prompt when it
# contains more than 20000 words

def words (prompt):
    return len(prompt.split())

prompt = prompt0 + "\n\n"

while words(prompt) < 20000 and examples:
    example = examples.pop(0)
    if words(prompt + example) < 20000:
        prompt += example + "\n\n"
    else:
        prompts.append(prompt.strip())
        prompt = prompt0 + "\n\n" + example + "\n\n"


# Print the number of prompts we will send to the Gemini API
print(f"Number of prompts to send to Gemini API: {len(prompts)}")

class Cluster(pydantic.BaseModel):
    title: str
    id: int

# We will store the results in a list of Cluster objects as they come
clusters = None
cluster_fn = 'clusters.pkl'

try:
    with open(cluster_fn, 'rb') as f:
        f.seek(0)
        clusters = pickle.load(f)
        print('Loaded existing clusters from file')
except FileNotFoundError:
    pass

if clusters is None:
    clusters = []
    for i, prompt in enumerate(prompts):
        print(f"Sending prompt {i+1}/{len(prompts)} to Gemini API")
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config={"response_mime_type": "application/json",
                    "response_schema": list[Cluster],
                    },
        )
        print(response.parsed)
        clusters.extend(response.parsed)
        with open(cluster_fn, 'wb') as f:
            pickle.dump(clusters, f)
        print(f"Saved {len(clusters)} clusters to file")



def main():
    print("Hello from 02-gpu!")


if __name__ == "__main__":
    main()
