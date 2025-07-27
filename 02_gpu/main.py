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
import re
from typing import Optional
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
            res.append({'example_id': i, 'cluster': cluster, 'summary': row['summary']})
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


def extract_summary_block(text: str) -> Optional[str]:
    """
    Parses a video summary to extract the block containing the abstract and title.

    The function locates the text starting from an "Abstract:" marker and
    ending just before the first timestamped line.

    The input text is expected to contain a structure like:
    ```
    [some kind of introduction]
    <the word abstract, most often Abstract>
    <text_of_abstract>
    [title of the video]
    <list_of_timestamps>
    ```

    Here is a specific example of the expected format:
    ```
    **Abstract:**

    This live stream documents Cherno's ongoing efforts to rewrite the renderer for the Haz ....

    **Summarizing the Hazel Engine Renderer Rewrite (NVRHI Integration)**

    *   **0:00 Renderer Rewrite & NVRHI Backend:** Cherno is actively rewriting the renderer for the Hazel game engine ("Renderer 2025"), migrating from a custom Vulkan implementation to NVIDIA's NVRHI backend. The immediate objective is to get the ImGui user interface up and running with the new system.
    *   **0:53 Platform Support Decisions:** Hazel is planned to support Windows, Linux, and macOS (leveraging MoltenVK for Metal on Mac). Mobile platform support is explicitly excluded due to Cherno's challenging past experiences with mobile game development at EA, including difficult debugging, Samsung driver issues, and slow build times.
    ...
    ```

    Args:
        text: The string containing the video summary.

    Returns:
        A string containing the abstract and title block.
        Returns None if the "Abstract:" marker or a subsequent timestamped
        list cannot be found.
    """
    # 1. Find the start of the content right after the "Abstract:" marker.
    # This pattern looks for "Abstract:" (case-insensitive, with optional markdown)
    # and any whitespace that follows it.
    abstract_marker_regex = r'(?i)(?:\*\*Abstract\*\*|Abstract):\s*'
    marker_match = re.search(abstract_marker_regex, text)

    if not marker_match:
        return None  # The essential "Abstract:" marker was not found.

    # This is the index in the string where the actual content begins.
    content_start_index = marker_match.end()

    # 2. From this starting point, find the beginning of the timestamp list.
    # The search is performed on the rest of the string.
    text_after_marker = text[content_start_index:]

    # Find the start of the first line containing a timestamp. This simplified
    # regex looks for a newline character, followed by any characters (`.*`),
    # and then a time pattern of one or more digits, a colon, and two digits
    # (e.g., "3:34" or "19:00").
    timestamp_marker_regex = r'\n.*\d+:\d{2}'
    timestamp_match = re.search(timestamp_marker_regex, text_after_marker)

    # 3. Extract the block based on whether a timestamp was found.
    if not timestamp_match:
        # If no timestamps are found, assume the entire rest of the text
        # is part of the summary block.
        summary_block = text_after_marker
    else:
        # If timestamps are found, the summary block ends where the
        # timestamp line begins.
        content_end_index = timestamp_match.start()
        summary_block = text_after_marker[:content_end_index]

    return summary_block.strip()

# Iterate over the DataFrame and construct a prompt for each cluster according to this pattern:
# "cluster=5, example_id=<example_id1> [summary1]\ncluster=5, example_id=<example_id2> [summary2] cluster=5, example_id=<example_id3> [summary3]"
# The three examples shall contain the extracted summary and maybe title from the summary (as generated by extract_summary_block).
# If extract_summary_block returns None, and there are more points in the cluster, we will use the next point from the cluster.

examples = []
for cluster, group in df.groupby(level=0):
    # Get the first three examples from the cluster
    example_texts = []
    for i in range(3):
        if i < len(group):
            example = group.iloc[i]
            summary = extract_summary_block(example['summary'])
            if summary:
                example_texts.append(f"cluster={cluster}, example_id={example.name} [{summary}]")
            else:
                continue  # Skip this example if no valid summary was found
        else:
            break  # No more examples in this cluster

    if example_texts:
        examples.append("\n".join(example_texts))

examples_orig = examples.copy() # make a copy of the original examples for debugging, we consume example list in the loop below

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

#prompts = [prompts[0]]

if clusters is None:
    clusters = []
    for i, prompt in enumerate(prompts):
        print(f"Sending prompt {i+1}/{len(prompts)} to Gemini API")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
