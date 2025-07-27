import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import umap.plot
from sqlite_minutils import *

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

debug = False

db = Database("summaries_20250720_embed.db")
tab = db.table('items')
# <Table items (identifier, model, transcript, host, summary, summary_done, summary_input_tokens, summary_output_tokens, summary_timestamp_start, summary_timestamp_end, timestamps, timestamps_done, timestamps_input_tokens, timestamps_output_tokens, timestamps_timestamp_start, timestamps_timestamp_end, timestamped_summary_in_youtube_format, cost, original_source_link, include_comments, include_timestamps, include_glossary, output_language, embedding)>

res = []
res_text = []
res_id = []
res_fulltext = []
for row in tab.rows:
    emb_bytes = row['embedding']
    if emb_bytes is not None:
        # load float32 array using numpy from bytes
        emb = np.frombuffer(emb_bytes, dtype='float32')
        if debug:
            print(f"{row['identifier']} {emb[0]} {emb[1]} {emb[2]}")
        # I only want the first two lines from the summary
        # and I don't want summaries starting with: Error: resource exhausted, Error: value error, emulate
        suma = row['summary'].strip()
        if suma is None:
            continue
        if suma.startswith('Error: resource exhausted') or suma.startswith('Error: value error') or suma.startswith('emulate'):
            continue
        res.append(emb)
        summarylines = suma.split('\n')
        # Delete any title line containing Abstract
        summarylines = [line.strip() for line in summarylines if
                        "abstract" not in line.lower() and "okay, here" not in line.lower() and "here's" not in line.lower() and "here is" not in line.lower()]
        text = " ".join(summarylines).strip()
        text = text[:min(100, len(text))]
        res_text.append({#"id": row['identifier'],
                         "summary": text})
        res_fulltext.append({"summary": row['summary']})
        res_id.append(row['identifier'])
dff = pd.DataFrame(res_fulltext)
dff.to_csv('fulltext.csv', index=False)
dft = pd.DataFrame(res_text)
dft.to_csv('parts.csv', index=True)
a = np.array(res)
# save a to file
np.save('embeddings.npy', a)


# if reducer.pkl exists, load it

reducer = None
reducer_fn = 'reducer.pkl'
try:
    with open(reducer_fn, 'rb') as f:
        f.seek(0)
        reducer = pickle.load(f)
        print('Loaded existing reducer from file')
except FileNotFoundError:
    pass

if reducer is None:
    reducer = umap.UMAP(n_neighbors=4, min_dist=.1)
    print('Will compute UMAP embedding')
    reducer.fit(a)

    with open(reducer_fn, 'wb') as f:
        pickle.dump(reducer, f)  # 188MB

# embedding = reducer.embedding_
# plt.scatter(embedding[:,0], embedding[:,1], cmap='Spectral', s=5)
# plt.gca().set_aspect('equal','datalim')
# plt.colorbar(boundaries=np.arange(11)-.5).set_ticks(np.arange(10))

# umap.plot.points(reducer)
# umap.plot.connectivity(reducer,show_points=True)
# # plt.savefig('youtube3.png')

# dask and scikit-image.
# umap.plot.connectivity(reducer,edge_bundling='hammer')
# plt.savefig('youtube4.png')

# umap.plot.diagnostic(reducer,diagnostic_type='pca')
# plt.savefig('youtube_pca.png')

# umap.plot.diagnostic(reducer,diagnostic_type='vq')
# plt.savefig('youtube_vectorquant.png')

# umap.plot.diagnostic(reducer,diagnostic_type='local_dim')
# plt.savefig('youtube_localdim.png')
# # blue means low local dimension

# umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')
# plt.savefig('youtube_neighborhood.png')

p = umap.plot.interactive(reducer, hover_data=dft, point_size=4, width=1800, height=900)
umap.plot.show(p)
