from sqlite_minutils import *
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

debug = False

db = Database("summaries_20250720_embed.db")
tab = db.table('items')

res=[]
for row in tab.rows:
    emb_bytes=row['embedding']
    if emb_bytes is not None:
        # load float32 array using numpy from bytes
        emb=np.frombuffer(emb_bytes, dtype='float32')
        if debug:
            print(f"{row['identifier']} {emb[0]} {emb[1]} {emb[2]}")
        res.append(emb)

a = np.array(res)

# if reducer.pkl exists, load it

reducer = None
reducer_fn = 'reducer.pkl'
with open(reducer_fn, 'rb') as f:
    f.seek(0)
    reducer = pickle.load(f)
    print('Loaded existing reducer from file')

if reducer is None:
    reducer = umap.UMAP()
    print('Will compute UMAP embedding')
    reducer.fit(a)

    with open(reducer_fn, 'wb') as f:
        pickle.dump(reducer, f) # 188MB



# embedding = reducer.embedding_
# plt.scatter(embedding[:,0], embedding[:,1], cmap='Spectral', s=5)
# plt.gca().set_aspect('equal','datalim')
# plt.colorbar(boundaries=np.arange(11)-.5).set_ticks(np.arange(10))

#umap.plot.points(reducer)
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

umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')
plt.savefig('youtube_neighborhood.png')

# p = umap.plot.interactive(reducer)
# type(p)
# umap.plot.show(p)