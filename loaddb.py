from sqlite_minutils import *
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


db = Database("summaries_20250720_embed.db")
tab = db.table('items')

res=[]
for row in tab.rows:
    emb_bytes=row['embedding']
    if emb_bytes is not None:
        # load float32 array using numpy from bytes
        emb=np.frombuffer(emb_bytes, dtype='float32')
        print(f"{row['identifier']} {emb[0]} {emb[1]} {emb[2]}")
        res.append(emb)

a = np.array(res)

reducer = umap.UMAP()
reducer.fit(a)

embedding = reducer.embedding_

# plt.scatter(embedding[:,0], embedding[:,1], cmap='Spectral', s=5)
# plt.gca().set_aspect('equal','datalim')
# plt.colorbar(boundaries=np.arange(11)-.5).set_ticks(np.arange(10))

umap.plot.points(reducer)
plt.savefig('youtube2.png')
