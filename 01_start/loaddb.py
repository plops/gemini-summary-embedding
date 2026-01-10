import pickle
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import umap.plot
from sqlite_minutils import *
import loguru

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load embeddings from database and visualize with UMAP')
parser.add_argument('-d', '--db-file',
                    default='/home/kiel/stage/cl-py-generator/example/143_helium_gemini/source04/tsum/data/summaries.db',
                    help='Database file to use (default: summaries.db in the webservers tsum/data)')
parser.add_argument('-e', '--embeddings-file',
                    default='embeddings.npy',
                    help='Output embeddings file (default: embeddings.npy)')
parser.add_argument('-f', '--fulltext-file',
                    default='fulltext.csv',
                    help='Output fulltext CSV file (default: fulltext.csv)')
parser.add_argument('-p', '--parts-file',
                    default='parts.csv',
                    help='Output parts CSV file (default: parts.csv)')
parser.add_argument('-D', '--debug',
                    action='store_true',
                    help='Enable debug mode')
parser.add_argument('-r', '--reducer-file',
                    default='reducer.pkl',
                    help='UMAP reducer file (default: reducer.pkl)')
parser.add_argument('-n', '--n-neighbors',
                    type=int,
                    default=4,
                    help='UMAP n_neighbors parameter (default: 4)')
parser.add_argument('-m', '--min-dist',
                    type=float,
                    default=0.1,
                    help='UMAP min_dist parameter (default: 0.1)')
parser.add_argument('-P', '--no-plot',
                    action='store_true',
                    help='Disable interactive plotting')
parser.add_argument('-x', '--exclude-errors',
                    action='store_true',
                    help='Exclude summaries starting with errors')
parser.add_argument('-l', '--log-level',
                    default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='Set logging level (default: INFO)')

args = parser.parse_args()

# initialize loguru logger
logger = loguru.logger
logger.remove()  # Remove default handler
logger.add(lambda msg: print(msg, end=''), level=args.log_level)

sns.set(style="white", context="notebook", rc={"figure.figsize": (14, 10)})

debug = args.debug

db = Database(args.db_file)
tab = db.table("items")
# <Table items (identifier, model, transcript, host, summary, summary_done, summary_input_tokens, summary_output_tokens, summary_timestamp_start, summary_timestamp_end, timestamps, timestamps_done, timestamps_input_tokens, timestamps_output_tokens, timestamps_timestamp_start, timestamps_timestamp_end, timestamped_summary_in_youtube_format, cost, original_source_link, include_comments, include_timestamps, include_glossary, output_language, embedding, full_embedding)>

# print the number of rows in tab using logger
logger.info(f"Number of rows in tab: {tab.count}")

res = [] # list of embeddings in numpy array format (float32)
res_text = [] # list of dicts with 'summary' key (with first two lines of the AI summary only)
res_id = [] # list of identifiers
res_fulltext = [] # list of dicts with full 'summary' key (with the full AI summary)
for row in tab.rows:
    emb_bytes = row["embedding"]
    if emb_bytes is not None:
        # load float32 array using numpy from bytes
        emb = np.frombuffer(emb_bytes, dtype="float32")
        if debug:
            print(f"{row['identifier']} {emb[0]} {emb[1]} {emb[2]}")
        # I only want the first two lines from the summary
        # and I don't want summaries starting with: Error: resource exhausted, Error: value error, emulate
        suma = row["summary"].strip()
        if suma is None:
            continue
        if args.exclude_errors and (
            suma.startswith("Error: resource exhausted")
            or suma.startswith("Error: value error")
            or suma.startswith("emulate")
        ):
            continue
        res.append(emb)
        summarylines = suma.split("\n")
        # Delete any title line containing Abstract
        summarylines = [
            line.strip()
            for line in summarylines
            if "abstract" not in line.lower()
            and "okay, here" not in line.lower()
            and "here's" not in line.lower()
            and "here is" not in line.lower()
        ]
        text = " ".join(summarylines).strip()
        text = text[: min(100, len(text))]
        res_text.append(
            {  # "id": row['identifier'],
                "summary": text
            }
        )
        res_fulltext.append({"summary": row["summary"]})
        res_id.append(row["identifier"])
dff = pd.DataFrame(res_fulltext) # full summaries
dff.to_csv(args.fulltext_file, index=False)
dft = pd.DataFrame(res_text) # truncated summaries
dft.to_csv(args.parts_file, index=True)
a = np.array(res)
# save a to file
np.save(args.embeddings_file, a) # shape (num_entries, embedding_dim)

# print the number of rows in dft (rows with embeddings)
logger.info(f"Number of rows in dft (entries with embeddings): {len(dft)}")

# print the shape of a using logger
logger.info(f"Shape of embeddings array: {a.shape}")


# if reducer.pkl exists, load it

reducer = None
reducer_fn = args.reducer_file
try:
    with open(reducer_fn, "rb") as f:
        f.seek(0)
        reducer = pickle.load(f)
        logger.info("Loaded existing reducer from file")
except FileNotFoundError:
    logger.info("No existing reducer found, will compute a new one")
    pass

if reducer is None:
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    logger.info("Fitting UMAP reducer to data")
    reducer.fit(a)

    with open(reducer_fn, "wb") as f:
        logger.info("Saving reducer to file")
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

# print the number of entries in dft
logger.info("Number of entries in dft: {}", len(dft))

if not args.no_plot:
    p = umap.plot.interactive(reducer, hover_data=dft, point_size=4, width=1800, height=900)
    umap.plot.show(p)
else:
    logger.info("Plotting disabled by --no-plot flag")
