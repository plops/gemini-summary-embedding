#!/usr/bin/env python3
#
# /// script
# dependencies = [
#   "sqlite-minutils",
#   "google-genai",
# ]
# ///

from sqlite_minutils.db import *
from google import genai
from google.genai import types
import numpy as np # uv add numpy


# %% Load the database with video summaries. The largest are 27221, 86034, 91174. I think that is fine.
db = Database("out.db") 

# Check the size distribution of the summaries
# res=[]
# for row in db['items'].rows:
#    res.append(len(row['summary']))
# sres = sorted(res)
# print(sres)


#print(list(db['items'].rows)[0]['summary'])
# %%

# 
items = Table(db, 'items')

# <Table items (identifier, model, transcript, host, summary, summary_done, summary_input_tokens, summary_output_tokens, summary_timestamp_start, summary_timestamp_end, timestamps, timestamps_done, timestamps_input_tokens, timestamps_output_tokens, timestamps_timestamp_start, timestamps_timestamp_end, timestamped_summary_in_youtube_format, cost, original_source_link, include_comments, include_timestamps, include_glossary, output_language)>

# identifier=int, model=str, transcript=str, host=str, original_source_link=str, include_comments=bool, include_timestamps=bool, include_glossary=bool, output_language=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float
# pk= 'identifier'


# Try to embed two summaries

sum = list(db['items'].rows)[2312]['summary']
sum2 = list(db['items'].rows)[2314]['summary']

# Read the gemini api key from disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
client = genai.Client(api_key=api_key)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[sum,sum2]
)
print(result.embeddings[0])

emb = result.embeddings[0]
emb2 = result.embeddings[1]

# >>> emb
# ContentEmbedding(
#   values=[
#     0.015374745,
#     -0.014419341,
#     0.027334025,
#     -0.077198684,
#     -0.0021566735,
#     <... 3067 more items ...>,
#   ]
# )
# >>> emb.values[0]
# 0.015374745
# >>> type(emb.values[0])
# <class 'float'>
# >>> emb2
# ContentEmbedding(
#   values=[
#     0.011935941,
#     -0.05182009,
#     0.04588157,
#     -0.06104778,
#     -0.020045845,
#     <... 3067 more items ...>,
#   ]
# )

vector = np.array(emb.values, dtype=np.float32)

# >>> vector
# array([ 0.01537475, -0.01441934,  0.02733403, ..., -0.00075878,
#         0.00421721, -0.0207689 ], shape=(3072,), dtype=float32)

# Store the embeddings in the database


# Collect all the summaries and their identifiers
summaries = []
ids = []
for row in db['items'].rows:
    summaries.append(row['summary'])
    ids.append(row['identifier'])

