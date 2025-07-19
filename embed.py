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


# %% Load the database with video summaries
db = Database("out.db") 

#%%
#for row in db['items'].rows:
#    print(row['summary'])
# %%

print(list(db['items'].rows)[0]['summary'])
# %%

sum = list(db['items'].rows)[2312]['summary']

# Read the gemini api key from disk
with open("api_key.txt") as f:
    api_key=f.read().strip()
client = genai.Client(api_key=api_key)

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=[sum]
)
print(result.embeddings[0])