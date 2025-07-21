#!/usr/bin/env python3
#
# /// script
# dependencies = [
#   "sqlite-minutils",
#   "google-genai",
#   "numpy",
# ]
# ///

from sqlite_minutils.db import *
from google import genai
from google.genai import types
import numpy as np  # uv add numpy
import time
import sqlite_minutils  # We use this for the type hint, but sqlite-minutils provides the objects


# --- 1. SETUP ---
# Load the database and GenAI client
db = Database("summaries_20250720.db")
items: sqlite_minutils.db.Table = Table(db, "items")

# Read the Gemini API key from disk
try:
    with open("api_key.txt") as f:
        api_key = f.read().strip()
    client = genai.Client(api_key=api_key)
except FileNotFoundError:
    print(
        "Error: api_key.txt not found. Please create this file with your Gemini API key."
    )
    exit()


# # Check the size distribution of the summaries
# # res=[]
# # for row in db['items'].rows:
# #    res.append(len(row['summary']))
# # sres = sorted(res)
# # print(sres)


# #print(list(db['items'].rows)[0]['summary'])
# # %%


# # <Table items (identifier, model, transcript, host, summary, summary_done, summary_input_tokens, summary_output_tokens, summary_timestamp_start, summary_timestamp_end, timestamps, timestamps_done, timestamps_input_tokens, timestamps_output_tokens, timestamps_timestamp_start, timestamps_timestamp_end, timestamped_summary_in_youtube_format, cost, original_source_link, include_comments, include_timestamps, include_glossary, output_language)>

# # identifier=int, model=str, transcript=str, host=str, original_source_link=str, include_comments=bool, include_timestamps=bool, include_glossary=bool, output_language=str, summary=str, summary_done=bool, summary_input_tokens=int, summary_output_tokens=int, summary_timestamp_start=str, summary_timestamp_end=str, timestamps=str, timestamps_done=bool, timestamps_input_tokens=int, timestamps_output_tokens=int, timestamps_timestamp_start=str, timestamps_timestamp_end=str, timestamped_summary_in_youtube_format=str, cost=float
# # pk= 'identifier'


# # Try to embed two summaries

# sum = list(db['items'].rows)[2312]['summary']
# sum2 = list(db['items'].rows)[2314]['summary']


# result = client.models.embed_content(
#     model="gemini-embedding-001",
#     contents=[sum,sum2]
# )
# print(result.embeddings[0])

# emb = result.embeddings[0]
# emb2 = result.embeddings[1]

# # >>> emb
# # ContentEmbedding(
# #   values=[
# #     0.015374745,
# #     -0.014419341,
# #     0.027334025,
# #     -0.077198684,
# #     -0.0021566735,
# #     <... 3067 more items ...>,
# #   ]
# # )
# # >>> emb.values[0]
# # 0.015374745
# # >>> type(emb.values[0])
# # <class 'float'>
# # >>> emb2
# # ContentEmbedding(
# #   values=[
# #     0.011935941,
# #     -0.05182009,
# #     0.04588157,
# #     -0.06104778,
# #     -0.020045845,
# #     <... 3067 more items ...>,
# #   ]
# # )

# vector = np.array(emb.values, dtype=np.float32)

# # >>> vector
# # array([ 0.01537475, -0.01441934,  0.02733403, ..., -0.00075878,
# #         0.00421721, -0.0207689 ], shape=(3072,), dtype=float32)

# # Store the embeddings in the database

# Pricing $0.15 per 1,000,000 tokens for embedding.

# # Collect all the summaries and their identifiers
# summaries = []
# ids = []
# for row in db['items'].rows:
#     summaries.append(row['summary'])
#     ids.append(row['identifier'])


# --- 2. PREPARE THE DATABASE ---
# Add an 'embedding' column of type BLOB if it doesn't already exist.
# This is the most efficient way to store vector data.
if "embedding" not in items.columns_dict:
    print("Adding 'embedding' column (BLOB) to the 'items' table...")
    items.add_column("embedding", "BLOB")
    print("Column added.")

# --- 3. COLLECT DATA FOR EMBEDDING ---

# Print number of rows in the table
print(f"Total items in the database: {len(list(items.rows))}")


# It's more efficient to only embed summaries for rows that don't have one yet.
# We collect tuples of (identifier, summary)
rows_to_embed = []
for row in items.rows_where(
    "embedding IS NULL AND summary IS NOT NULL AND summary != ''"
):
    rows_to_embed.append((row["identifier"], row["summary"]))

if not rows_to_embed:
    print("No new summaries to embed. All items are up to date.")
    exit()

print(f"Found {len(rows_to_embed)} summaries to embed.")

# Found 2788 summaries to embed.


# --- 4. BATCH EMBEDDING AND DATABASE UPDATE ---
# The Gemini API has a limit of 100 items per request.
# We must process the data in batches.
BATCH_SIZE = 1

for i in range(0, len(rows_to_embed), BATCH_SIZE):
    batch_rows = rows_to_embed[i : i + BATCH_SIZE]
    ids_batch, summaries_batch = zip(*batch_rows)

    print(
        f"\nProcessing batch {i // BATCH_SIZE + 1} of {((len(rows_to_embed) - 1) // BATCH_SIZE) + 1} ({len(summaries_batch)} items)..."
    )

    try:
        # Get embeddings from the Gemini API
        # Using the recommended model and specifying the task_type for better results.
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=list(summaries_batch),
            # Task type is crucial for optimizing embeddings for your specific use case.
            config=types.EmbedContentConfig(task_type="CLUSTERING"),
        )

        # The API returns embeddings in the same order as the input content.
        # Now, update the database rows.
        print("Storing embeddings in the database...")

        # >>> result
        # EmbedContentResponse(
        #   embeddings=[
        #     ContentEmbedding(
        #       values=[
        #         0.007794169,
        #         -0.022866383,
        #         -0.03302118,
        #         0.04320251,
        #         0.0021397904,
        #         <... 763 more items ...>,
        #       ]
        #     ),
        #     ContentEmbedding(
        #       values=[
        #         -0.009899384,
        #         0.010539095,
        #         -0.054872405,
        #         -0.009400869,
        #         0.03453479,
        #         <... 763 more items ...>,
        #       ]
        #     ),
        #     ContentEmbedding(
        #       values=[
        #         -0.019019835,
        #         -0.022392415,
        #         -0.042795807,
        #         -0.013381309,
        #         0.05283021,
        #         <... 763 more items ...>,
        #       ]
        #     ),

        # The result object has an 'embeddings' attribute which is a list.
        # We zip the original IDs with this list of embedding objects.
        for identifier, embedding_obj in zip(ids_batch, result.embeddings):
            # Convert the list of floats from embedding_obj.values to a
            # numpy array of float32, then to raw bytes (BLOB)
            vector_blob = np.array(embedding_obj.values, dtype=np.float32).tobytes()

            # Use the .update() method from sqlite-minutils.
            # The first argument is the primary key, the second is a dict of columns to update.
            items.update(identifier, {"embedding": vector_blob})

        print(f"Batch {i // BATCH_SIZE + 1} completed successfully.")

    except Exception as e:
        print(f"An error occurred during batch {i // BATCH_SIZE + 1}: {e}")
        print("Skipping this batch and continuing...")

    # Be a good citizen and respect API rate limits.
    time.sleep(3)

print("\nEmbedding process finished.")


# --- 5. VERIFICATION (Optional) ---
# Let's read one back to prove it worked
print("\n--- Verification ---")
try:
    # Use next() to get the first item from the generator without loading all rows.
    first_item_with_embedding = next(items.rows_where("embedding IS NOT NULL"))
    pk = first_item_with_embedding["identifier"]
    embedding_blob = first_item_with_embedding["embedding"]

    # Convert the blob back to a numpy array
    retrieved_vector = np.frombuffer(embedding_blob, dtype=np.float32)

    print(f"Successfully retrieved embedding for item with identifier: {pk}")
    print(
        f"Data type of stored value in DB: {type(embedding_blob)}"
    )  # Should be <class 'bytes'>
    print(f"Shape of decoded vector: {retrieved_vector.shape}")
    print(f"First 5 values of vector: {retrieved_vector[:5]}")

except StopIteration:
    print("Could not find any items with embeddings to verify.")
