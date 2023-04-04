#!/usr/bin/python
import tiktoken
import openai
import os
import pinecone
from tqdm.auto import tqdm
from time import sleep

## the data file
datafile_name = "gene-descriptions.txt"

## our OpenAI embedding model
embed_model = "text-embedding-ada-002"

## the pinecone index
pinecone_index_name = 'legumebot'

## limit the number of lines processed
#max_stored_lines = 4000000
max_stored_lines = 100

## start past first line
start_line = 10000

## how many embeddings we create and insert at once
batch_size = 100

def clean_up_text(txt):
    txt = txt.replace("\n", "")
    txt = txt.replace("  ", " ")
    txt = txt.replace("..", ".")
    txt = txt.strip()
    return txt

## list of data
data = []

## load the data from a file
print('Loading ' + datafile_name + '...')
    
file = open(datafile_name, 'r')
count = 0
stored = 0
while True:
    if stored >= max_stored_lines:
        break
    line = file.readline()
    if not line:
        break
    count += 1
    if count > start_line:
        parts = line.split("|")
        if len(parts) >= 4:
            stored += 1
            ident = parts[0]
            genus = parts[1]
            species = parts[2]
            description = clean_up_text(parts[3])
            json = {
                'id': ident,
                'text': "identifier:" + ident + ", genus:" + genus + ", species:" + species + ", description:'" + description + "'",
                'identifier': ident,
                'genus': genus,
                'species': species
            }
            data.append(json)
file.close()

print("Read " + str(count) + " lines and stored " + str(stored) + ".")

## initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key = os.getenv('PINECONE_API_KEY'),
    environment = os.getenv('PINECONE_ENVIRONMENT')
)

## create index if doesn't exist
if pinecone_index_name not in pinecone.list_indexes():
    print("Creating index " + pinecone_index_name + "...")
    pinecone.create_index(
        pinecone_index_name,
        dimension=1536,
        metric='cosine',
        metadata_config={
            'indexed': ['identifier', 'genus', 'species' ]
        }
    )

## connect to index
index = pinecone.Index(pinecone_index_name)

print("Populating pinecone index " + pinecone_index_name + "...")

## create embeddings in batches and upsert to pinecone
for i in tqdm(range(0, len(data), batch_size)):
    # find end of batch
    i_end = min(len(data), i+batch_size)
    meta_batch = data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass

    embeds = [record['embedding'] for record in res['data']]

    ## the pinecone vectors
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    
    ## upsert to Pinecone
    index.upsert(vectors=to_upsert)

## view index stats
index.describe_index_stats()
