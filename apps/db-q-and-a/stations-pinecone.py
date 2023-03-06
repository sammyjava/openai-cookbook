#!/usr/bin/python
import psycopg2
import tiktoken
import openai
import pinecone
from tqdm.auto import tqdm
from time import sleep

## Limit the number of stations loaded
num_stations = 20000

## our OpenAI embedding model
embed_model = "text-embedding-ada-002"

## how many embeddings we create and insert at once
batch_size = 100

def clean_up_text(txt):
    txt = txt.replace("\n", "")
    txt = txt.replace("  ", " ")
    txt = txt.replace("..", ".")
    txt = txt.strip()
    return txt

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

## list of data
data = []

## load the data
conn = None
try:
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database...')
    conn = psycopg2.connect("dbname=puregas user=shokin")
    
    # create a cursor
    cur = conn.cursor()
    
    ## list to store station records
    station_records = []

    ## query stations that are currently active
    # station_id    | integer                        |           | not null | 
    # brand_id      | integer                        |           | not null | 
    # stationname   | character varying              |           | not null | 
    # streetaddress | character varying              |           |          | 
    # city          | character varying              |           | not null | 
    # stateprov     | character(2)                   |           | not null | 
    # latitude      | numeric(8,5)                   |           |          | 
    # longitude     | numeric(8,5)                   |           |          | 
    # timeposted    | timestamp(0) without time zone |           | not null | now()
    # author        | character varying              |           | not null | 
    # comment       | character varying              |           |          | 
    # gpscomment    | character varying              |           |          | 
    # removed       | timestamp(0) without time zone |           |          | 
    # phone         | character varying              |           |          | 
    #               0          1           2             3    4         5     6      7       8          
    query = "SELECT station_id,stationname,streetaddress,city,stateprov,phone,author,comment,timeposted FROM stations WHERE removed IS NULL ORDER BY station_id LIMIT " + str(num_stations)
    print(query)
    cur.execute(query)
    for record in cur:
        station_records.append(record)

    ## Cycle through the station records, gathering octanes and updates, and loading text.
    for station_record in station_records:
        station_id = station_record[0]
        stationname = station_record[1]
        streetaddress = station_record[2]
        city = station_record[3]
        stateprov = station_record[4]
        phone = station_record[5]
        author = station_record[6]
        comment = station_record[7]
        timeposted = station_record[8].strftime('%B %-d, %Y')
        # start the full text blurb
        blurb = stationname + " on " + streetaddress + " in " + city + ", " + stateprov + " "
        if phone:
            blurb += "(" + phone + ") "
        blurb += "was posted by " + author + " on " + timeposted
        if comment:
            blurb += " with comment: " + comment + ". "
        else:
            blurb += " with no comment. "

        ## Get the station's octanes
        # station_id | integer
        # octane     | integer
        #               0
        query = "SELECT octane FROM stationoctanes WHERE station_id=" + str(station_id) + " ORDER BY octane"
        cur.execute(query)
        first = True
        for octane_record in cur:
            octane = octane_record[0]
            if first:
                blurb += str(octane)
                first = False
            else:
                blurb += ", " + str(octane)
        if not first:
            blurb += " octane. "

        ## Now query the most recent station update for the stations we've loaded above.
        # stationupdate_id | integer
        # station_id       | integer
        # timeupdated      | timestamp(0) without time zone
        # author           | character varying
        # comment          | character varying
        # removal          | boolean
        #               0           1      2
        query = "SELECT timeupdated,author,comment FROM stationupdates WHERE station_id=" + str(station_id) + " ORDER BY stationupdate_id DESC"
        cur.execute(query)
        if cur.rowcount:
            update_record = cur.fetchone()
            timeupdated = update_record[0].strftime('%B %-d, %Y')
            author = update_record[1]
            comment = update_record[2]
            blurb += author + " updated the station on " + timeupdated + " with comment: " + comment + ". "

        # append the full text for this station
        data.append({
            'id': str(station_id),
            'state': stateprov,
            'text': clean_up_text(blurb)
        })

    ## close the communication with the PostgreSQL
    cur.close()

except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('Database connection closed.')


## initialize connection to pinecone (get API key at app.pinecone.io)
index_name = 'pure-gas'
pinecone.init(api_key="19470b6c-f7d0-4df2-a30c-c394f9dc4ffd",
              environment="us-west1-gcp")

## create index if doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        metadata_config={'indexed': ['state']}
    )

## connect to index
index = pinecone.Index(index_name)

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




