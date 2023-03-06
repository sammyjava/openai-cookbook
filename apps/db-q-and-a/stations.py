#!/usr/bin/python
import psycopg2
import pandas as pd
from config import config
import tiktoken
import openai

## Limit the number of stations loaded
num_stations = 1000

def clean_up_text(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('\.\.', '.')
    serie = serie.str.strip()
    return serie

## list to store station records
station_records = []

## list to store the texts
texts=[]

""" Connect to the PostgreSQL database server """
conn = None
try:
    # read connection parameters
    params = config()

    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database...')
    conn = psycopg2.connect(**params)
    
    # create a cursor
    cur = conn.cursor()
    
    ## first query stations that are currently active
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
    query = "SELECT station_id,stationname,streetaddress,city,stateprov,phone,author,comment,timeposted FROM stations WHERE removed IS NULL LIMIT " + str(num_stations)
    print(query)
    cur.execute(query)
    for record in cur:
        station_records.append(record)

    ## Cycle through the stations, gathering updates and loading text.
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

        ## Now query the station updates for the stations we've loaded above.
        # stationupdate_id | integer
        # station_id       | integer
        # timeupdated      | timestamp(0) without time zone
        # author           | character varying
        # comment          | character varying
        # removal          | boolean
        #               0           1      2
        query = "SELECT timeupdated,author,comment FROM stationupdates WHERE station_id=" + str(station_id) + " ORDER BY stationupdate_id"
        cur.execute(query)
        for update_record in cur:
            timeupdated = update_record[0].strftime('%B %-d, %Y')
            author = update_record[1]
            comment = update_record[2]
            blurb += author + " updated the station on " + timeupdated + " with comment: " + comment + ". "
            
        # append the full text for this station
        text = [ str(station_id), blurb]
        texts.append(text)

    ## close the communication with the PostgreSQL
    cur.close()

except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('Database connection closed.')

## Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])
    
## Set the text column to be the cleaned-up raw text
# df['text'] = df.fname + ". " + clean_up_text(df.text)
df['text'] = clean_up_text(df.text)
df.to_csv('processed/scraped.csv')
df.head()

print("Wrote processed/scraped.csv.")

################################################################################
### Step 7
################################################################################

## Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

## Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

## Broken on Wayland?
## Visualize the distribution of the number of tokens per row using a histogram
# df.n_tokens.hist()

################################################################################
### Step 8
################################################################################

max_tokens = 400

## Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')
    
    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
        
    chunks = []
    tokens_so_far = 0
    chunk = []
        
    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):
        
        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

shortened = []

## Loop through the dataframe
for row in df.iterrows():
    ## If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    ## If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['text'])
    
        ## Otherwise, add the text to the list of shortened texts
    else:
        shortened.append( row[1]['text'] )

            
################################################################################
### Step 9
################################################################################

df = pd.DataFrame(shortened, columns = ['text'])
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

## Broken in Wayland?
# df.n_tokens.hist()

################################################################################
### Step 10
################################################################################

## Note that you may run into rate limit issues depending on how many files you try to embed
## Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits

df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()

print("Wrote processed/embeddings.csv")


