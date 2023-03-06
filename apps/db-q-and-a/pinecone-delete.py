#!/usr/bin/python
import pinecone

pinecone.init(api_key="19470b6c-f7d0-4df2-a30c-c394f9dc4ffd",
              environment="us-west1-gcp")

active_indexes = pinecone.list_indexes()

for index in active_indexes:
    print(pinecone.describe_index(index))
    pinecone.delete_index(index)
    print("DELETED.")
    
    

