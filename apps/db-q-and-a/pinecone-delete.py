#!/usr/bin/python
import pinecone
import os

pinecone.init(
        api_key = os.getenv("PINECONE_API_KEY"),    
        environment = os.getenv("PINECONE_ENVIRONMENT")
)

active_indexes = pinecone.list_indexes()

for index in active_indexes:
    print(pinecone.describe_index(index))
    pinecone.delete_index(index)
    print("DELETED.")
