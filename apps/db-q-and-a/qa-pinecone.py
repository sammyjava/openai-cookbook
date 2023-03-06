import pinecone
import openai

## limit the number of contexts we retrieve
limit = 1000

## our OpenAI embedding model
embed_model = "text-embedding-ada-002"

## our Pinecone index
index_name = 'pure-gas'

## connect to our Pinecone index
pinecone.init(api_key="19470b6c-f7d0-4df2-a30c-c394f9dc4ffd",
              environment="us-west1-gcp")
index = pinecone.Index(index_name)

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

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

## Answer questions until blank return
operating = True
while operating:
    print("")
    question = input("Question: ")
    if len(question) > 0:
        prompt = retrieve(question)
        print(complete(prompt))
    else:
        operating = False

