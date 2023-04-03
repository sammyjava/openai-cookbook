import pinecone
import openai
import os

openai_embed_model = "text-embedding-ada-002"
openai_completion_model = "gpt-3.5-turbo"
pinecone_index_name = "legumebot"

pinecone_top_k_value = 100

openai_max_len = 6000
openai_max_tokens = 1800

## connect to our Pinecone index
pinecone.init(
    api_key = os.getenv('PINECONE_API_KEY'),
    environment = os.getenv('PINECONE_ENVIRONMENT')
)
index = pinecone.Index(pinecone_index_name)

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=openai_embed_model
    )

    ## retrieve from Pinecone
    xq = res['data'][0]['embedding']

    ## get relevant contexts
    res = index.query(xq, top_k=pinecone_top_k_value, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    ## build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    ## append contexts until hitting openai_max_len
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= openai_max_len:
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

## query the chat completion model
def complete(prompt):
    res = openai.ChatCompletion.create(
        model=openai_completion_model, 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=openai_max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['message']['content'].strip()

## Answer questions until blank return
operating = True
while operating:
    print("")
    question = input("Question: ")
    if len(question) > 0:
        prompt = retrieve(question)
        print("Answer:")
        print(complete(prompt))
    else:
        operating = False

