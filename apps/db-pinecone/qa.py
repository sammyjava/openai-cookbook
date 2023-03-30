import pinecone
import openai
import os

## limit the context length
max_len = 10000

## OpenAI stuff
openai_embed_model = "text-embedding-ada-002"
openai_max_tokens = 400
openai_completion_model = "gpt-3.5-turbo"

## Pinecone stuff
pinecone_index_name = "pure-gas"
pinecone_top_k_value = 25

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
    ## append contexts until hitting max_len
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= max_len:
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
    # {
    #   "choices": [
    #     {
    #       "finish_reason": "stop",
    #       "index": 0,
    #       "message": {
    #         "content": "The following stations in Wisconsin have 93 octane....",
    #         "role": "assistant"
    #       }
    #     }
    #   ],
    #   "created": 1680182767,
    #   "id": "chatcmpl-6zmZzeEPf0b7aZs36yoYH0pAncDVh",
    #   "model": "gpt-3.5-turbo-0301",
    #   "object": "chat.completion",
    #   "usage": {
    #     "completion_tokens": 155,
    #     "prompt_tokens": 2085,
    #     "total_tokens": 2240
    #   }
    # }
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

