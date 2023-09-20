import os
from pprint import pprint
import datetime
from dotenv import load_dotenv
import numpy as np
import json
import cohere
from annoy import AnnoyIndex
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from torch import cuda, bfloat16
import accelerate
from typing import List
from langchain.memory import ConversationBufferMemory

load_dotenv(override=True)

# Initialise Cohere Client
cohere_apikey = os.environ.get("COHERE_APIKEY")
hf_auth = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
co = cohere.Client(cohere_apikey)

# Load Product Descriptions
with open("product_descriptions.json", "r") as openfile:
    product_descriptions = json.load(openfile)

# Reformat JSON so all information is in a single string of text for each product
def reformat_dict(x): 
    return f"The product name is {x['product_name']}. The product type is {x['product_type']}. The product price in pounds is {x['product_price_in_pounds']}. {x['product_description']}"

product_description_list = [reformat_dict(prod) for prod in product_descriptions]

# Use Cohere's embeddings to embed the product descriptions
product_description_embeddings = co.embed(texts = product_description_list, model='embed-english-v2.0').embeddings

# Create an Index to store embeddings
search_index = AnnoyIndex(np.array(product_description_embeddings).shape[1], 'angular')

# Add all vectors to search index
for i in range(len(product_description_embeddings)):
    search_index.add_item(i, product_description_embeddings[i])
search_index.build(10) # 10 trees
search_index.save('product_embedding.ann')

# Select LLM to use
model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)

model.eval()
print(f"Model loaded on {device}")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# Make a Hugging Face Pipeline to generate text
generate_text = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,  # langchain expects full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=30,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# Initialise external memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
memory.input_key="question"
memory.output_key="answer"

# Build prompt to wrap our answers and give context: We use LLama2 special tokens
"""
The Llama 2 chat model was fine-tuned for chat using a specific structure for prompts. This structure relied on four special tokens:

<s>: the beginning of the entire sequence.

<<SYS>>\n: the beginning of the system message.

\n<</SYS>>\n\n: the end of the system message.

[INST]: the beginning of some instructions.

[/INST]
"""

template = """
    <s>[INST] <<SYS>>\n
    You are a helpful chatbot that answers questions accurately about a product.
    A customer will ask questions about a product they are interested in, your role is to use the context supplied to give the customer the information they require about a product.
    If you don't know the answer, say that you don't know, don't try to make up an answer.
    If the product does not exist, mention that you have no product that meets a customer's requirements.
    Keep the answer as concise as possible.
    \n<</SYS>>\n\n
    Use the following pieces of context to answer the question at the end.
    {}
    We also provide a conversation history to see what a customer has asked before
    {}
    [\INST]
    Question: {}
    Helpful Answer: 
    """

# Define a function to embed a user query, obtain nearest neighbours, rerank results and use an LLM to answer questions about the product
def dense_retrieval_and_rerank(query, rel_score_threshold = 0.75, memory=memory, prompt_template=template):
    
    # Dense retrieval of user query
    query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings

    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 10, include_distances=True)
    text_ids = similar_item_ids[0]
    similarity_score = similar_item_ids[1]

    similar_items_df = pd.DataFrame({"text_ids": text_ids, "similarity_score": similarity_score})
    text_df = pd.DataFrame({'text': product_description_list, 'text_ids': list(range(0, len(product_description_list)))})

    similar_text_df = pd.merge(similar_items_df, text_df, on = ["text_ids"], how = "left")

    retrieved_product_descriptions_list = similar_text_df.text.tolist()

    # Perform Rerank
    ## Implement with Rerank
    results = co.rerank(query=query, documents=retrieved_product_descriptions_list, model='rerank-english-v2.0')

    # Filter by relevance score
    text = [i.document['text'] for i in results]
    relevance_scores = [i.relevance_score for i in results]

    relevance_score_df = pd.DataFrame({"text": text, "rerank_relevance_score": relevance_scores})
    relevance_score_df = relevance_score_df.sort_values(by = ['rerank_relevance_score'], ascending = False).reset_index(drop=True)

    rel_score_thres = np.quantile(relevance_scores, q = [rel_score_threshold])

    filtered_results = relevance_score_df[relevance_score_df.rerank_relevance_score >= rel_score_thres[0]]

    # Build prompt to wrap our answers and give context
    context = "\n".join(filtered_results['text'].tolist())

    # Provide input data for the prompt
    input_data = {"context": context, "question": query, "history": memory.chat_memory.messages}

    # Use HF pipeline to generate text
    response = generate_text(prompt_template.format(input_data['context'],
                                             input_data['history'],
                                             input_data['question'])
    )

    return response

# Test with example queries
queries = [
    "I just purchased a Gaming console, what other products do you have to augment my gaming needs?",
    "How much does the Modern Glass Dining Table cost?",
    "I have a sweet tooth, have you got any chocolate? If so, how much does it cost?",
    "How much does the Sony Television cost?",
    "Do you know if the Artisanal Chocolate Truffles are sourced from fair trade chocolate?"
]

query_responses = [dense_retrieval_and_rerank(query=query) for query in queries]

# Write responses to JSON
queries_and_responses = []

for idx, _ in enumerate(query_responses):
    queries_and_responses.append( {"query": queries[idx], "response": query_responses[idx]} )

pprint(queries_and_responses)

with open("queries_and_responses.json", "w") as fp:
    json.dump(queries_and_responses, fp)
