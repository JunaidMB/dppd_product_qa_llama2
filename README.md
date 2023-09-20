# Dynamic Personalised Product Display - Product Question and Answer

This repository contains code which uses Retrieval Augmented Generation (RAG) to answer user queries about a product catalogue. The approach uses Cohere's embeddings and rerank model to perform dense retrieval followed by reranking results for relevancy. Once the results have been returned, they are passed as context, in a prompt template, to the ChatGPT API to return a response that answers the user query. This is a clone of the original repository and uses the Llama2 model from Meta as the LLM vs GPT3.5. This script needs a GPU to run because we need to download the LLama2 model locally for inference.

## Areas for Improvement

1. Intergrate external memory: The script initialises a memory variable and makes an input variable in the prompt variable but I haven't implemented a working workflow to update memory and include it in the context. This will be the next milestone. This will also include caching to answer repeat queries.

2. Reduce latency: The response is not fast (~2 seconds). The workflow must be optimised to use appropriate data structures (Polars vs Pandas), make all lists into arrays and explore a faster search index.

3. Include screening for unsuitable queries: Some queries do not need to be entertained by the workflow or may be too simple to require a similarity search - thereby requiring a lookup. Need to add logic before the search workflow to deal with these queries without using the RAG workflow. Another example is when we ask global questions like "How many products are there?" or "what is the cheapest product?". This method is good at retrieving product level information but not good at answering global and cross catalogue levels questions. This is a limitation of the restrictive retrival and rerank parameter settings. Maybe we can construct a metadata structure to contain global data or we can explore tree of thought prompting.

4. Explore ways to reduce hallucination: There is an odd behaviour where the product name is altered to fit a user query (but product price and description is identical). The model fabricates products similar to catalogue products to appeal to the user. Need to address way to control this.

5. Extensive testing of queries and explore RAGAS: Include formalised testing of output. This looks interesting - [RAGAS](https://github.com/explodinggradients/ragas)
   
6. LLama 2 issues: The LLama2 model doesn't answer as succinctly as GPT. If the `max_new_tokens` argument is high, the model will repeat the question in it's answer. Need to read up about how to get the best out of llama2 model. 

7. Experiment with different vector index: Use a vector index that supports reporting source documents.

**Note**: I used pip-tools to create requirements.txt. Use `pip-sync` inside a virtual environment to load dependencies. A Cohere API key is required.

## References

1. [Large Language Models with Semantic Search by Cohere](https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/)
2. [Cohere](https://cohere.com/)
3. [Llama2 Developer's Handbook](https://www.pinecone.io/learn/llama-2/#Building-a-Llama-2-Conversational-Agent)
4. [LLama2 Field Guide](https://github.com/pinecone-io/examples/tree/master/learn/generation/llm-field-guide)
5. [HuggingFace Pipeline](https://api.python.langchain.com/en/latest/llms/langchain.llms.huggingface_text_gen_inference.HuggingFaceTextGenInference.html)
6. [DPPD_Product_QA](https://github.com/JunaidMB/dppd_product_qa/tree/master)