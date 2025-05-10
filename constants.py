from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# Data preparation
TEXT_SPLITTERS = ('Recursive Character Text Splitter', 'Character Text Splitter', 'Token Text Splitter', 'Semantic Splitter')
DEFAULT_TEXT_SPLITTER = TEXT_SPLITTERS[0]

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

EMBEDDING_MODELS = ('text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-003')
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODELS[0]

VECTOR_STORES = ("faiss", "chroma")
DEFAULT_VECTOR_STORE = VECTOR_STORES[0]



# Retrieval
PROMPT_FOR_REPHRASE_QUERY = """Given the following conversation and a follow up question, rephrase 
the follow up question to be a standalone question, in its original language. The standalone question should be complete, self-sufficient and enough
to answer the intended question without any previous memory/context. If the input is a greeting not a question, then there is no need to rephrase it. In this case, return the original unchanged greeting.
If the query is not related to previous memory then return the query as is.

Chat History:

{memory}
Follow Up Input: {query}
Standalone question: """
PROMPT_FOR_REPHRASE_QUERY_VARIABLES = ('memory', 'query')

RETRIEVAL_SEARCH_TYPE = ("top_k", "Similarity score threshold", "mmr")
DEFAULT_RETRIEVAL_SEARCH_TYPE = RETRIEVAL_SEARCH_TYPE[0]

DEFAULT_RETRIEVER_TOP_K = 3
DEFAULT_RETRIEVER_SIMILARITY_SCORE_THRESHOLD = 0.5



# Generation
PROMPT_FOR_GENERATION = """Your task is being a friendly assistant that specializes in providing answers derived from  a given context. \
If the query is greeting, you should answer with an appropriate response to the greeting. \
If an answer cannot be provided to the query from the context, Don't make up answers by your own from any external source. Instead, just say, \
"I don't have enough information about this. Can I help you with something else?" \

If the query asks for steps on how to perform a specific task, make sure to provide detailed and comprehensive steps from the provided context.

Context: ```{context}```

Query: ```{query}```
Helpful Answer: """
PROMPT_FOR_GENERATION_VARIABLES = ('context', 'query')

MEMORY_TYPES = ('ConversationBufferMemory', 'ConversationBufferWindowMemory', 'ConversationTokenBufferMemory', 'ConversationSummaryMemory')
DEFAULT_MEMORY_TYPE = MEMORY_TYPES[0]
MEMORY_LAST_K = 5
MEMORY_MAX_TOKENS = 2000

# Summary

PROMPT_FOR_SUMMARY = '''Give a summary of the following piece of text/steps from Solar Inverter OEMs to make it brief but sufficiently detailed and easy to follow.
The output style should be consistent to input. No important information should be omitted when making it shorter.

{response}
'''
PROMPT_FOR_SUMMARY_VARIABLES = ('response')

 
LLMS = ('gpt-3.5-turbo', 'gpt-4o-mini','gpt-4.1-nano')
DEFAULT_LLM = LLMS[0]

RETRIEVAL_METHOD = ("Vector store-backed", "MultiQueryRetriever", "Contextual compression")
DEFAULT_RETRIEVAL_METHOD = RETRIEVAL_METHOD[0]

llm_s = ChatOpenAI(temperature=0,
                 max_tokens=1000,
                 model='gpt-4.1-nano',
                 request_timeout=120,
                )

llm_r = ChatOpenAI(temperature=0,
                 max_tokens=1000,
                 model='gpt-4o',
                 request_timeout=120,
                )

