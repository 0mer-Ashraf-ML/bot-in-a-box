## Demo UI

<img width="953" alt="botinbox" src="https://github.com/SenariosAIDev/bot-in-a-box/assets/157577843/debe68ae-4a37-4670-bdfb-2657afcfec1f">


## Introduction

Bot-in-a-Box is a configurable chabot, with a vast amount of options and each one is customizable. The main functionality is for the user to be able to upload document(s) of their choice and then be able to chat with them. It is a RAG pipeline which consists of three main components:
1. Indexing of Uploaded Documents
2. Retrieval of Relevant Docs
3. Synthesis of Response

The application is built using ```Langchain``` and ```Streamlit```.

## Breaking Down the Components

**1. Document Upload**

The allowed file types in the finished version of the app will be: pdf, txt, word, html, json, markdown. For now only PDFs are allowed.

<img width="237" alt="Screenshot 2024-02-27 110350" src="https://github.com/SenariosAIDev/bot-in-a-box/assets/157577843/4f10f609-2fbc-4955-922f-238c2cfe38df">


**2. Splitter Selection**

Use LangChain to break documents into manageable pieces, as LLMs have input size limitations. Splitting consists of the type of splitter being used and splitting hyperparameters.

- CharacterTextSplit: This splits based on Separator(by default “”) and measures chunk length by number of characters.
- RecursiveTextSplit: The recursive text splitter works by recursively splitting text into smaller chunks based on a list of separators
- TokenTextSplit: Splitting is done by a number of tokens for which any tokenizer can be used. E.g NLTK, Spacy, and tiktoken.
- Markdown Split: This will split a markdown file by a specified set of headers.
- Semantic Chunking: Splits the text based on semantic similarity; this splits into sentences, then groups into groups of 3 sentences, and then merges one that is similar in the embedding space.

For now only the first three are implemented.

<img width="218" alt="Screenshot 2024-02-27 110509" src="https://github.com/SenariosAIDev/bot-in-a-box/assets/157577843/0dfbe42e-c6fd-4887-b17b-3504ab9ff48f">


**3. Embedding Model Selection**

Convert the chunks into numerical representations via embedding models and store them in a vector database for retrieval. The chunk size and overlap options are configurable as well. The embedding models will be configurable from a wide range of options, inclusing OpenAI Models, Google Palm Model as well as top models from Huggingface MTEB Leaderboard. e.g:

- text-embedding-3-small
- text-embedding-3-large
- text-embedding-ada-002
- Google Palm Embeddings
- bge-base-1.5
- bge-large-1.5

Few models require deploying on own server, so they are not available in the current version of the app.

<img width="220" alt="Screenshot 2024-02-27 110439" src="https://github.com/SenariosAIDev/bot-in-a-box/assets/157577843/ec674a2a-1936-4477-bd45-f96bf32ab24b">


**4. Vector Store Selection**

After embedding the chunks, these vector need to be stored in a Vector Database, for efficient storing and retrieval. The right vectorstore depend on the use case and amount of data. The user can select the vector store from the following:

- FAISS
- ChromaDB
- Pinecone
- Qdrant

Pinecone and Qdrant require self hosting, so for now only FAISS and ChromaDB can be selected.

<img width="228" alt="Screenshot 2024-02-27 110408" src="https://github.com/SenariosAIDev/bot-in-a-box/assets/157577843/b702af53-e6cf-45cc-8dd1-1127d0cc9e24">

**5. Retrieval**

The retriever retrieves document chunks that best match the user's query. There are mainly two types of retrieval: Distance-based retrieval and LLM-aided retrieval. Distance based retrieval then has several choices, including:

- Top k (retrieving the top k number of docs, no matter how similar or disimilar)
- MMR (The MMR selects examples based on a combination of which examples are most similar to the inputs, while also optimizing for diversity)
- Similarity Threshold (setting a threshold for cosine similarity and then retrieving all docs above that threshold)

  <img width="224" alt="Screenshot 2024-02-27 110423" src="https://github.com/SenariosAIDev/bot-in-a-box/assets/157577843/44d0d68a-781e-4a26-86af-9886bfc435f7">


LLM aided retrieval has a few option like:
- Compression
- Map reduce
- Refine
- Map rerank

**6. Generation**

The generation model are the models which synthesize response, from the question and the context. The choice of this model usually depends on the speed-accuracy tradeoff which is acceptable to the user. Another factor is self-hosted model vs API-based model. A final factor can also be context window of the model. Based on these factors, the options for LLMs can be:

- GPT 3.5 Turbo
- GPT 3.5 Turbo 16k
- GPT 4
- GPT 4 Turbo 128k
- Google Palm
- Claude
- LLaMa 2 (7B, 13B)
- Mistral 7B

For now only the top API based models are available, i.e GPT 3.5 Turbo 16k, GPT 4 Turbo 128k and Google Palm

<img width="221" alt="Screenshot 2024-02-27 110455" src="https://github.com/SenariosAIDev/bot-in-a-box/assets/157577843/7e00c85f-41f9-47f1-bda6-22b037efded6">

## Future Improvements

The future development mainly consists of adding the remaining selectable options for more customization. The option for configuring the memory is also under consideration, i.e choice between ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory etc.


