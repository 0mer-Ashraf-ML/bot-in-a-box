import json
import os
import shutil
from uuid import uuid1
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import glob
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from datetime import datetime
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_community.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import constants
from constants import llm_s,llm_r


class Chatbot:
    def __init__(
            self,
            documents,
            text_splitter=constants.DEFAULT_TEXT_SPLITTER,
            chunk_size=constants.CHUNK_SIZE,
            chunk_overlap=constants.CHUNK_OVERLAP,
            embedding_model=constants.DEFAULT_EMBEDDING_MODEL,
            vector_store=constants.DEFAULT_VECTOR_STORE,
            prompt_for_rephrase_query=constants.PROMPT_FOR_REPHRASE_QUERY,
            prompt_for_rephrase_query_variables=constants.PROMPT_FOR_REPHRASE_QUERY_VARIABLES,
            prompt_for_summary = constants.PROMPT_FOR_SUMMARY,
            prompt_for_summary_variables = constants.PROMPT_FOR_SUMMARY_VARIABLES,
            retriever_method=constants.DEFAULT_RETRIEVAL_METHOD,
            retriever_search_type=constants.DEFAULT_RETRIEVAL_SEARCH_TYPE,
            retriever_top_k=constants.DEFAULT_RETRIEVER_TOP_K,
            retriever_similarity_score_threshold=constants.DEFAULT_RETRIEVER_SIMILARITY_SCORE_THRESHOLD,
            prompt_for_generation=constants.PROMPT_FOR_GENERATION,
            prompt_for_generation_variables=constants.PROMPT_FOR_GENERATION_VARIABLES,
            memory_type=constants.DEFAULT_MEMORY_TYPE,
            memory_last_k=constants.MEMORY_LAST_K,
            memory_max_tokens=constants.MEMORY_MAX_TOKENS,
            llm_model=constants.DEFAULT_LLM,
            verbose=False,
            load_chatbot_from_dir=None,
            memory_filename = f'memory_at_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.json',
            openai_api_key=None
        ):
        self.documents = documents
        self.text_splitter_name = text_splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        self.vector_store_name = vector_store
        self.prompt_for_rephrase_query = prompt_for_rephrase_query
        self.prompt_for_rephrase_query_variables = prompt_for_rephrase_query_variables
        self.prompt_for_summary = prompt_for_summary
        self.prompt_for_summary_variables = prompt_for_summary_variables
        self.retriever_search_type = retriever_search_type
        self.retriever_top_k=retriever_top_k
        self.retriever_similarity_score_threshold = retriever_similarity_score_threshold
        self.prompt_for_generation = prompt_for_generation
        self.prompt_for_generation_variables = prompt_for_generation_variables
        self.memory_type = memory_type
        self.memory_last_k = memory_last_k
        self.memory_max_tokens = memory_max_tokens
        self.llm_model_name = llm_model
        self.verbose = verbose
        self.load_chatbot_from_dir = load_chatbot_from_dir 
        self.retriever_method = retriever_method           
        self.summary_prompt = PromptTemplate(template=self.prompt_for_summary, input_variables=['response'])
        self.openai_api_key = openai_api_key
        
        # Use custom API key for llm_s and llm_r if provided
        self.llm_s = llm_s
        self.llm_r = llm_r
        if self.openai_api_key and 'gpt' in llm_s.model_name:
            self.llm_s = ChatOpenAI(
                temperature=llm_s.temperature,
                max_tokens=llm_s.max_tokens,
                model=llm_s.model_name,
                request_timeout=llm_s.request_timeout,
                openai_api_key=self.openai_api_key
            )
            self.llm_r = ChatOpenAI(
                temperature=llm_r.temperature,
                max_tokens=llm_r.max_tokens,
                model=llm_r.model_name,
                request_timeout=llm_r.request_timeout,
                openai_api_key=self.openai_api_key
            )
        
        self.summarize_chain = LLMChain(llm=self.llm_s, prompt=self.summary_prompt)
        self.prompt_template = PromptTemplate(template=self.prompt_for_generation,
                                              input_variables=self.prompt_for_generation_variables)
        self.llm = self._get_llm()
        self.vector_store = self._init_vector_store()
        self.retriever = self._get_retriever(self.vector_store, self.retriever_search_type,
                                             self.retriever_top_k, self.retriever_similarity_score_threshold,
                                             self.retriever_method, self.llm)
        self.memory = self._get_memory()
        self.memory_filename = memory_filename
        self.qa_chain = load_qa_chain(self.llm,
                                      chain_type='stuff',
                                      prompt=self.prompt_template,
                                      memory=self.memory)


    def query(self, query):
        if self.memory.chat_memory != '':
            query = self._rephrase_query(query)

        docs, scores = self._get_docs(query)
        with get_openai_callback() as cb:
            res = self.qa_chain.invoke({'input_documents': docs, 'query': query}, return_only_outputs=True)
        output_full = res['output_text'].strip()
        # output = self.summarize_chain.invoke(output_full)['text']
        return output_full, docs, scores
    

    def clear_memory(self):
        self.memory.clear()


    def _get_docs(self, query):
        docs = self.retriever.get_relevant_documents(query)

        query_embed = self.embeddings_model.embed_query(query)
        docs_embed = self.embeddings_model.embed_documents([doc.page_content for doc in docs])

        if len(docs) == 0:
            return [], []

        query_embed = np.array(query_embed).reshape(1, -1)
        docs_embed = np.array(docs_embed)
        scores = cosine_similarity(query_embed, docs_embed)[0]

        return docs, scores


    def _rephrase_query(self, query):
        prompt_temp = PromptTemplate(template=self.prompt_for_rephrase_query,
                                     input_variables=self.prompt_for_rephrase_query_variables)
        prompt_chain = LLMChain(llm=self.llm_r, prompt=prompt_temp, verbose=self.verbose)
        memory  = self.memory.load_memory_variables({})['chat_history']
        rephrased_query = prompt_chain.invoke({'memory': memory, 'query': query})
        print(rephrased_query['text'])
        return rephrased_query['text']


    def _read_pdf_and_make_chunks(self,docs_list):
        if 'Recursive' in self.text_splitter_name:
            self.text_splitter = RecursiveCharacterTextSplitter(separators=[' ','\n'],
                                                                     chunk_size=self.chunk_size,
                                                                     chunk_overlap=self.chunk_overlap)
        elif 'Token' in self.text_splitter_name:
            self.text_splitter = TokenTextSplitter(chunk_size=self.chunk_size,
                                                        chunk_overlap=self.chunk_overlap)
        elif 'Semantic' in self.text_splitter_name:
            self.text_splitter = SemanticChunker(
            self.embeddings_model, breakpoint_threshold_type="percentile")

        else:
            self.text_splitter = CharacterTextSplitter(separator=' ',
                                                            chunk_size=self.chunk_size,
                                                            chunk_overlap=self.chunk_overlap)

        docs = []
        for doc_path in docs_list:
            raw_context = ""
            if doc_path.endswith('txt') or doc_path.endswith('pdf'):
                print('txt/pdf',doc_path)
                loader = PyMuPDFLoader(doc_path)
            
            elif doc_path.endswith('docx') or doc_path.endswith('doc'):
                print('docx',doc_path)
                loader = Docx2txtLoader(doc_path)
            
            documents = loader.load()    
            for page in documents:
                raw_context+=page.page_content+'\n'

            book = [Document(page_content=raw_context,metadata={'name':doc_path})]
            chunks = self.text_splitter.split_documents(book)
            chunk_no = 1
            for chunk in chunks:
                chunk.metadata['Section'] = chunk_no
                docs.append(chunk)
                chunk_no +=1

        return docs


    def _init_vector_store(self):
        if 'Google' in self.embedding_model_name:
            self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        elif 'text-embedding' in self.embedding_model_name:
            if self.openai_api_key:
                self.embeddings_model = OpenAIEmbeddings(model=self.embedding_model_name, openai_api_key=self.openai_api_key)
            else:
                self.embeddings_model = OpenAIEmbeddings(model=self.embedding_model_name)

        if self.load_chatbot_from_dir:
            self.vectorstore_dir = self.load_chatbot_from_dir+'/vectorstore'
            if self.vector_store_name == 'faiss':
                vector_store = FAISS.load_local(self.vectorstore_dir, self.embeddings_model)
            elif self.vector_store_name == 'chroma':
                vector_store = Chroma(persist_directory=self.vectorstore_dir,
                                      embedding_function=self.embeddings_model)
            return vector_store

        chunks = self._read_pdf_and_make_chunks(self.documents)

        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        self.vectorstore_dir = './tmp/'+str(uuid1())

        if self.vector_store_name == 'faiss':
            vector_store = FAISS.from_documents(chunks, embedding=self.embeddings_model)
            vector_store.save_local(self.vectorstore_dir) 

            
                
        elif self.vector_store_name == 'chroma':
            vector_store = Chroma.from_documents(chunks, embedding=self.embeddings_model, persist_directory=self.vectorstore_dir)
            
        return vector_store

    def delete_doc_from_vstore(self,name_of_deletion):
        def faiss_to_df(store): # convert vectore store to dataframe to easily get target ids
                store_dict = store.docstore._dict
                dict_rows = []
                for key in store_dict.keys():
                    name = store_dict[key].metadata['name']
                    dict_rows.append({'id':key,'name':name})
            
                return pd.DataFrame(dict_rows) # df with ids and paths
            
        def delete_doc_from_faiss(store,name_of_deletion): # deletes in place
            vector_df = faiss_to_df(store)
            ids_to_delete = vector_df.loc[vector_df['name']==name_of_deletion]['id'].tolist() # list of ids to delete, only where id matches to deletion path
            store.delete(ids_to_delete)
            return ids_to_delete

        def delete_doc_from_chroma(chroma_store,name_of_deletion):
            def extract_name(metadata):
                return metadata.get('name')
        
            vector_df = pd.DataFrame(chroma_store.get())
            vector_df['name'] = vector_df['metadatas'].apply(lambda x: extract_name(x))
            ids_to_delete = vector_df.loc[vector_df['name']==name_of_deletion]['ids'].tolist() # list of ids to delete, only where id matches to deletion path
            chroma_store._collection.delete(ids = ids_to_delete) 
            return ids_to_delete

        if self.vector_store_name == 'faiss':
            delete_doc_from_faiss(self.vector_store,name_of_deletion)
            self.vector_store.save_local(self.vectorstore_dir)
        elif self.vector_store_name == 'chroma':
            delete_doc_from_chroma(self.vector_store,name_of_deletion)
            self.vector_store.persist()
        print('Deleted Docs')

    def add_docs_to_vstore(self):
            
        new_docs_dir = os.path.join(self.load_chatbot_from_dir, 'docs', 'tmp')
        new_pdfs = [os.path.join(new_docs_dir, f) for f in os.listdir(new_docs_dir) if f.endswith('.txt') or f.endswith('.pdf') or f.endswith('.docx')]
        # print(new_pdfs)
        old_docs_dir = os.path.join(self.load_chatbot_from_dir, 'docs')
        old_pdfs = [os.path.join(old_docs_dir, f) for f in os.listdir(old_docs_dir) if f.endswith('.txt') or f.endswith('.pdf') or f.endswith('.docx')]
        # print(old_pdfs)
        new_pdfs = [f for f in new_pdfs if os.path.basename(f) not in [os.path.basename(x) for x in old_pdfs]]
        if new_pdfs != []:
            new_docs = self._read_pdf_and_make_chunks(new_pdfs)
    
            
            if self.vector_store_name == 'faiss':
                dc = FAISS.from_documents(new_docs,self.embeddings_model)
                self.vector_store.merge_from(dc)
                self.vector_store.save_local(self.vectorstore_dir)
            
        
            elif self.vector_store_name == 'chroma':
                self.vector_store.add_documents(new_docs)
                self.vector_store.persist()
    
            for doc_path in new_pdfs:
                # print(doc_path)
                print(self.load_chatbot_from_dir + '/docs/')
                try:
                    shutil.move(doc_path,self.load_chatbot_from_dir + '/docs/')
                    os.remove(doc_path)
                except:
                    pass
                
        print('Added Docs')
        
    
    def _get_retriever(self, vector_store, retriever_search_type, top_k, similarity_score_threshold, retriever_method, llm_model):

        if retriever_search_type == "Similarity score threshold":
            retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                                  search_kwargs={"score_threshold": similarity_score_threshold})
        elif retriever_search_type == "top_k":
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        elif retriever_search_type == "mmr":
            retriever = vector_store.as_retriever(search_type="mmr")

        if retriever_method == "MultiQueryRetriever":
            retriever = MultiQueryRetriever.from_llm(
                retriever=retriever, llm=llm_model
            )
        elif retriever_method =="Contextual compression":
            compressor = LLMChainExtractor.from_llm(llm_model)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )

        return retriever


    def _get_llm(self):
        if 'Google' in self.llm_model_name:
            llm = GoogleGenerativeAI(model="gemini-pro")
        else:
            if self.openai_api_key:
                llm = ChatOpenAI(temperature=0,
                                max_tokens=4000,
                                model=self.llm_model_name,
                                request_timeout=120,
                                openai_api_key=self.openai_api_key)
            else:
                llm = ChatOpenAI(temperature=0,
                                max_tokens=4000,
                                model=self.llm_model_name,
                                request_timeout=120)
        return llm


    def _get_memory(self):
        if self.memory_type == 'ConversationBufferMemory':
            memory = ConversationBufferMemory(memory_key='chat_history', input_key='query')
        elif self.memory_type == 'ConversationBufferWindowMemory':
            memory = ConversationBufferWindowMemory(memory_key='chat_history', input_key='query', k=self.memory_last_k)
        elif self.memory_type == 'ConversationTokenBufferMemory':
            memory = ConversationTokenBufferMemory(llm=self.llm, memory_key='chat_history', input_key='query', max_token_limit=self.memory_max_tokens)
        elif self.memory_type == 'ConversationSummaryMemory':
            memory = ConversationSummaryMemory(llm=self.llm, memory_key='chat_history', input_key='query')

        return memory


    def clear_memory(self):
        self.memory.clear()


    def save_docs_embeddings(self, dir_path):
        shutil.copytree(self.vectorstore_dir, dir_path)


    @staticmethod
    def load_chatbot(dir_path, api_key=None):
        with open(dir_path+'/settings.json') as f:
            settings = json.load(f)

        # Pass the API key if it exists
        if api_key:
            settings['openai_api_key'] = api_key
        elif 'openai_api_key' in settings:
            # Use the stored API key if already present
            pass

        print('Chatbot Initialized')
        chatbot = Chatbot(**settings, load_chatbot_from_dir=dir_path)
        return chatbot


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    chatbot = Chatbot(
        documents=['docs/example.pdf'],
        verbose=False
    )

    chatbot.query('what is the total number of AI publications?')
    chatbot.query('What is this number divided by 2?')