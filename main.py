import os
import glob
import re
import time
import yaml
import json
import shutil
import logging
import streamlit as st
from streamlit_modal import Modal
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
import constants
from chatbot import Chatbot


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists('tmp'):
    os.makedirs('tmp',exist_ok=True)


def streamlit_ui():
    def create_streamlit_chat_history(history, scores):
        for doc, score in zip(history, scores):
            score = round(score*100)
            to_display = re.sub(r'\n{2,}', '\n', doc.page_content)
            to_display = re.sub(r' {2,}',' ',to_display)
            with st.expander(f"{os.path.basename(doc.metadata['name'])}        `(Section Number {doc.metadata['Section']})`", expanded=False):
                st.markdown(f"</br>"
                            f"Similarity score: {score}%\n"
                            f"```{to_display}",                          
                            unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state['chat_history'] = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.set_page_config("Bot-in-a-Box", layout='centered')

    save_modal = Modal(
        "Save chatbot", 
        key="save_chatbot",
        max_width=500,
    )

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    _, _, username = authenticator.login()

    if st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')
        st.session_state['chat_history'] = []
        try:
            del st.session_state['chatbot']
        except:
            pass
    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
        st.session_state['chat_history'] = []
        try:
            del st.session_state['chatbot']
        except:
            pass
    elif st.session_state["authentication_status"]:
        with st.sidebar:
            if username == 'admin':
                st.header('Hello Admin')
                st.subheader('Create new chatbot')
            else:
                st.header(username.title())
                if "file_uploader_key" not in st.session_state:
                    st.session_state["file_uploader_key"] = 0
                    
                pdf_docs = st.file_uploader("Add more files to your knowledge base", key = st.session_state["file_uploader_key"] ,accept_multiple_files=True, type=["pdf",'docx','txt'])
                if pdf_docs:
                    for pdf_doc in pdf_docs:
                        os.makedirs(f'chatbots/{username}/docs/tmp', exist_ok=True)

                        new_name = f'chatbots/{username}/docs/tmp/{pdf_doc.name}'
                        with open(new_name, 'wb') as tmp_file:
                            tmp_file.write(pdf_doc.read())
                            
                if st.button('Save Docs',disabled = False if pdf_docs else True  ):
                    st.session_state['chatbot'].add_docs_to_vstore()
                    st.session_state["file_uploader_key"] += 1
                    st.rerun()
                # st.write('Knowledge Base')
                with st.container(height=300):
                    st.write('Knowledge Base')
                    knowledge_base_root = f'./chatbots/{username}/docs'
                    knowledge_base = [os.path.join(knowledge_base_root, f) for f in os.listdir(knowledge_base_root) if f.endswith('.txt') or f.endswith('.pdf') or f.endswith('.docx')]
                    for doc_name in knowledge_base:
                        knowledge_base_column = st.columns([2,1])
                        with knowledge_base_column[1]:
                            key=time.time()
                            if st.button('🗑️',key=os.path.basename(doc_name)):
                                try:
                                    st.session_state['chatbot'].delete_doc_from_vstore(f'tmp/{os.path.basename(doc_name)}')
                                except:
                                    st.session_state['chatbot'].delete_doc_from_vstore(f'chatbots/{username}/docs/tmp/{os.path.basename(doc_name)}')
                                print(f'./chatbots/{username}/docs/{os.path.basename(doc_name)}')
                                os.remove(f'./chatbots/{username}/docs/{os.path.basename(doc_name)}')
                                st.rerun()
                        with knowledge_base_column[0]:
                            st.caption(f"- `{os.path.basename(doc_name)}`")
                                
            authenticator.logout('Logout', 'main')

            if username != "admin":
                if 'chatbot' not in st.session_state:
                    st.session_state['chatbot'] = Chatbot.load_chatbot(f'chatbots/{username}')
                if st.button("Clear chat history"):
                    st.session_state['chat_history'] = []
            else:
                uploaded_files = []
               
                pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf",'docx','txt'])

                if pdf_docs:
                    for pdf_doc in pdf_docs:
                        new_name = f'tmp/{pdf_doc.name}'
                        uploaded_files.append(new_name)
                        with open(new_name, 'wb') as tmp_file:
                            tmp_file.write(pdf_doc.read())

                st.title("Settings")
                with st.expander('Data preparation', expanded=False):
                    chunk_size = constants.CHUNK_SIZE
                    chunk_overlap = constants.CHUNK_OVERLAP

                    splitter = st.selectbox('Text Splitter', constants.TEXT_SPLITTERS, index=0)

                    if splitter != "Semantic Splitter":
                        chunk_size = st.number_input("Chunk Size", value=chunk_size)
                        chunk_overlap = st.number_input("Chunk Overlap", value=chunk_overlap)
                    embedding_model = st.selectbox('Embedding Model', constants.EMBEDDING_MODELS, index=0)
                    vectorstore = st.radio("Vector store", constants.VECTOR_STORES, index=0)


                with st.expander('Retrieval', expanded=False):
                    retriever_top_k = constants.DEFAULT_RETRIEVER_TOP_K
                    sim_score_threshold = constants.DEFAULT_RETRIEVER_SIMILARITY_SCORE_THRESHOLD

                    prompt_for_rephrase_query = st.text_area("Prompt for rephrase query", constants.PROMPT_FOR_REPHRASE_QUERY, height=200)
                    retriever_method = st.selectbox('Retrieval method', constants.RETRIEVAL_METHOD, index=0)
                    retrieval_search_type = st.radio("Retrieval search type", constants.RETRIEVAL_SEARCH_TYPE, index=0)
                    if retrieval_search_type == "Similarity score threshold":
                        sim_score_threshold = st.slider("Similarity score Threshold", 0.0, 1.0, sim_score_threshold)
                    elif retrieval_search_type == "top_k":
                        retriever_top_k = st.number_input("Top K", value=retriever_top_k)

                with st.expander('Generation', expanded=False):
                    memory_last_k = constants.MEMORY_LAST_K
                    memory_max_tokens = constants.MEMORY_MAX_TOKENS

                    prompt_for_generation = st.text_area("Prompt for generation", constants.PROMPT_FOR_GENERATION, height=200)
                    memory_type = st.selectbox('Memory type', constants.MEMORY_TYPES, index=0)
                    if memory_type == 'ConversationBufferWindowMemory':
                        memory_last_k = st.number_input("Memory last K", value=memory_last_k)
                    elif memory_type == 'ConversationTokenBufferMemory':
                        memory_max_tokens = st.number_input("Memory max tokens", value=memory_max_tokens)
                    llm_name = st.selectbox('Choose LLM', constants.LLMS, index=0)

                sidebar_cols = st.columns([1, 1])
                with sidebar_cols[0]:
                    if st.button("Process", use_container_width=True):
                        if not pdf_docs:
                            st.toast("Please upload PDF files first.", icon='⚠️')
                        else:
                            with st.spinner("Processing"):
                                if not os.path.exists('tmp'):
                                    os.makedirs('tmp')

                                st.session_state['chatbot'] = Chatbot(
                                    documents=uploaded_files,
                                    text_splitter=splitter,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    embedding_model=embedding_model,
                                    vector_store=vectorstore,
                                    prompt_for_rephrase_query=prompt_for_rephrase_query,
                                    prompt_for_rephrase_query_variables=constants.PROMPT_FOR_REPHRASE_QUERY_VARIABLES,
                                    prompt_for_summary = constants.PROMPT_FOR_SUMMARY,
                                    prompt_for_summary_variables = constants.PROMPT_FOR_SUMMARY_VARIABLES,
                                    retriever_method=retriever_method,
                                    retriever_search_type=retrieval_search_type,
                                    retriever_top_k=retriever_top_k,
                                    retriever_similarity_score_threshold=sim_score_threshold,
                                    prompt_for_generation=prompt_for_generation,
                                    prompt_for_generation_variables=constants.PROMPT_FOR_GENERATION_VARIABLES,
                                    memory_type=memory_type,
                                    memory_last_k=memory_last_k,
                                    memory_max_tokens=memory_max_tokens,
                                    llm_model=llm_name,
                                    verbose=False,
                                )
                                st.session_state['chat_history'] = []
                                st.toast("Done! You can now chat with AI.")

                with sidebar_cols[1]:
                    if st.button("Clear chat history", use_container_width=True):
                        st.session_state['chat_history'] = []
                        st.session_state['chatbot'].clear_memory()
                if st.button("Save chatbot", use_container_width=True):
                    if not pdf_docs:
                        st.toast("Please upload PDF files first.", icon='⚠️')
                    else:
                        save_modal.open()

                st.caption('_____')

                with st.expander('Saved Chatbots',expanded = False):
                    for bot_name in [i for i in os.listdir('./chatbots/') if i[0]!='.']:
                        st.write(f'- {bot_name}')
                    


        if 'chatbot' in st.session_state:
            if username == 'sunnyx':
                head_columns = st.columns([2,1])

                with head_columns[0]:
                    st.header("SunnyX AI Solar Companion")
                with head_columns[1]:
                    st.write(
                        """<style>
                        [data-testid="stHorizontalBlock"] {
                            align-items: center;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.image('./metasolar.png',width = 80)
            else:
                head_columns = st.columns([1,2.5])
                with head_columns[0]:
                    st.header("Bot in a Box")
                with head_columns[1]:
                    st.write(
                        """<style>
                        [data-testid="stHorizontalBlock"] {
                            align-items: center;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    st.image('./botbox.png',width = 60)
        else:
            st.header("Before you start the chat, please upload your PDF files in the sidebar and press the process button.")

        for msg in st.session_state['chat_history']:
            try:
                st.chat_message(msg["role"], avatar=f'logos/{username}.png' if msg['role'] == 'assistant' else None).write(msg["content"])
            except:
                st.chat_message(msg["role"]).write(msg["content"])
            if "docs" in msg:
                create_streamlit_chat_history(msg["docs"], msg["scores"])

        user_query = st.chat_input(disabled='chatbot' not in st.session_state)
        if user_query:
            st.session_state['chat_history'].append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            logger.info(f"Query: {user_query}")
            answer, docs, scores = st.session_state['chatbot'].query(user_query)

            # answer, docs, scores,cb = st.session_state['chatbot'].query(user_query)
            # try:
            #     with open(f'chatbots/{username}/{st.session_state["chatbot"].memory_filename}' ,'r') as file:
            #         chat = json.load(file)
            # except:
            #     chat = {'User':[],'Response':[],'Context':[],'Cost': [] }
            # chat['User'].append(user_query)
            # chat['Response'].append(answer)
            # chat['Context'].append({f'{os.path.basename(i.metadata["name"])} Section {i.metadata["Section"]}': i.page_content for i in docs})
            # chat['Cost'].append(f'Total Tokens {cb.total_tokens} ({cb.prompt_tokens} for input and rest for completion)  Total Cost ${cb.total_cost}')
            # try:
            #     with open( f'chatbots/{username}/{st.session_state["chatbot"].memory_filename}' ,'w') as file:
            #         json.dump(chat,file)
            # except:
            #     with open(st.session_state['chatbot'].memory_filename,'w') as file:
            #         json.dump(chat,file)
                
            st.session_state['chat_history'].append({"role": "assistant", "content": answer, "docs": docs, "scores": scores})
            logger.info(f"Response: {answer}")
            try:
                st.chat_message("assistant",avatar = f'logos/{username}.png').write(answer)
            except:
                st.chat_message("assistant").write(answer)

                
            create_streamlit_chat_history(docs, scores)

        def save_chatbot(name, passowrd, **settings):
            if os.path.exists(f'chatbots/{name}'):
                st.error(f"Chatbot with name **{name}** already exists.")
                return
            os.makedirs(f'chatbots/{name}/docs')

            with open(f'chatbots/{name}/settings.json', 'w') as f:
                json.dump(settings, f, indent=4)

            for f in settings['documents']:
                shutil.copy2(f, f'chatbots/{name}/docs/{os.path.basename(f)}')

            if 'chatbot' not in st.session_state:
                st.session_state['chatbot'] = Chatbot(**settings)
            st.session_state['chatbot'].save_docs_embeddings(f'chatbots/{name}/vectorstore')

            with open('config.yaml') as file:
                tmp_config = yaml.load(file, Loader=SafeLoader)
            tmp_config['credentials']['usernames'][name] = {'name': name, 'password': passowrd}
            with open('config.yaml', 'w') as f:
                yaml.dump(tmp_config, f, default_flow_style=False, sort_keys=False)


            save_modal.close()
            st.toast(f"Chatbot **{name}** has been saved successfully.", icon='🎉')

        if save_modal.is_open():
            with save_modal.container():
                chatbot_name = st.text_input("Chatbot name")
                chatbot_password = st.text_input("Password", type="password")
                save_button = st.button("Save", use_container_width=True)
                if save_button:
                    if not chatbot_name or not chatbot_password:
                        st.error("Please fill chatbot name and password.")
                    else:
                        with st.spinner("Saving chatbot"):
                            save_chatbot(
                                chatbot_name,
                                chatbot_password,
                                documents=uploaded_files,
                                text_splitter=splitter,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                embedding_model=embedding_model,
                                vector_store=vectorstore,
                                prompt_for_rephrase_query=prompt_for_rephrase_query,
                                prompt_for_rephrase_query_variables=constants.PROMPT_FOR_REPHRASE_QUERY_VARIABLES,
                                prompt_for_summary = constants.PROMPT_FOR_SUMMARY,
                                prompt_for_summary_variables = constants.PROMPT_FOR_SUMMARY_VARIABLES,
                                retriever_method=retriever_method,
                                retriever_search_type=retrieval_search_type,
                                retriever_top_k=retriever_top_k,
                                retriever_similarity_score_threshold=sim_score_threshold,
                                prompt_for_generation=prompt_for_generation,
                                prompt_for_generation_variables=constants.PROMPT_FOR_GENERATION_VARIABLES,
                                memory_type=memory_type,
                                memory_last_k=memory_last_k,
                                memory_max_tokens=memory_max_tokens,
                                llm_model=llm_name,
                                verbose=False,
                            )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    streamlit_ui()