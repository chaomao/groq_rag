import os
import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_groq import ChatGroq

# embed_model = "text-embedding-3-small"
embed_model = "text-embedding-ada-002"

embed = OpenAIEmbeddings(
    model=embed_model,
    openai_api_key=os.environ['OPENAI_API_KEY']
)

groq_api_key = os.environ['GROQ_API_KEY']
os.environ["LANGCHAIN_PROJECT"] = "groq-chat"

st.title("Chat with Docs - Groq Edition :)")

if "vector" not in st.session_state:
    st.session_state.embeddings = embed
    st.session_state['loader'] = WebBaseLoader("https://en.wikipedia.org/wiki/OpenAI")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)


llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")
prompt = ChatPromptTemplate.from_template('''
    Answer question based on provided context.
    I will tip you $200 if you are correct.
    <context>
    {context}
    </context>
    
    Question: {input}''')

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector.as_retriever()
retrieve_chain = create_retrieval_chain(retriever, document_chain)
prompt = st.text_input("Ask a question about the document:")

if prompt:
    start = time.process_time()
    response = retrieve_chain.invoke({"input": prompt})
    print(f"Groq response time is {time.process_time() - start}")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("==============")
