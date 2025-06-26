# bot_core.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from private_params import *

os.environ["OPENAI_API_KEY"] = openai_key

embedding_model = OpenAIEmbeddings()
session_histories = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    return session_histories[session_id]

def load_all_documents(folder="uploaded_files"):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, fname))
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs.extend(text_splitter.split_documents(pages))
    return docs

def build_chain():
    docs = load_all_documents()
    vectorstore = Chroma.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context if relevant."),
        MessagesPlaceholder(variable_name="history"),
        ("system", "Context:\n{context}"),
        ("human", "{input}")
    ])

    llm = ChatOpenAI()
    chain = RunnableWithMessageHistory(
        runnable=prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    return chain, retriever
