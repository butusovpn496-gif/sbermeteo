import streamlit as st

from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings

auth = st.secrets["gigachat_api_key"]
embeddings = GigaChatEmbeddings(
    credentials=auth,
    verify_ssl_certs=False
)

from langchain_gigachat import GigaChat

llm = GigaChat(
            credentials=auth,
            model='GigaChat',
            verify_ssl_certs=False,
            profanity_check=False
            )

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("meteo_rag.pdf")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Индексируем чанки
_ = vector_store.add_documents(documents=all_splits)



def generate_response(question):

    from langchain_core.prompts import ChatPromptTemplate

    retrieved_docs = vector_store.similarity_search(question)

    context = '\n'.join([doc.page_content for doc in retrieved_docs])

    prompt_template = ChatPromptTemplate([
        ("system", "Ты — полезный помощник для вопросно ответных приложений. Используй следующий контекст, чтобы ответить на вопрос. "
                "Если ответа нет в контексте — скажи, что не знаешь.\n\nКонтекст:\n {context}"),
        ("user", "{question}")
    ])

    chain = prompt_template | llm

    response = chain.invoke({
        "question": question,
        "context": context,
    })
 
    return response

result = ""

with st.form(key='qa_form', clear_on_submit=True, border=True):
    query_text = st.text_input(
    'Отправьте свой вопрос LLM:',
    placeholder='Здесь нужно написать вопрос',
)
    submitted = st.form_submit_button("Отправить")

    if submitted:
        with st.spinner('Calculating...'):
            # Генерируем ответ с помощью функции
            response = generate_response(query_text)
            result = response

# Отображаем результат, если он есть
if result:
    st.info(result.content)
